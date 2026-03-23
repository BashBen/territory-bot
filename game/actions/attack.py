"""Attack queueing and propagation logic."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from game.constants import (
    LAND_ATTACK_DEFENDED_TILE_COST,
    LAND_ATTACK_TAX_FRACTION,
    LAND_ATTACK_UNDEFENDED_TILE_COST,
)
from game.actions.base import ActionHandler
from game.actions.payloads import AttackPayload
from game.player import Player
from game.terrain import WATER

__all__ = ["AttackEngine"]


@dataclass(slots=True)
class _AttackIntent:
    """Queued request to start an attack."""

    attacker_id: int
    defender_id: int
    target_row: int
    target_col: int
    attack_fraction: float


@dataclass(slots=True)
class _ActiveAttack:
    """In-progress BFS wave attack that advances one layer per tick."""

    attacker_id: int
    defender_id: int
    remaining_attack_units: int
    defender_damage_budget_remaining: int
    component_mask: np.ndarray
    visited: np.ndarray
    frontier: list[tuple[int, int]]
    defender_damage_carry: float = 0.0


class AttackEngine(ActionHandler):
    """Public attack API with internal queued/active wave state."""

    def __init__(self) -> None:
        self._pending_actions: list[_AttackIntent] = []
        self._active_attacks: list[_ActiveAttack] = []

    def can_handle(self, *, action_type: str) -> bool:
        return action_type == "attack"

    def queue_action(
        self,
        *,
        game_map: np.ndarray,
        players: dict[int, Player],
        player_id: int,
        payload: AttackPayload | Mapping[str, object],
    ) -> bool:
        return self.attack(
            game_map=game_map,
            players=players,
            player_id=player_id,
            payload=payload,
        )

    def attack(
        self,
        *,
        game_map: np.ndarray,
        players: dict[int, Player],
        player_id: int,
        payload: AttackPayload | Mapping[str, object],
    ) -> bool:
        """Queue one attack action for the next tick."""
        return _queue_attack(
            game_map=game_map,
            players=players,
            active_attacks=self._active_attacks,
            pending_actions=self._pending_actions,
            player_id=player_id,
            payload=payload,
        )

    def tick(self, *, game_map: np.ndarray, players: dict[int, Player]) -> None:
        """Start queued attacks and advance active attacks by one layer."""
        _resolve_queued_actions(
            game_map=game_map,
            players=players,
            active_attacks=self._active_attacks,
            pending_actions=self._pending_actions,
        )
        _advance_active_attacks(
            game_map=game_map,
            players=players,
            active_attacks=self._active_attacks,
        )


def _queue_attack(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    active_attacks: list[_ActiveAttack],
    pending_actions: list[_AttackIntent],
    player_id: int,
    payload: AttackPayload | Mapping[str, object],
) -> bool:
    """Validate and queue one player action."""
    attacker = players.get(player_id)
    if attacker is None or not attacker.is_alive or not isinstance(payload, Mapping):
        return False

    action_type = str(payload.get("type", "attack")).strip().lower()
    if action_type != "attack":
        return False

    parsed = _parse_attack_payload(payload)
    if parsed is None:
        return False

    target_row, target_col, attack_fraction = parsed
    if not _in_bounds(game_map, target_row, target_col):
        return False

    target_owner = int(game_map[target_row, target_col])
    if target_owner == WATER or target_owner == player_id:
        return False

    _, component_tiles = _collect_connected_component(
        game_map=game_map,
        start_row=target_row,
        start_col=target_col,
        owner_id=target_owner,
    )
    border_tiles = _component_border_tiles_touching_player(
        game_map=game_map,
        component_tiles=component_tiles,
        player_id=player_id,
    )
    if not border_tiles:
        return False

    if target_owner >= 2 and _already_attacking_defender(
        attacker_id=player_id,
        defender_id=target_owner,
        active_attacks=active_attacks,
        pending_actions=pending_actions,
    ):
        return False

    pending_actions.append(
        _AttackIntent(
            attacker_id=player_id,
            defender_id=target_owner,
            target_row=target_row,
            target_col=target_col,
            attack_fraction=attack_fraction,
        )
    )
    return True


def _resolve_queued_actions(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    active_attacks: list[_ActiveAttack],
    pending_actions: list[_AttackIntent],
) -> None:
    """Start attacks from queued intents."""
    if not pending_actions:
        return

    queued = list(pending_actions)
    pending_actions.clear()

    for intent in queued:
        attack = _start_attack_from_intent(
            game_map=game_map,
            players=players,
            active_attacks=active_attacks,
            intent=intent,
        )
        if attack is not None:
            active_attacks.append(attack)


def _advance_active_attacks(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    active_attacks: list[_ActiveAttack],
) -> None:
    """Advance each active attack by one BFS layer."""
    if not active_attacks:
        return

    remaining: list[_ActiveAttack] = []
    for attack in active_attacks:
        still_active = _advance_single_attack_layer(
            game_map=game_map,
            players=players,
            attack=attack,
        )
        if still_active:
            remaining.append(attack)

    active_attacks[:] = remaining


def _start_attack_from_intent(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    active_attacks: list[_ActiveAttack],
    intent: _AttackIntent,
) -> _ActiveAttack | None:
    # Revalidate that the queued intent still makes sense at execution time.
    attacker = players.get(intent.attacker_id)
    if attacker is None or not attacker.is_alive:
        return None

    if not _in_bounds(game_map, intent.target_row, intent.target_col):
        return None

    target_owner = int(game_map[intent.target_row, intent.target_col])
    if target_owner in (WATER, intent.attacker_id):
        return None

    if target_owner != intent.defender_id:
        return None

    if target_owner >= 2 and _already_attacking_defender(
        attacker_id=intent.attacker_id,
        defender_id=target_owner,
        active_attacks=active_attacks,
        pending_actions=[],
    ):
        return None

    # Rebuild the intended connected component and confirm attacker contact.
    component_mask, component_tiles = _collect_connected_component(
        game_map=game_map,
        start_row=intent.target_row,
        start_col=intent.target_col,
        owner_id=target_owner,
    )
    border_tiles = _component_border_tiles_touching_player(
        game_map=game_map,
        component_tiles=component_tiles,
        player_id=intent.attacker_id,
    )
    if not border_tiles:
        return None

    # Reserve balance once up front and convert to taxed attack units.
    investment = int(round(attacker.balance * intent.attack_fraction))
    if attacker.balance <= 0 or investment <= 0:
        return None

    attacker.balance -= investment

    true_attack = investment * (1.0 - LAND_ATTACK_TAX_FRACTION)
    true_attack_units = max(0, int(round(true_attack)))
    if true_attack_units <= 0:
        return None

    # Seed the initial frontier from every touching border tile.
    seed_tiles = list(border_tiles)
    visited = np.zeros(game_map.shape, dtype=bool)
    for row, col in seed_tiles:
        visited[row, col] = True

    return _ActiveAttack(
        attacker_id=intent.attacker_id,
        defender_id=target_owner,
        remaining_attack_units=true_attack_units,
        defender_damage_budget_remaining=max(0, int(round(true_attack_units / 2.0))),
        component_mask=component_mask,
        visited=visited,
        frontier=list(seed_tiles),
    )


def _advance_single_attack_layer(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    attack: _ActiveAttack,
) -> bool:
    attacker = players.get(attack.attacker_id)
    if attacker is None or not attacker.is_alive:
        return False

    if attack.remaining_attack_units <= 0 or not attack.frontier:
        return False

    next_frontier: list[tuple[int, int]] = []
    spent_vs_defender_units = 0

    for row, col in attack.frontier:
        if not attack.component_mask[row, col]:
            continue

        owner = int(game_map[row, col])
        if owner != attack.attacker_id and owner != WATER:
            tile_cost = _tile_cost_for_owner(owner=owner, players=players)
            if attack.remaining_attack_units >= tile_cost:
                attack.remaining_attack_units -= tile_cost
                game_map[row, col] = attack.attacker_id
                if owner == attack.defender_id:
                    spent_vs_defender_units += tile_cost

        # The wave only spreads through tiles already captured by attacker.
        if int(game_map[row, col]) != attack.attacker_id:
            continue

        for n_row, n_col in _neighbors4(game_map, row, col):
            if attack.visited[n_row, n_col]:
                continue
            if not attack.component_mask[n_row, n_col]:
                continue
            attack.visited[n_row, n_col] = True
            next_frontier.append((n_row, n_col))

    _apply_defender_balance_damage(
        players=players,
        attack=attack,
        spent_vs_defender_units=spent_vs_defender_units,
    )

    attack.frontier = next_frontier
    return attack.remaining_attack_units > 0 and bool(attack.frontier)


def _apply_defender_balance_damage(
    *,
    players: dict[int, Player],
    attack: _ActiveAttack,
    spent_vs_defender_units: int,
) -> None:
    if spent_vs_defender_units <= 0:
        return

    defender = players.get(attack.defender_id)
    if defender is None or not defender.is_alive or defender.balance <= 0:
        return

    if attack.defender_damage_budget_remaining <= 0:
        return

    raw_damage = attack.defender_damage_carry + (spent_vs_defender_units / 2.0)
    damage = int(raw_damage)
    attack.defender_damage_carry = raw_damage - damage

    damage = min(damage, attack.defender_damage_budget_remaining, defender.balance)
    if damage <= 0:
        return

    defender.balance -= damage
    attack.defender_damage_budget_remaining -= damage


def _tile_cost_for_owner(*, owner: int, players: dict[int, Player]) -> int:
    if owner <= 1:
        return LAND_ATTACK_UNDEFENDED_TILE_COST

    owner_player = players.get(owner)
    if owner_player is not None and owner_player.is_alive and owner_player.balance > 0:
        return LAND_ATTACK_DEFENDED_TILE_COST
    return LAND_ATTACK_UNDEFENDED_TILE_COST


def _already_attacking_defender(
    *,
    attacker_id: int,
    defender_id: int,
    active_attacks: list[_ActiveAttack],
    pending_actions: list[_AttackIntent],
) -> bool:
    for attack in active_attacks:
        if attack.attacker_id == attacker_id and attack.defender_id == defender_id:
            return True
    for intent in pending_actions:
        if intent.attacker_id == attacker_id and intent.defender_id == defender_id:
            return True
    return False


def _parse_attack_payload(
    payload: AttackPayload | Mapping[str, object],
) -> tuple[int, int, float] | None:
    target = payload.get("target")
    row: object
    col: object

    if isinstance(target, (list, tuple)) and len(target) == 2:
        row, col = target
    elif "row" in payload and "col" in payload:
        row = payload["row"]
        col = payload["col"]
    else:
        return None

    percentage = payload.get("percentage", payload.get("percent"))
    if percentage is None:
        return None

    attack_fraction = _normalize_percentage(percentage)
    if attack_fraction is None:
        return None

    try:
        target_row = int(row)
        target_col = int(col)
    except (TypeError, ValueError):
        return None

    return target_row, target_col, attack_fraction


def _normalize_percentage(value: object) -> float | None:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None

    if raw > 1.0:
        if raw > 100.0:
            return None
        raw = raw / 100.0

    if raw <= 0.0 or raw > 1.0:
        return None
    return raw


def _collect_connected_component(
    *,
    game_map: np.ndarray,
    start_row: int,
    start_col: int,
    owner_id: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Collect the 4-neighbor connected component for owner_id."""
    mask = np.zeros(game_map.shape, dtype=bool)
    tiles: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    mask[start_row, start_col] = True

    while queue:
        row, col = queue.popleft()
        tiles.append((row, col))
        for n_row, n_col in _neighbors4(game_map, row, col):
            if mask[n_row, n_col]:
                continue
            if int(game_map[n_row, n_col]) != owner_id:
                continue
            mask[n_row, n_col] = True
            queue.append((n_row, n_col))

    return mask, tiles


def _component_border_tiles_touching_player(
    *,
    game_map: np.ndarray,
    component_tiles: list[tuple[int, int]],
    player_id: int,
) -> list[tuple[int, int]]:
    """Return component tiles adjacent to the given player's territory."""
    border_tiles: list[tuple[int, int]] = []
    for row, col in component_tiles:
        for n_row, n_col in _neighbors4(game_map, row, col):
            if int(game_map[n_row, n_col]) == player_id:
                border_tiles.append((row, col))
                break
    return border_tiles


def _neighbors4(game_map: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    max_row, max_col = game_map.shape
    neighbors: list[tuple[int, int]] = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row + 1 < max_row:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col + 1 < max_col:
        neighbors.append((row, col + 1))
    return neighbors


def _in_bounds(game_map: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < game_map.shape[0] and 0 <= col < game_map.shape[1]
