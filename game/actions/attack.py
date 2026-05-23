"""Attack queueing and propagation logic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
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

# 4-neighbor dilation (one tile outward per tick).
_KERNEL_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


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
    """In-progress invasion of a frozen target region."""

    attacker_id: int
    defender_id: int
    remaining_attack_units: int
    defender_damage_budget_remaining: int
    component_mask: np.ndarray
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
        # Clash before expansion so reciprocal attacks compare strength first.
        _resolve_mutual_clashes(active_attacks=self._active_attacks)
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

    component_mask = _connected_component_mask(
        game_map=game_map,
        start_row=target_row,
        start_col=target_col,
        owner_id=target_owner,
    )
    if component_mask is None:
        return False

    if not _component_touches_player(
        game_map=game_map,
        component_mask=component_mask,
        player_id=player_id,
    ):
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


def _resolve_mutual_clashes(*, active_attacks: list[_ActiveAttack]) -> None:
    """Resolve reciprocal attack pairs before waves expand.

    Reciprocal A↔B waves never share target tiles (each invades the other's land), so
    clash is pair-level on remaining_attack_units, not tile intersection.
    """
    if len(active_attacks) < 2:
        return

    # At most one attack per direction; lets us find B→A given A→B in O(1).
    by_direction: dict[tuple[int, int], _ActiveAttack] = {
        (attack.attacker_id, attack.defender_id): attack for attack in active_attacks
    }
    # Player-pair ids: we see both (A, B) and (B, A) keys when iterating the dict.
    resolved_pairs: set[tuple[int, int]] = set()
    # Defer list mutation until the end; use id() because attacks are unhashable.
    removed_attack_ids: set[int] = set()

    for (attacker_id, defender_id), attack in by_direction.items():
        pair_key = (min(attacker_id, defender_id), max(attacker_id, defender_id))
        if pair_key in resolved_pairs:
            continue

        counter = by_direction.get((defender_id, attacker_id))
        if counter is None:
            continue

        resolved_pairs.add(pair_key)

        stronger, weaker = attack, counter
        stronger_strength = stronger.remaining_attack_units
        weaker_strength = weaker.remaining_attack_units
        if weaker_strength > stronger_strength:
            stronger, weaker = counter, attack
            stronger_strength, weaker_strength = weaker_strength, stronger_strength

        if stronger_strength == weaker_strength:
            # Full annihilation: neither wave expands this tick.
            removed_attack_ids.add(id(attack))
            removed_attack_ids.add(id(counter))
            continue

        # Weaker attack ends; stronger loses exactly the weaker's committed strength.
        removed_attack_ids.add(id(weaker))
        _apply_clash_cost(stronger, clash_units=weaker_strength)
        if stronger.remaining_attack_units <= 0:
            removed_attack_ids.add(id(stronger))

    if removed_attack_ids:
        active_attacks[:] = [
            attack for attack in active_attacks if id(attack) not in removed_attack_ids
        ]


def _apply_clash_cost(attack: _ActiveAttack, *, clash_units: int) -> None:
    # Mirror tile combat: clash spends combat power and caps economic pressure alike.
    attack.remaining_attack_units = max(0, attack.remaining_attack_units - clash_units)
    attack.defender_damage_budget_remaining = max(
        0, attack.defender_damage_budget_remaining - clash_units
    )


def _advance_active_attacks(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    active_attacks: list[_ActiveAttack],
) -> None:
    """Advance each active attack by one dilation step into its target region."""
    if not active_attacks:
        return

    cost_lookup = _build_tile_cost_lookup(game_map=game_map, players=players)
    remaining: list[_ActiveAttack] = []
    for attack in active_attacks:
        still_active = _advance_single_attack_layer(
            game_map=game_map,
            players=players,
            attack=attack,
            cost_lookup=cost_lookup,
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

    component_mask = _connected_component_mask(
        game_map=game_map,
        start_row=intent.target_row,
        start_col=intent.target_col,
        owner_id=target_owner,
    )
    if component_mask is None:
        return None

    if not _component_touches_player(
        game_map=game_map,
        component_mask=component_mask,
        player_id=intent.attacker_id,
    ):
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

    return _ActiveAttack(
        attacker_id=intent.attacker_id,
        defender_id=target_owner,
        remaining_attack_units=true_attack_units,
        defender_damage_budget_remaining=max(0, int(round(true_attack_units / 2.0))),
        component_mask=component_mask,
    )


def _advance_single_attack_layer(
    *,
    game_map: np.ndarray,
    players: dict[int, Player],
    attack: _ActiveAttack,
    cost_lookup: np.ndarray,
) -> bool:
    attacker = players.get(attack.attacker_id)
    if attacker is None or not attacker.is_alive:
        return False

    if attack.remaining_attack_units <= 0:
        return False

    capturable = _compute_capturable_mask(game_map=game_map, attack=attack)
    if not capturable.any():
        return False

    owners = game_map[capturable]
    tile_costs = cost_lookup[owners]
    total_cost = int(tile_costs.sum())
    if total_cost > attack.remaining_attack_units:
        # Cannot afford this ring — end attack without mutating the map.
        return False

    spent_vs_defender_units = int(tile_costs[owners == attack.defender_id].sum())

    game_map[capturable] = np.uint8(attack.attacker_id)
    attack.remaining_attack_units -= total_cost

    _apply_defender_balance_damage(
        players=players,
        attack=attack,
        spent_vs_defender_units=spent_vs_defender_units,
    )

    if attack.remaining_attack_units <= 0:
        return False

    return _compute_capturable_mask(game_map=game_map, attack=attack).any()


def _compute_capturable_mask(
    *, game_map: np.ndarray, attack: _ActiveAttack
) -> np.ndarray:
    """Tiles in the attack region one step outside attacker land (OpenCV dilate)."""
    player_land = (game_map == attack.attacker_id).astype(np.uint8)
    expanded = cv2.dilate(player_land, _KERNEL_4)

    capturable = (
        expanded.astype(bool)
        & attack.component_mask
        & (game_map != attack.attacker_id)
        & (game_map != WATER)
    )

    # Stall when another player's land also touches this tile (shared invasion front).
    other_players = ((game_map >= 2) & (game_map != attack.attacker_id)).astype(np.uint8)
    contested = cv2.dilate(other_players, _KERNEL_4).astype(bool)
    capturable &= ~contested

    return capturable


def _connected_component_mask(
    *,
    game_map: np.ndarray,
    start_row: int,
    start_col: int,
    owner_id: int,
) -> np.ndarray | None:
    """4-connected component mask for owner_id containing the start tile."""
    owner_mask = (game_map == owner_id).astype(np.uint8)
    if owner_mask[start_row, start_col] == 0:
        return None

    _, labels = cv2.connectedComponents(owner_mask, connectivity=4)
    label_id = int(labels[start_row, start_col])
    if label_id == 0:
        return None

    return labels == label_id


def _component_touches_player(
    *,
    game_map: np.ndarray,
    component_mask: np.ndarray,
    player_id: int,
) -> bool:
    player_land = (game_map == player_id).astype(np.uint8)
    expanded = cv2.dilate(player_land, _KERNEL_4)
    return bool(np.any(component_mask & expanded.astype(bool)))


def _build_tile_cost_lookup(
    *, game_map: np.ndarray, players: dict[int, Player]
) -> np.ndarray:
    """Per map-value capture cost; index by tile owner id from game_map."""
    max_owner = max(int(game_map.max()), max(players, default=1))
    lookup = np.full(max_owner + 1, LAND_ATTACK_UNDEFENDED_TILE_COST, dtype=np.int32)
    lookup[WATER] = 0

    for player_id, player in players.items():
        if player_id < 2:
            continue
        if player.is_alive and player.balance > 0:
            lookup[player_id] = LAND_ATTACK_DEFENDED_TILE_COST

    return lookup


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


def _in_bounds(game_map: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < game_map.shape[0] and 0 <= col < game_map.shape[1]
