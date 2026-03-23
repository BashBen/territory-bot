"""Core game state and loop primitives."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from game.actions import ActionEngine
from game.actions.payloads import ActionPayload, AttackPayload
from game.constants import FIRST_PLAYER_ID, MAX_PLAYER_COUNT, SPAWN_CLAIM_RADIUS
from game.interest import apply_interest
from game.player import Player, spawn_player
from game.terrain import generate_terrain_grid


class Game:
    """Main game container.

    Grid value semantics:
    - 0: water
    - 1: unoccupied land
    - >=2: land occupied by player with that ID
    """

    def __init__(self, seed: int | None = None, land_coverage: float = 0.62) -> None:
        self.map: np.ndarray = generate_terrain_grid(
            seed=seed, land_coverage=land_coverage
        ).astype(np.uint8, copy=False)
        self.tick_count = 0
        self.players: dict[int, Player] = {}

        self._rng = np.random.default_rng(seed)
        self._next_player_id = FIRST_PLAYER_ID
        self._occupiable_area = int(np.count_nonzero(self.map != 0))
        self._action_engine = ActionEngine()

    def tick(self) -> int:
        """Advance game time by one tick.

        Applies interest and advances queued/in-progress actions.
        """
        self._apply_interest_to_players(tick=self.tick_count)
        self._action_engine.tick(
            game_map=self.map,
            players=self.players,
        )
        self.tick_count += 1
        return self.tick_count

    def attack(
        self, player_id: int, payload: AttackPayload | Mapping[str, object]
    ) -> bool:
        """Queue an attack for the next tick."""
        if player_id not in self.players:
            return False

        return self._action_engine.attack(
            game_map=self.map,
            players=self.players,
            player_id=player_id,
            payload=payload,
        )

    def action(
        self, player_id: int, payload: ActionPayload | Mapping[str, object]
    ) -> bool:
        """Queue an action for the next tick.

        Returns:
        - True when action is queued
        - False when player is invalid or action preconditions fail
        """
        return self._action_engine.action(
            game_map=self.map,
            players=self.players,
            player_id=player_id,
            payload=payload,
        )

    def get_state(self, relative: int | None = None) -> np.ndarray:
        """Return a stacked state array with ownership and per-tile balances.

        Args:
        - `relative`: optional player ID to view as player `2`.
          When set, the returned ownership layer swaps that player's tiles with
          player `2`'s tiles. This keeps the caller's perspective normalized so
          player `2` always means "self". The balance layer is not remapped
          because it stores balances, not player IDs.

        Returns:
        - `state` with shape `(2, rows, cols)` where:
          - `state[0]` is the ownership map.
          - `state[1]` is the balance map.
        - The balance layer stores each owner's balance on their land tiles,
          with water and unoccupied land set to `0`.
        """
        ownership_map = self.map.copy()
        if relative is not None and relative != FIRST_PLAYER_ID:
            if relative < FIRST_PLAYER_ID:
                raise ValueError(
                    f"relative must be >= {FIRST_PLAYER_ID}, got {relative}."
                )

            relative_mask = ownership_map == relative
            first_player_mask = ownership_map == FIRST_PLAYER_ID
            ownership_map[relative_mask] = FIRST_PLAYER_ID
            ownership_map[first_player_mask] = relative

        max_id = max(int(self.map.max()), max(self.players, default=1))
        balance_by_id = np.zeros(max_id + 1, dtype=np.int64)
        for player_id, player in self.players.items():
            balance_by_id[player_id] = player.balance
        balance_map = balance_by_id[self.map]

        return np.stack((ownership_map.astype(np.int64, copy=False), balance_map))

    def add_player(self) -> int:
        """Create a new player and spawn on unoccupied land.

        Returns:
        - Player ID (starting at 2) when spawned successfully.
        - -1 when no unoccupied land remains.

        Raises:
        - ValueError when trying to exceed MAX_PLAYER_COUNT.
        """
        player_id = self._next_player_id
        spawned = spawn_player(
            game_map=self.map,
            players=self.players,
            player_id=player_id,
            rng=self._rng,
            max_player_count=MAX_PLAYER_COUNT,
            claim_radius=SPAWN_CLAIM_RADIUS,
        )
        if spawned:
            self._next_player_id += 1
            return player_id
        return -1

    def _apply_interest_to_players(self, tick: int) -> None:
        if not self.players:
            return

        max_player_id = max(self.players)
        owned_counts = np.bincount(self.map.ravel(), minlength=max_player_id + 1)
        for player_id, player in self.players.items():
            owned_area = int(owned_counts[player_id])
            new_balance, _, _ = apply_interest(
                balance=player.balance,
                owned_area=owned_area,
                occupiable_area=self._occupiable_area,
                tick=tick,
            )
            player.balance = new_balance
