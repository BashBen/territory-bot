"""Core game state and loop primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from game.constants import FIRST_PLAYER_ID, MAX_PLAYER_COUNT, SPAWN_CLAIM_RADIUS
from game.player import spawn_player
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
        self.players: dict[int, tuple[int, int]] = {}

        self._rng = np.random.default_rng(seed)
        self._next_player_id = FIRST_PLAYER_ID
        self._pending_actions: list[tuple[int, Any]] = []

    def tick(self) -> int:
        """Advance game time by one tick.

        For now this only increments the tick counter and clears queued actions.
        """
        self.tick_count += 1
        self._pending_actions.clear()
        return self.tick_count

    def action(self, player_id: int, payload: Any) -> bool:
        """Queue an action for a player.

        Returns:
        - True when action is accepted
        - False when the player does not exist
        """
        if player_id not in self.players:
            return False

        self._pending_actions.append((player_id, payload))
        return True

    def get_state(self) -> np.ndarray:
        """Return a copy of the current map state."""
        return self.map.copy()

    def add_player(self) -> int:
        """Create a new player and spawn on unoccupied land.

        Returns:
        - Player ID (starting at 2) when spawned successfully.
        - -1 when no unoccupied land remains.

        Raises:
        - ValueError when trying to exceed MAX_PLAYER_COUNT.
        """
        player_id, next_player_id = spawn_player(
            game_map=self.map,
            players=self.players,
            next_player_id=self._next_player_id,
            rng=self._rng,
            max_player_count=MAX_PLAYER_COUNT,
            claim_radius=SPAWN_CLAIM_RADIUS,
        )
        self._next_player_id = next_player_id
        return player_id
