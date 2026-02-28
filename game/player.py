"""Player spawning helpers."""

from __future__ import annotations

import numpy as np

from game.terrain import LAND


def spawn_player(
    game_map: np.ndarray,
    players: dict[int, tuple[int, int]],
    next_player_id: int,
    rng: np.random.Generator | None = None,
    *,
    max_player_count: int,
    claim_radius: int,
) -> tuple[int, int]:
    """Spawn one player and claim surrounding land.

    Returns:
    - `(player_id, updated_next_player_id)` on success.
    - `(-1, next_player_id)` when no unoccupied land remains.

    Raises:
    - `ValueError` when `max_player_count` would be exceeded.
    """
    _ensure_capacity(player_count=len(players), max_player_count=max_player_count)

    if rng is None:
        rng = np.random.default_rng()

    spawn_cell = _choose_spawn_cell(game_map=game_map, rng=rng)
    if spawn_cell is None:
        return -1, next_player_id

    spawn_row, spawn_col = spawn_cell
    player_id = next_player_id

    game_map[spawn_row, spawn_col] = player_id
    _claim_radius(
        game_map=game_map,
        player_id=player_id,
        center_row=spawn_row,
        center_col=spawn_col,
        radius=claim_radius,
    )

    players[player_id] = (spawn_row, spawn_col)
    return player_id, next_player_id + 1


def _ensure_capacity(player_count: int, max_player_count: int) -> None:
    if player_count >= max_player_count:
        raise ValueError(f"Maximum player count reached ({max_player_count}).")


def _choose_spawn_cell(
    game_map: np.ndarray, rng: np.random.Generator
) -> tuple[int, int] | None:
    open_land = np.argwhere(game_map == LAND)
    if open_land.size == 0:
        return None

    spawn_index = int(rng.integers(0, open_land.shape[0]))
    spawn_row, spawn_col = open_land[spawn_index]
    return int(spawn_row), int(spawn_col)


def _claim_radius(
    game_map: np.ndarray,
    player_id: int,
    center_row: int,
    center_col: int,
    radius: int,
) -> None:
    """Claim unoccupied land tiles in a circular radius around a point."""
    row_start = max(0, center_row - radius)
    row_end = min(game_map.shape[0], center_row + radius + 1)
    col_start = max(0, center_col - radius)
    col_end = min(game_map.shape[1], center_col + radius + 1)

    local = game_map[row_start:row_end, col_start:col_end]
    row_coords = np.arange(row_start, row_end, dtype=np.int32)[:, None]
    col_coords = np.arange(col_start, col_end, dtype=np.int32)[None, :]

    distance_sq = (row_coords - center_row) ** 2 + (col_coords - center_col) ** 2
    in_radius = distance_sq <= (radius * radius)
    claimable = in_radius & (local == LAND)
    local[claimable] = player_id
