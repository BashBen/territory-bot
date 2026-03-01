"""Main terrain generation and processing logic."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from game.constants import ISLAND_GRID_SIZE, TARGET_ISLAND_COUNT

WATER = np.uint8(0)
LAND = np.uint8(1)
DEFAULT_LAND_COVERAGE = 0.62
MAX_REGION_FILL = 0.86


def generate_terrain_grid(
    seed: int | None = None, land_coverage: float = DEFAULT_LAND_COVERAGE
) -> np.ndarray:
    """Generate a 0/1 terrain grid with archipelago-style islands.

    The output is a numpy array where:
    - 0 means water
    - 1 means land
    """

    width, height = _grid_dimensions(ISLAND_GRID_SIZE)
    island_count = max(1, int(TARGET_ISLAND_COUNT))
    land_coverage = float(np.clip(land_coverage, 0.10, 0.90))

    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width), dtype=np.float32)
    centers = _archipelago_centers(width, height, island_count, rng)
    regions = _voronoi_regions(x, y, centers)

    weights = rng.uniform(0.85, 1.15, size=island_count)
    weights /= weights.sum()
    total_land_pixels = int(width * height * land_coverage)

    grid = np.full((height, width), WATER, dtype=np.uint8)

    for island_id, (center_x, center_y) in enumerate(centers):
        region_mask = regions == island_id
        if not np.any(region_mask):
            continue

        usable_region = _erode_mask(region_mask, iterations=3)
        if usable_region.sum() < 16:
            usable_region = region_mask

        target_pixels = int(total_land_pixels * weights[island_id])
        max_pixels_in_region = int(usable_region.sum() * MAX_REGION_FILL)
        target_pixels = max(1, min(target_pixels, max_pixels_in_region))

        island_field = _island_field(
            x=x,
            y=y,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            rng=rng,
        )
        region_values = island_field[usable_region]

        # Keep the strongest values in this Voronoi region to make one large island.
        kth = max(region_values.size - target_pixels, 0)
        threshold = np.partition(region_values, kth)[kth]
        island_mask = (island_field >= threshold) & usable_region
        island_mask = _smooth_mask(island_mask, usable_region)

        grid[island_mask] = LAND

    return grid


def _grid_dimensions(grid_size: Sequence[int]) -> tuple[int, int]:
    if len(grid_size) != 2:
        raise ValueError("ISLAND_GRID_SIZE must contain [width, height].")

    width = int(grid_size[0])
    height = int(grid_size[1])

    if width <= 0 or height <= 0:
        raise ValueError("Grid size values must be positive.")

    return width, height


def _archipelago_centers(
    width: int, height: int, island_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Place island centers on a curved band for an archipelago look."""
    if island_count == 1:
        return np.array([[width * 0.5, height * 0.5]], dtype=np.float32)

    t = np.linspace(-1.0, 1.0, island_count, dtype=np.float32)
    center_x = width * (0.5 + 0.33 * t)
    center_y = height * (0.56 - 0.18 * (1.0 - t * t))

    center_x += rng.uniform(-0.035, 0.035, size=island_count) * width
    center_y += rng.uniform(-0.03, 0.03, size=island_count) * height

    margin_x = width * 0.12
    margin_y = height * 0.12
    center_x = np.clip(center_x, margin_x, width - margin_x)
    center_y = np.clip(center_y, margin_y, height - margin_y)

    return np.column_stack((center_x, center_y)).astype(np.float32)


def _voronoi_regions(x: np.ndarray, y: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distance_stack = np.stack(
        [(x - cx) ** 2 + (y - cy) ** 2 for cx, cy in centers],
        axis=0,
    )
    return np.argmin(distance_stack, axis=0)


def _island_field(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a smooth scalar field; high values become land."""
    dx = x - center_x
    dy = y - center_y
    theta = np.arctan2(dy, dx)

    base_radius_x = width * rng.uniform(0.20, 0.27)
    base_radius_y = height * rng.uniform(0.17, 0.24)

    warp = (
        1.0
        + 0.22 * np.sin(3.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
        + 0.11 * np.sin(5.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
        + 0.07 * np.sin(7.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
    )
    warp = np.clip(warp, 0.55, None)

    radial_distance = np.sqrt(
        (dx / (base_radius_x * warp)) ** 2 + (dy / (base_radius_y * warp)) ** 2
    )
    field = 1.0 - radial_distance

    # Add a few positive bumps to create peninsulas and uneven coastlines.
    for _ in range(3):
        bump_x = center_x + rng.uniform(-0.45, 0.45) * base_radius_x
        bump_y = center_y + rng.uniform(-0.45, 0.45) * base_radius_y
        bump_scale_x = base_radius_x * rng.uniform(0.20, 0.45)
        bump_scale_y = base_radius_y * rng.uniform(0.20, 0.45)

        bump = np.exp(
            -(((x - bump_x) / bump_scale_x) ** 2 + ((y - bump_y) / bump_scale_y) ** 2)
        )
        field += bump * rng.uniform(0.15, 0.32)

    # Carve a couple of shallow bays so islands do not look circular.
    for _ in range(2):
        bay_x = center_x + rng.uniform(-0.65, 0.65) * base_radius_x
        bay_y = center_y + rng.uniform(-0.65, 0.65) * base_radius_y
        bay_scale_x = base_radius_x * rng.uniform(0.25, 0.50)
        bay_scale_y = base_radius_y * rng.uniform(0.25, 0.50)

        bay = np.exp(
            -(((x - bay_x) / bay_scale_x) ** 2 + ((y - bay_y) / bay_scale_y) ** 2)
        )
        field -= bay * rng.uniform(0.08, 0.18)

    return field


def _smooth_mask(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    """One smoothing pass to remove tiny jagged fragments."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    neighborhood = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    smoothed = neighborhood >= 5
    return smoothed & region


def _erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary erosion using a 3x3 kernel to keep water channels between islands."""
    eroded = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(eroded.astype(np.uint8), 1, mode="constant", constant_values=0)
        neighborhood = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        eroded = neighborhood == 9
    return eroded
