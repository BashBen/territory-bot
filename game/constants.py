"""Shared game constants."""

# --------------------------------
# Terrain generation
# --------------------------------

# Number of large islands to generate in the base map.
TARGET_ISLAND_COUNT = 3

# Grid size as [width, height].
ISLAND_GRID_SIZE = [512, 512]

# --------------------------------
# Player and spawning
# --------------------------------

# Maximum number of players allowed in one game.
MAX_PLAYER_COUNT = 100

# First valid player ID (0 = water, 1 = unoccupied land).
FIRST_PLAYER_ID = 2

# Spawn claim radius (tiles) around the player's spawn point.
SPAWN_CLAIM_RADIUS = 5
