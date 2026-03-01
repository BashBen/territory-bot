"""Shared game constants."""

# --------------------------------
# Terrain generation
# --------------------------------

# Number of large islands to generate in the base map.
TARGET_ISLAND_COUNT = 3
# This is the tiles in the grid.
# Tiles are the smallest piece of island the player can have.
ISLAND_GRID_SIZE = [512, 512]

# --------------------------------
# Player and spawning
# --------------------------------

MAX_PLAYER_COUNT = 100
# 0 = water, 1 = unoccupied land.
FIRST_PLAYER_ID = 2
# Amount of land you get when you spawn in. This will claim
# less if it's crowded by other players or the ocean.
SPAWN_CLAIM_RADIUS = 5

# --------------------------------
# Interest and balance
# --------------------------------

# Base territorial interest at 0 occupation.
BASE_TERRITORIAL_INTEREST = 0.01
# Extra territorial interest scaled by sqrt(occupation).
TERRITORIAL_SQRT_GAIN = 0.016

# Early-game augmentation starts here and linearly decays to MIN.
INITIAL_AUGMENTATION_MAX = 7.0
INITIAL_AUGMENTATION_MIN = 1.0
# Number of ticks for augmentation to decay from MAX to MIN.
INITIAL_AUGMENTATION_DURATION_TICKS = 214.0

# Balance limits are area-scaled: soft = 100*A, hard = 150*A.
SOFT_CAP_AREA_MULTIPLIER = 100
HARD_CAP_AREA_MULTIPLIER = 150

# --------------------------------
# Attack
# --------------------------------

# Land attack tax = A * LAND_ATTACK_TAX_FRACTION.
LAND_ATTACK_TAX_FRACTION = 12 / 1024

# Modeling assumptions for map capture budget.
# Defended cost is used only while a defending player still has balance.
LAND_ATTACK_UNDEFENDED_TILE_COST = 1
LAND_ATTACK_DEFENDED_TILE_COST = 2
