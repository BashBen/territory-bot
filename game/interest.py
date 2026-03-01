"""Interest and balance calculations for territory players."""

from __future__ import annotations

import math

from game.constants import (
    BASE_TERRITORIAL_INTEREST,
    HARD_CAP_AREA_MULTIPLIER,
    INITIAL_AUGMENTATION_DURATION_TICKS,
    INITIAL_AUGMENTATION_MAX,
    INITIAL_AUGMENTATION_MIN,
    SOFT_CAP_AREA_MULTIPLIER,
    TERRITORIAL_SQRT_GAIN,
)


def territorial_interest_rate(owned_area: int, occupiable_area: int) -> float:
    """Per-tick territorial interest fraction from owned map share."""
    if occupiable_area <= 0 or owned_area <= 0:
        occ = 0.0
    else:
        occ = min(1.0, owned_area / occupiable_area)

    return BASE_TERRITORIAL_INTEREST + TERRITORIAL_SQRT_GAIN * math.sqrt(occ)


def initial_augmentation_factor(tick: int | float) -> float:
    """Early-game multiplier, linearly decaying from 7 to 1 over 214 ticks."""
    t = float(tick)
    if t <= 0.0:
        return INITIAL_AUGMENTATION_MAX
    if t >= INITIAL_AUGMENTATION_DURATION_TICKS:
        return INITIAL_AUGMENTATION_MIN

    progress = t / INITIAL_AUGMENTATION_DURATION_TICKS
    return INITIAL_AUGMENTATION_MIN + (
        INITIAL_AUGMENTATION_MAX - INITIAL_AUGMENTATION_MIN
    ) * (1.0 - progress)


def soft_cap(owned_area: int) -> int:
    """Soft balance cap = 100 * area."""
    area = max(0, int(owned_area))
    return SOFT_CAP_AREA_MULTIPLIER * area


def hard_cap(owned_area: int) -> int:
    """Hard balance cap = 150 * area."""
    area = max(0, int(owned_area))
    return HARD_CAP_AREA_MULTIPLIER * area


def balance_limiter(balance: int | float, owned_area: int) -> float:
    """Limiter in [0, 1] based on balance vs. soft/hard caps."""
    if owned_area <= 0:
        return 0.0

    current_balance = float(balance)
    soft = float(soft_cap(owned_area))
    hard = float(hard_cap(owned_area))

    if current_balance < soft:
        return 1.0
    if current_balance >= hard:
        return 0.0

    span = hard - soft
    if span <= 0.0:
        return 0.0

    return max(0.0, min(1.0, (hard - current_balance) / span))


def interest_rate_per_tick(
    *,
    balance: int | float,
    owned_area: int,
    occupiable_area: int,
    tick: int | float,
) -> float:
    """Total per-tick interest rate."""
    terr = territorial_interest_rate(owned_area=owned_area, occupiable_area=occupiable_area)
    aug = initial_augmentation_factor(tick)
    limit = balance_limiter(balance=balance, owned_area=owned_area)
    return terr * aug * limit


def apply_interest(
    *,
    balance: int,
    owned_area: int,
    occupiable_area: int,
    tick: int | float,
) -> tuple[int, int, float]:
    """Apply one interest step.

    Returns `(new_balance, delta, applied_rate)`.
    """
    current_balance = max(0, int(balance))
    cap = hard_cap(owned_area)
    if cap <= 0:
        return 0, 0, 0.0

    capped_balance = min(current_balance, cap)
    rate = interest_rate_per_tick(
        balance=capped_balance,
        owned_area=owned_area,
        occupiable_area=occupiable_area,
        tick=tick,
    )

    raw_gain = float(capped_balance) * max(0.0, rate)
    delta = max(0, int(round(raw_gain)))

    new_balance = min(cap, capped_balance + delta)
    return new_balance, new_balance - capped_balance, rate
