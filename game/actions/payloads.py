"""Typed action payload definitions."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict


class AttackPayload(TypedDict, total=False):
    """Payload accepted by the attack action handler."""

    type: Literal["attack"]
    target: list[int] | tuple[int, int]
    row: int
    col: int
    percentage: float | int
    percent: float | int


# Discriminated-union shape (currently one variant).
ActionPayload: TypeAlias = AttackPayload
