"""Backward-compatible action exports.

Prefer importing from ``game.actions``.
"""

from game.actions import (
    ActionEngine,
    ActionHandler,
    ActionPayload,
    AttackPayload,
)

# Legacy alias retained for compatibility.
AttackEngine = ActionEngine

__all__ = [
    "ActionEngine",
    "ActionHandler",
    "ActionPayload",
    "AttackPayload",
    "AttackEngine",
]
