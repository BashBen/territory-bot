"""Game event payloads emitted by tick processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class PlayerGameOverEvent:
    """A player has been fully conquered and no longer owns land."""

    tick: int
    player_id: int
    type: str = "player_game_over"


@dataclass(frozen=True, slots=True)
class GameWonEvent:
    """A player has reached the occupation threshold and won."""

    tick: int
    player_id: int
    occupation_fraction: float
    type: str = "game_won"


GameEvent: TypeAlias = PlayerGameOverEvent | GameWonEvent
