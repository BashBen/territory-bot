"""Abstract interfaces for pluggable action handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from game.actions.payloads import ActionPayload
from game.player import Player


class ActionHandler(ABC):
    """Base interface for one action domain (attack, trade, etc.)."""

    @abstractmethod
    def can_handle(self, *, action_type: str) -> bool:
        """Return True when this handler supports the action type."""

    @abstractmethod
    def queue_action(
        self,
        *,
        game_map: np.ndarray,
        players: dict[int, Player],
        player_id: int,
        payload: ActionPayload,
    ) -> bool:
        """Validate and queue one action payload for later execution."""

    @abstractmethod
    def tick(self, *, game_map: np.ndarray, players: dict[int, Player]) -> None:
        """Advance this handler by one game tick."""
