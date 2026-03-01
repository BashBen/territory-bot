"""Public action dispatch engine."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from game.actions.base import ActionHandler
from game.actions.attack import AttackEngine
from game.actions.payloads import ActionPayload, AttackPayload
from game.player import Player


class ActionEngine:
    """Routes public action calls to concrete action handlers."""

    def __init__(self, handlers: list[ActionHandler] | None = None) -> None:
        if handlers is None:
            handlers = [AttackEngine()]
        if not handlers:
            raise ValueError("ActionEngine requires at least one action handler.")

        self._handlers = list(handlers)

    def tick(self, *, game_map: np.ndarray, players: dict[int, Player]) -> None:
        """Advance all in-progress action handlers by one game tick."""
        for handler in self._handlers:
            handler.tick(game_map=game_map, players=players)

    def attack(
        self,
        *,
        game_map: np.ndarray,
        players: dict[int, Player],
        player_id: int,
        payload: AttackPayload | Mapping[str, object],
    ) -> bool:
        """Queue an attack action for the next tick."""
        if not isinstance(payload, Mapping):
            return False

        normalized_payload = dict(payload)
        normalized_payload["type"] = "attack"
        return self._queue_by_type(
            action_type="attack",
            game_map=game_map,
            players=players,
            player_id=player_id,
            payload=normalized_payload,
        )

    def action(
        self,
        *,
        game_map: np.ndarray,
        players: dict[int, Player],
        player_id: int,
        payload: ActionPayload | Mapping[str, object],
    ) -> bool:
        """Queue one action for the next tick.

        Attack is currently the only supported action type.
        """
        if not isinstance(payload, Mapping):
            return False

        action_type = str(payload.get("type", "attack")).strip().lower()
        if not action_type:
            return False

        return self._queue_by_type(
            action_type=action_type,
            game_map=game_map,
            players=players,
            player_id=player_id,
            payload=payload,
        )

    def _queue_by_type(
        self,
        *,
        action_type: str,
        game_map: np.ndarray,
        players: dict[int, Player],
        player_id: int,
        payload: ActionPayload,
    ) -> bool:
        for handler in self._handlers:
            if not handler.can_handle(action_type=action_type):
                continue
            return handler.queue_action(
                game_map=game_map,
                players=players,
                player_id=player_id,
                payload=payload,
            )
        return False
