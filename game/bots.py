"""Bot player types and simple built-in policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from game.actions.payloads import ActionPayload
from game.interest import soft_cap
from game.player import Player

if TYPE_CHECKING:
    from game.core import Game


@dataclass(slots=True)
class Bot(Player, ABC):
    """Base player subtype that can autonomously choose actions."""

    @abstractmethod
    def make_choice(self, *, player_id: int, game: "Game") -> ActionPayload | None:
        """Return one action payload to queue for the next tick, or None."""


@dataclass(slots=True)
class BorderBot(Bot):
    """Simple bot that attacks the first border region it can reach."""

    attack_percentage: float = 0.25
    attack_cooldown_ticks: int = 200
    min_soft_cap_fraction_to_attack: float = 0.70
    last_choice_tick: int = -200

    def make_choice(self, *, player_id: int, game: "Game") -> ActionPayload | None:
        if not self.is_alive or self.balance <= 0:
            return None
        if (game.tick_count - self.last_choice_tick) < self.attack_cooldown_ticks:
            return None

        owned_tiles = int((game.map == player_id).sum())
        player_soft_cap = soft_cap(owned_tiles)
        if player_soft_cap <= 0:
            return None
        if self.balance <= player_soft_cap * self.min_soft_cap_fraction_to_attack:
            return None

        hostile_target: tuple[int, int] | None = None
        neutral_target: tuple[int, int] | None = None

        rows, cols = game.map.shape
        for row, col in zip(*((game.map == player_id).nonzero()), strict=False):
            row = int(row)
            col = int(col)
            for n_row, n_col in _neighbors4(row=row, col=col, rows=rows, cols=cols):
                owner_id = int(game.map[n_row, n_col])
                if owner_id in (0, player_id):
                    continue
                if owner_id >= 2:
                    hostile_target = (n_row, n_col)
                    break
                if neutral_target is None:
                    neutral_target = (n_row, n_col)
            if hostile_target is not None:
                break

        target = hostile_target if hostile_target is not None else neutral_target
        if target is None:
            return None
        self.last_choice_tick = game.tick_count

        return {
            "type": "attack",
            "target": [target[0], target[1]],
            "percentage": self.attack_percentage,
        }


def _neighbors4(
    *, row: int, col: int, rows: int, cols: int
) -> tuple[tuple[int, int], ...]:
    candidates = (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )
    return tuple(
        (n_row, n_col)
        for n_row, n_col in candidates
        if 0 <= n_row < rows and 0 <= n_col < cols
    )
