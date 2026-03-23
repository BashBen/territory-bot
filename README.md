# territory-bot

Minimal terrain game primitives.

## Quick start

```python
from game.core import Game

game = Game(seed=42)
player_id = game.add_player()
state = game.get_state()
ownership_map = state[0]
balance_map = state[1]
```

## API summary

- `Game(seed=None, land_coverage=0.62)`: builds the terrain map on init.
- `add_player() -> int`: spawns a player on unoccupied land and returns the player ID.
  New players also claim nearby unoccupied land in a radius of `5`.
  Raises `ValueError` if you try to add more than `MAX_PLAYER_COUNT` (100).
- `attack(player_id, payload) -> bool`: queues an attack for the next tick.
  Payload: `{"type": "attack", "target": [row, col], "percentage": 0.25}`.
- `action(player_id, payload) -> bool`: compatibility wrapper that currently routes to `attack(...)`.
- `tick() -> int`: applies interest and advances active attacks by one BFS layer.
- `get_state(relative=None) -> np.ndarray`: returns a state array with shape
  `(2, rows, cols)`.
  `state[0]` is the ownership map.
  `state[1]` is the balance map, which stores the owning player's balance on
  each owned tile, with water and unoccupied land set to `0`.
  When `relative` is set to a player ID, that player's land is remapped to `2`
  in `state[0]`, and existing player `2` tiles are remapped to that player ID.
  This is intended for agent/NN inputs where player `2` should always mean
  "self".
- `game.players[player_id]`: `Player` object with `balance`, `income_value`, and spawn coordinates.

## Interest helpers

- `game.interest` contains pure functions for:
  territorial rate, initial augmentation, cap limiter,
  total tick rate, and `apply_interest(...)` for capped balance updates.
- Action internals are split under `game/actions/`; `ActionEngine` dispatches to a
  list of `ActionHandler` implementations (currently `attack`).

## Map values

- `0`: water
- `1`: land (unoccupied)
- `2+`: land occupied by a player ID

## Common gotchas

- Player IDs start at `2` by design (`0` and `1` are reserved).
- `MAX_PLAYER_COUNT` is `100`; adding the 101st player raises `ValueError`.
- Player spawn claims only convert unoccupied land (`1`), never water (`0`) or other players.
- `add_player()` returns `-1` when there are no unoccupied land tiles left.
- `attack` fails when target is water, your own tile, or a non-touching region.
- `attack` spreads from all border tiles where your land touches the targeted region.
- `attack` also fails if you already have an active/pending attack against that same defender.
- `get_state()` returns a copy; editing it does not mutate the game state.
