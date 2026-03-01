# territory-bot

Minimal terrain game primitives.

## Quick start

```python
from game.core import Game

game = Game(seed=42)
player_id = game.add_player()
state = game.get_state()
```

## API summary

- `Game(seed=None, land_coverage=0.62)`: builds the terrain map on init.
- `add_player() -> int`: spawns a player on unoccupied land and returns the player ID.
  New players also claim nearby unoccupied land in a radius of `5`.
  Raises `ValueError` if you try to add more than `MAX_PLAYER_COUNT` (100).
- `action(player_id, payload) -> bool`: queues an action for that player.
- `tick() -> int`: applies interest to all players, then advances one tick.
- `get_state() -> np.ndarray`: returns a copy of the map state.
- `game.players[player_id]`: `Player` object with `balance`, `income_value`, and spawn coordinates.

## Interest helpers

- `game.interest` contains pure functions for:
  territorial rate, initial augmentation, cap limiter,
  total tick rate, and `apply_interest(...)` for capped balance updates.

## Map values

- `0`: water
- `1`: land (unoccupied)
- `2+`: land occupied by a player ID

## Common gotchas

- Player IDs start at `2` by design (`0` and `1` are reserved).
- `MAX_PLAYER_COUNT` is `100`; adding the 101st player raises `ValueError`.
- Player spawn claims only convert unoccupied land (`1`), never water (`0`) or other players.
- `add_player()` returns `-1` when there are no unoccupied land tiles left.
- `get_state()` returns a copy; editing it does not mutate the game state.
