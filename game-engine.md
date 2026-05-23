# Game engine (high level)

What the engine does — not implementation details. See the code for exact behavior.

## Tick order

`attack()` only queues; the map changes on the next `tick()`.

1. Interest (alive players)  
2. Start queued attacks (deduct balance, open waves)  
3. Reciprocal clash (A↔B pairs)  
4. Advance active attacks (one tile-depth per attack)  
5. Single-blob cleanup per player  
6. Eliminations & win check  
7. Bots queue attacks (skipped if game already won)

**Queueing an attack:** valid player → enemy/neutral target → your land touches that region → not already attacking that defender → pending.

## Map

- `0` water · `1` unoccupied land · `≥2` player id  
- **Single blob:** after each tick, keep each player’s largest 4-connected territory; other owned tiles become unoccupied land.

## Players

- **Balance**, alive/eliminated, spawn location.  
- Interest from owned area (caps + early-game boost). Dead players own nothing after cleanup and gain no interest.

## Attack

- Only action type. Target a tile in a **connected region** (player or neutral); commit a **fraction of balance**.  
- Target region is **frozen** when the attack starts; the wave only spreads inside it.  
- **One step per tick** per active attack: capture tiles your land can reach one tile into that region.  
- Can’t afford the whole step → attack ends, **no map change that tick**.  
- Multiple active attacks run in **list order** (later ones see updated ownership).  
- At most one pending/active attack per **attacker → defender** pair.

**Costs:** combat units per tile (higher if tile owner still has balance). Separate budget can drain the defender’s balance while you take their tiles.

**No refunds:** investment, tax, and unused combat/defender-damage budget are gone when the attack ends.

| Term | Meaning |
|------|---------|
| Investment | Balance paid when the attack starts |
| Attack units | Combat budget after tax |
| Remaining units | Combat budget left on the wave |
| Defender damage budget | Cap on balance damage to the defender |

## How attacks interact

| Case | Notes |
|------|--------|
| **A→B and B→A** | Different target regions. Clash on strength before expanding (below). |
| **A→C and B→C** (same region) | Same frozen blob possible. Order matters; borders can meet without a neutral gap. |
| **A→B and C→A** | Only via map changes, not shared waves. |
| **A→B and A→C** | Allowed; regions don’t overlap. |

## Reciprocal clash (A↔B)

Before expansion, while both directions are active: compare **remaining attack units**. Equal → both attacks end. Unequal → weaker ends; stronger loses that many units (and matching defender-damage budget). Stronger may expand one step if anything remains. No territory rollback, no refunds.

## Win / elimination

- Eliminated at 0 tiles after cleanup.  
- Win at occupiable-fraction threshold (`GAME_WIN_OCCUPATION_FRACTION` in constants).
