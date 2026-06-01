# mancala-solver

An exact + heuristic **position solver** for Kalah-style Mancala, written in Rust.

Give it a board position and it tells you, for the side to move:

- the **exact game-theoretic result** with perfect play (win/loss/draw and the
  final score margin in seeds), computed by alpha-beta search to the end of the
  game;
- a ranked **evaluation of every legal move**;
- the **principal variation** (the best line for both sides).

For positions too large to solve exactly within a node budget, it automatically
falls back to a **depth-limited heuristic** search and says so.

## Variant and rules

This solver plays **Kalah** with a **configurable board size** (pits per side
and seeds per pit) and the following rules:

- Seeds are sown counterclockwise, one per pit, **skipping the opponent's
  store**.
- Landing the last seed in **your own store** grants an **extra turn**.
- Landing the last seed in one of **your own previously-empty pits captures**
  that seed plus the seeds in the **opposite** pit, into your store.
  - By default a capture only happens when the **opposite pit is non-empty**
    (the standard Kalah rule). Pass `--capture-empty` to allow capturing even
    when the opposite pit is empty.
- The game ends when **either** side's pits are all empty; each player then
  sweeps their own remaining pit seeds into their store. The higher store wins.

## Build

```sh
cargo build --release
```

The binary is `target/release/mancala-solver`. There are no external
dependencies.

## Usage

### Analyze a position from board notation

```sh
mancala-solver analyze --board "4,4,4,4,4,4,0 | 4,4,4,4,4,4,0"
```

**Board notation** is two comma-separated groups separated by `|`:

```
p0_pit0,...,p0_pitN-1,p0_store | p1_pit0,...,p1_pitN-1,p1_store
```

The left group is player 0's pits then player 0's store; the right group is
player 1's pits then player 1's store. The number of pits per side is inferred
from the notation, so any board size works.

### Analyze a standard opening

```sh
mancala-solver start --pits 6 --seeds 4
```

### Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--turn <0\|1>` | Side to move (`0`/`p0`/`south` or `1`/`p1`/`north`) | `0` |
| `--budget <N>` | Exact-search node budget before falling back to the heuristic | `20000000` |
| `--depth <D>` | Heuristic fallback search depth, in plies | `11` |
| `--capture-empty` | Allow captures when the opposite pit is empty | off |

### Example output

```
$ mancala-solver start --pits 4 --seeds 3
        N (P1)
         3   3   3   3
   0 ---------------  0   (P1 store | P0 store)
         3   3   3   3
        S (P0)
Notation:    3,3,3,3,0 | 3,3,3,3,0
Side to move: P0 (South)
Rules:        capture requires non-empty opposite pit = true

Analysis mode: EXACT (solved to game end)
Nodes searched: 531960

Result for P0 (South): WIN (perfect play ends + 6 seeds)
Best move: pit 1

Move evaluations (best first):
  * pit 1: +6 seeds  [extra turn]
    pit 2: +2 seeds
    pit 3: -2 seeds
    pit 0: -6 seeds

Principal variation:
  P0:1↻  →  P0:2  →  P1:1  →  P0:0↻  →  P0:3×  →  P1:0  →  ...
```

PV annotations: `↻` marks a move that earns an extra turn, `×` marks a capture.

### A note on exact solving and board size

Exact solving is a full search to the end of the game, so its cost grows
quickly with board size. On this engine, boards up to roughly Kalah(4,4) /
Kalah(6,2) solve exactly in well under a minute; larger boards (e.g. the
classic Kalah(6,4)) exceed the default node budget and automatically fall back
to the depth-limited **heuristic** search, which is clearly labelled in the
output. Raise `--budget` to spend more effort on an exact result (memory stays
bounded — the solver will not exhaust RAM), or lower `--depth` for a faster
heuristic answer.

## Project layout

The engine is a UI-independent library (`mancala`) so a future GUI/web front-end
can reuse it:

- `src/board.rs` — board representation, rules, move generation/application.
- `src/solver.rs` — alpha-beta solver with transposition table, node budget, and
  heuristic fallback; position analysis.
- `src/notation.rs` — board-notation parsing/formatting.
- `src/main.rs` — the `mancala-solver` CLI.

## Tests

```sh
cargo test                # fast unit + integration tests
cargo test -- --ignored   # also run the full Kalah(6,4) exact solve (slow)
```
