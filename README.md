# mancala-solver

An exact + heuristic **position solver** for Kalah-style Mancala, written in Rust.

Give it a board position and it tells you, for the side to move:

- the **exact game-theoretic result** with perfect play (win/loss/draw and the
  final score margin in seeds), computed by alpha-beta search to the end of the
  game;
- a ranked **evaluation of every legal move**;
- the **principal variation** (the best line for both sides).

It accelerates the endgame with an **automatic tablebase** (built and cached on
first use) and, for positions still too large to solve exactly within a node
budget, automatically falls back to a **depth-limited heuristic** search and
says so.

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

## Web UI (runs in the browser, no server)

The engine also compiles to **WebAssembly**, so it runs entirely client-side —
an interactive board where you play moves and the solver shows the exact result,
best move, per-move table, and principal variation. Everything in [`docs/`](docs/)
is a static site (HTML/CSS/JS + a ~90 KB `mancala.wasm`). The analysis runs in a
**Web Worker** so a long search never freezes the page — the board stays
responsive and a **Stop** button abandons a search in progress.

Try it locally:

```sh
docs/build.sh                 # rebuild docs/mancala.wasm from the engine
python3 -m http.server -d docs 8000   # then open http://localhost:8000
```

**Free hosting on GitHub Pages:** in the repo, *Settings → Pages → Build and
deployment → Source: Deploy from a branch*, then pick your branch and the
`/docs` folder. GitHub serves it (with the correct `application/wasm` type) at
`https://<user>.github.io/mancala-solver/`. No build step or server needed — the
prebuilt wasm is committed; rerun `docs/build.sh` to refresh it after engine
changes.

In the browser the **play engine is the engine**: it consults the opening book
for an instant exact result if the position is in it, otherwise runs a
node-budgeted iterative-deepening search that plays the endgame perfectly via
the bundled tablebase and uses a quiescence search (~+90 Elo over the bare
heuristic, measured by the self-play harness). The selectable **effort** (node
budget: 2M / 20M / 80M) is that search's budget — bigger means it deepens
further, so it directly buys stronger play. A small **6-pit endgame tablebase**
(`docs/kalah6.bin`, ~5 MB, ≤ 12 seeds in play) and the opening book ship with the
page; non-book mid-game positions are a strong, clearly-labelled *estimate*
rather than a proof (the high-cap tablebases that would prove them aren't shipped
to the browser). The browser transposition table is memory-capped (the
dense table tops out at 256 MB), so a deep search slows down rather than
crashing the tab — the earlier out-of-memory at the highest effort is fixed.
Rebuild the bundled table with
`mancala-solver gen-tb --pits 6 --seeds-cap 12 --out docs/kalah6.bin`.

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
| `--tb <file>` | Use a specific precomputed endgame tablebase | — |
| `--no-tb` | Disable the automatic endgame tablebase | off |
| `--seeds-cap <N>` | Seed cap for the automatic tablebase | `14` |

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

Exact solving is a full search to the end of the game (alpha-beta with a
store-independent transposition table, PVS + history-heuristic ordering, and an
endgame tablebase), so its cost grows steeply with board size. With the
automatic tablebase (on by default — see below), approximate exact-solve times
on a 4-core machine:

| Board | Result (player 1) | Exact-solve time |
| --- | --- | --- |
| Kalah(5,2) | win by 2 | <0.1 s |
| Kalah(6,2) | win by 6 | <0.5 s |
| Kalah(4,4) | win by 2 | ~1 s |
| Kalah(5,3) | win by 6 | ~1 s |
| Kalah(5,4) | win by 10 | ~7 s |
| Kalah(6,3) | win by 2 | ~4 s |
| **Kalah(6,4)** | **win by 8** (best opening: pit 2) | **~3 min** (cap-17 tablebase: ~100 s build, ~99 MB) |

The classic **Kalah(6,4)** is solvable: it's a first-player win by 8 seeds, with
the optimal opening being pit 2 (the move that scores into the store for a free
turn). It needs a larger tablebase than the default cap (e.g.
`--seeds-cap 17`) and a raised `--budget`.

The transposition table is a custom dense open-addressing table (one `u128`
per slot, packing the position key and value) that grows with the search and
is `madvise`d for transparent huge pages. It caches far more positions per
gigabyte than a general-purpose hash map, which is what makes a full Kalah(6,4)
solve fit in memory while dropping fewer positions.

For positions still too large to finish within the node budget, the solver
automatically falls back to a depth-limited **heuristic** search, clearly
labelled in the output. Memory stays bounded throughout (the transposition
table grows only up to a fixed cap — it will not exhaust RAM).

The move-evaluation table reports an **exact** value for the best move; clearly
inferior moves are shown as a bound (e.g. `≤ +2 seeds`) because the solver
proves they cannot beat the best move rather than spending time computing their
exact value.

## Endgame tablebases

The cost of an exact solve is dominated by positions with many seeds still in
play. A key fact lets us shortcut the rest: a position's optimal margin is
`(my_store − opp_store) + g(pits, side)`, where the *future* differential `g`
depends only on the pit layout and side to move — not on the seeds already
banked. So once few enough seeds remain, the answer is a banked-score difference
plus a `g` lookup.

`g` is precomputed for **every** layout up to a seed cap into a tablebase file
(a flat array indexed by a combinatorial rank of the layout, so no keys are
stored on disk). Entries are **mirror-canonical** — Kalah is player-symmetric,
so a layout is indexed as (mover's pits, opponent's pits) and one entry serves
both players, halving the table. The same table also serves every *fill* of a
board size — Kalah(6,2), (6,3), (6,4), … all share the 6-pit table.

**Automatic (default).** For boards up to 6 pits per side with enough seeds in
play, the solver builds a tablebase on first use, caches it under
`$MANCALA_TB_DIR` (default `./.mancala_tb`), and reuses it on later runs — no
manual step needed. Disable with `--no-tb`, or change the cap with
`--seeds-cap`. Building uses all CPU cores.

**Explicit.** You can also build and reuse one yourself:

```sh
# Build (uses all cores): all layouts with <= 16 seeds in play on a 6-pit board.
mancala-solver gen-tb --pits 6 --seeds-cap 17 --out kalah6.tb

# Use it on any 6-pit position; endgame positions become O(1) array lookups.
mancala-solver start --pits 6 --seeds 4 --tb kalah6.tb --budget 100000000000
```

Higher caps cover more of the search tree (faster solves of hard boards like
Kalah(6,4)) at the cost of a larger one-time build and file.

## Opening book

The endgame tablebase cuts the search off at the **bottom** (few seeds left); an
opening book cuts it off at the **top** (the first few plies). Unlike the
endgame — whose ≤cap positions are a small closed set — an early position's exact
value depends on the whole tree beneath it, so a *proven* book can't make the
**first** solve faster. It is a cheap **byproduct** of solving once: the layouts
reachable within a few plies of the opening are solved together with one warm
transposition table, and the result makes the opening / early game an **O(1)
exact lookup** afterward. Entries are mirror-canonical and store-independent,
exactly like the tablebase.

```sh
# Build (needs an endgame tablebase) and then use it: the opening is now instant.
mancala-solver gen-book --pits 6 --seeds 4 --plies 2 --tb kalah6.tb --out kalah6_book.bin
mancala-solver start --pits 6 --seeds 4 --book kalah6_book.bin --tb kalah6.tb
```

This is what lets the **web UI** show the exact "win by 8, best pit 2" for the
opening instantly, instead of the heuristic estimate it would otherwise fall back
to (the browser ships only a small endgame table).

## Measuring playing strength

For positions too large to solve exactly, strength only shows up over many
games. The `playtest` command pits two engine configurations against each other
over a diverse opening book (each opening played from both sides so colour bias
cancels) and reports the score, the **Elo difference with a 95% confidence
interval**, and the likelihood one side is stronger — the way engine changes are
normally graded:

```sh
# Does searching deeper actually play better? (it does)
mancala-solver playtest --a h:8 --b h:6 --games 1000 --threads 8
```

Engine specs are `h:<depth>` (depth-limited heuristic) or `a:<budget>:<depth>`
(exact within a node budget, else heuristic). A symmetric matchup (`--a h:6 --b
h:6`) scores exactly 50% / 0 Elo, confirming the harness is unbiased.

## Project layout

The engine is a UI-independent library (`mancala`) so a future GUI/web front-end
can reuse it:

- `src/board.rs` — board representation, rules, move generation/application.
- `src/solver.rs` — alpha-beta solver (PVS + history ordering), a
  store-independent transposition table, node budget, and heuristic fallback;
  position analysis.
- `src/endgame.rs` — lazily-built, store-independent in-memory endgame table.
- `src/tablebase.rs` — offline endgame tablebase: parallel build, save, load, lookup.
- `src/book.rs` — proven opening book: build (warm-TT solves), lookup, exact
  early-game analysis, save/load.
- `src/hash.rs` — fast `u128`-key hasher shared by the TT and endgame tables.
- `src/notation.rs` — board-notation parsing/formatting.
- `src/playtest.rs` — self-play strength testing: opening book, match runner,
  Elo / confidence-interval / likelihood-of-superiority statistics.
- `src/main.rs` — the `mancala-solver` CLI (`analyze`, `start`, `gen-tb`,
  `gen-book`, `playtest`).

## Tests

```sh
cargo test                # fast unit + integration tests
cargo test -- --ignored   # also run the heavier Kalah(6,3) exact solve (~1 min)
```
