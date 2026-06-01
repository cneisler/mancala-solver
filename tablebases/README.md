# Endgame tablebases

This directory holds precomputed endgame tablebases (`*.tb`), which are tracked
with **Git LFS** (see `../.gitattributes`).

A tablebase stores the exact store-independent future differential `g(pits,
side)` for every pit layout up to a seed cap on a fixed board size. Loading one
with `--tb` turns endgame positions (≤ cap seeds in play) into O(1) lookups.

## Building one

Tablebases are large binary artifacts, so they are generated rather than kept in
the source tree by default. Reproduce one with:

```sh
# pits=6, all layouts with <= 16 seeds in play (~2.4 min build, ~117 MB)
cargo run --release -- gen-tb --pits 6 --seeds-cap 16 --out tablebases/kalah6_cap16.tb
```

Then use it on any board of that size:

```sh
cargo run --release -- start --pits 6 --seeds 4 --tb tablebases/kalah6_cap16.tb --budget 100000000000
```

With the cap-16 table, Kalah(6,4) solves exactly (first player wins by 8, best
opening pit 2).

## Committing a built tablebase

`*.tb` files are LFS-tracked, so committing one works normally
(`git add tablebases/<file>.tb`) **wherever the remote supports Git LFS**
(e.g. github.com with LFS enabled). Some CI/sandbox git proxies do not implement
the LFS API; in that case build the table locally instead of fetching it.
