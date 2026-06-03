#!/usr/bin/env bash
# Rebuild the WebAssembly engine and drop it next to the web front-end.
# Usage: docs/build.sh   (run from the repo root or anywhere)
set -euo pipefail
cd "$(dirname "$0")/.."

rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo build --lib --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/mancala.wasm docs/mancala.wasm
echo "Updated docs/mancala.wasm ($(du -h docs/mancala.wasm | cut -f1))"
