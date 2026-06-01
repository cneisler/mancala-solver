//! A fast hasher for `u128` keys used by the transposition table and the
//! endgame memo. The keys are already well-distributed packed bit-fields, so a
//! single multiply-xor mix is enough; the maps still store the full key, so a
//! collision only costs a probe, never correctness.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct U128Hasher(u64);

impl Hasher for U128Hasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        // Keys are fed via `write_u128`; this path is unused.
    }
    fn write_u128(&mut self, i: u128) {
        let mut h = (i as u64) ^ ((i >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 32;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 29;
        self.0 = h;
    }
}

/// A `HashMap` over `u128` keys using [`U128Hasher`].
pub type U128Map<V> = HashMap<u128, V, BuildHasherDefault<U128Hasher>>;
