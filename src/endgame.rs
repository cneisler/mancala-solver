//! Store-independent endgame database.
//!
//! The optimal *future* store differential of a position — call it `g` — depends
//! only on the seed counts in the **pits** and the side to move, never on the
//! seeds already banked in the two stores (those are a sunk constant added to
//! the final margin). Formally, for any position:
//!
//! ```text
//! optimal_margin = (my_store - opp_store) + g(pits, side_to_move)
//! ```
//!
//! Because `g` ignores the stores, every position that shares the same pit
//! layout and side to move collapses to a single entry — far fewer than the
//! full-board states the main transposition table distinguishes. The forward
//! search uses this as an endgame cutoff: once few enough seeds remain in play,
//! it returns `store_diff + g(...)` instead of recursing to the end of the game.
//!
//! `g` is computed by a memoised negamax where each move's immediate reward is
//! the number of seeds it banks into the mover's own store:
//!
//! ```text
//! g(pos) = max over moves of  gain(move) +/- g(child)
//! ```
//!
//! (`+` when the move grants an extra turn and the same player continues, `-`
//! when the turn passes — standard negamax). Recursion terminates because Kalah
//! itself always terminates, so the pit-only state graph is acyclic.

use crate::board::{Board, Rules};
use crate::hash::U128Map;

/// Bits used to encode each pit in the endgame key. A pit can hold at most the
/// number of seeds in play, so this supports cutoffs up to 63.
const PIT_BITS: u32 = 6;

/// A lazily-built table of `g` values keyed on pit layout + side to move.
pub struct Endgame {
    rules: Rules,
    /// Maximum seeds-in-play this table is consulted for.
    cutoff: u32,
    memo: U128Map<i16>,
}

impl Endgame {
    pub fn new(rules: Rules, cutoff: u32) -> Endgame {
        Endgame {
            rules,
            cutoff,
            memo: U128Map::default(),
        }
    }

    pub fn cutoff(&self) -> u32 {
        self.cutoff
    }

    pub fn len(&self) -> usize {
        self.memo.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memo.is_empty()
    }

    /// Key on the pit cells (both sides) plus the side to move; stores excluded.
    fn key(b: &Board) -> u128 {
        // Mirror-canonical: mover's pits first, no turn bit. Kalah is
        // player-symmetric and `g` is mover-perspective, so both mirrors of a
        // (mover, opponent) layout share one entry.
        let n = b.pits_per_side();
        let cells = b.cells();
        let mover = b.turn();
        let mut k = 0u128;
        for p in [mover, mover.other()] {
            for i in 0..n {
                k = (k << PIT_BITS) | cells[b.pit_global(p, i)] as u128;
            }
        }
        k
    }

    /// Optimal future store differential (mover minus opponent) from `b`,
    /// independent of the current store contents.
    pub fn g(&mut self, b: &Board) -> i16 {
        if b.is_terminal() {
            // Game over: each side sweeps its own remaining pit seeds.
            let p = b.turn();
            return b.pit_seeds(p) as i16 - b.pit_seeds(p.other()) as i16;
        }

        let key = Self::key(b);
        if let Some(&v) = self.memo.get(&key) {
            return v;
        }

        let p = b.turn();
        let n = b.pits_per_side();
        let store_before = b.store(p) as i16;
        let mut best = i16::MIN;
        for i in 0..n {
            if b.cells()[b.pit_global(p, i)] == 0 {
                continue;
            }
            let r = b.apply(&self.rules, i);
            let gain = r.board.store(p) as i16 - store_before; // seeds banked this move
            let child = self.g(&r.board);
            let v = if r.extra_turn { gain + child } else { gain - child };
            if v > best {
                best = v;
            }
        }

        self.memo.insert(key, best);
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Player;

    /// `g` of the start position equals the full-game margin from a zero-store
    /// board, which must match the known Kalah(3,3) value (+2).
    #[test]
    fn g_matches_known_small_value() {
        let mut eg = Endgame::new(Rules::default(), 64);
        let b = Board::start(3, 3, Player::P0).unwrap();
        // Stores are zero at the start, so the margin equals g.
        assert_eq!(eg.g(&b), 2);
    }

    #[test]
    fn g_is_store_independent() {
        let rules = Rules::default();
        // Same pits, different store contents -> identical g.
        let a = Board::new(2, vec![1, 2, 0, 3, 1, 0], Player::P0).unwrap();
        let b = Board::new(2, vec![1, 2, 9, 3, 1, 7], Player::P0).unwrap();
        let mut eg = Endgame::new(rules, 64);
        assert_eq!(eg.g(&a), eg.g(&b));
    }
}
