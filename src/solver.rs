//! Position solver: an exact alpha-beta search with a node budget, falling back
//! to a depth-limited heuristic search when the exact search is too expensive.
//!
//! # Negamax with extra turns
//!
//! Mancala grants an extra turn when the last seed lands in your own store. We
//! search with negamax, but only flip the sign when the turn actually passes to
//! the opponent; an extra-turn move keeps the same perspective and the same
//! alpha-beta window, because it is the *same* player continuing to maximize.
//!
//! Exact values are measured in seeds (the final score margin from the side to
//! move). Heuristic values use an internal scale dominated by the store
//! difference and are not directly comparable to exact seed margins.

use std::collections::HashMap;

use crate::board::{Board, Player, Rules};

/// Weight applied to a stored seed in the heuristic evaluation (and to the
/// terminal margin in depth-limited mode, so terminal results dominate).
const STORE_WEIGHT: i32 = 100;

/// A large finite value used as the open alpha-beta window bound.
const INF: i32 = 1_000_000;

/// The outcome category of a position for the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
    Draw,
}

/// Evaluation of a single candidate move from the analyzed position.
#[derive(Clone, Copy, Debug)]
pub struct MoveEval {
    /// 0-based pit index for the side to move.
    pub pit: usize,
    /// Value of the resulting position from the side-to-move's perspective.
    pub value: i32,
    /// Whether this move grants an extra turn.
    pub extra_turn: bool,
    /// Whether this move performs a capture.
    pub captured: bool,
}

/// The full analysis of a position.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// `true` if the result is an exact game-theoretic value; `false` if it is a
    /// depth-limited heuristic estimate.
    pub exact: bool,
    /// The side the analysis is computed for.
    pub side_to_move: Player,
    /// Value from the side-to-move's perspective. In exact mode this is the
    /// final score margin in seeds; otherwise an internal heuristic score.
    pub value: i32,
    /// The best move, or `None` if the position is already terminal.
    pub best_move: Option<usize>,
    /// All legal moves with their evaluations, sorted best-first.
    pub move_evals: Vec<MoveEval>,
    /// Principal variation (sequence of pit indices) starting from this position.
    pub pv: Vec<usize>,
    /// Number of search nodes visited.
    pub nodes: u64,
    /// Search depth in plies if the result is heuristic; `None` if exact.
    pub depth: Option<u32>,
}

impl Analysis {
    /// The outcome for the side to move. For heuristic results this reflects the
    /// sign of the estimate and should be read as a best guess.
    pub fn outcome(&self) -> Outcome {
        if self.value > 0 {
            Outcome::Win
        } else if self.value < 0 {
            Outcome::Loss
        } else {
            Outcome::Draw
        }
    }

    /// Final score margin in seeds, if known exactly.
    pub fn margin_seeds(&self) -> Option<i32> {
        if self.exact {
            Some(self.value)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
enum Flag {
    Exact,
    Lower,
    Upper,
}

struct TtEntry {
    value: i32,
    flag: Flag,
    /// Best move's pit index, or 255 if none was recorded.
    best: u8,
}

/// Transposition-table key. Small boards pack the whole position (cells + side +
/// depth marker) into a single `u128` for fast, allocation-free hashing; larger
/// boards fall back to a heap-allocated byte key.
#[derive(Clone, PartialEq, Eq, Hash)]
enum TtKey {
    Packed(u128),
    Wide(Box<[u8]>),
}

/// Number of bits used to encode a single cell in the packed key. Cells holding
/// 64 or more seeds force the wide fallback.
const CELL_BITS: u32 = 6;

/// Maximum number of transposition-table entries. Caps memory (~40 bytes/entry,
/// so ≈1 GB here) so an unbounded exact search can never exhaust RAM: once the
/// table is full we simply stop caching new positions (they are recomputed if
/// revisited). The search remains correct, only slower.
const TT_MAX_ENTRIES: usize = 24_000_000;

struct Searcher {
    rules: Rules,
    tt: HashMap<TtKey, TtEntry>,
    tt_cap: usize,
    nodes: u64,
    budget: u64,
    aborted: bool,
}

impl Searcher {
    fn new(rules: Rules, budget: u64) -> Searcher {
        Searcher {
            rules,
            tt: HashMap::new(),
            tt_cap: TT_MAX_ENTRIES,
            nodes: 0,
            budget,
            aborted: false,
        }
    }

    /// Transposition-table key: cells + side to move + a depth marker
    /// (255 for exact searches so they never collide with depth-limited entries).
    ///
    /// Packs into a `u128` when every cell fits in [`CELL_BITS`] and the total
    /// bit-width fits; otherwise falls back to a byte key.
    fn key(b: &Board, depth: Option<u32>) -> TtKey {
        let cells = b.cells();
        let turn_bit: u128 = match b.turn() {
            Player::P0 => 0,
            Player::P1 => 1,
        };
        let depth_marker: u128 = match depth {
            None => 255,
            Some(d) => d.min(254) as u128,
        };
        // Bits: cells (CELL_BITS each) + 1 turn bit + 8 depth-marker bits.
        let needed = cells.len() as u32 * CELL_BITS + 1 + 8;
        let packable = needed <= 128 && cells.iter().all(|&c| (c as u32) < (1 << CELL_BITS));
        if packable {
            let mut packed: u128 = 0;
            for &c in cells {
                packed = (packed << CELL_BITS) | c as u128;
            }
            packed = (packed << 1) | turn_bit;
            packed = (packed << 8) | depth_marker;
            TtKey::Packed(packed)
        } else {
            let mut k = Vec::with_capacity(cells.len() + 2);
            k.extend_from_slice(cells);
            k.push(turn_bit as u8);
            k.push(depth_marker as u8);
            TtKey::Wide(k.into_boxed_slice())
        }
    }

    /// Order moves to improve alpha-beta pruning: a transposition-table move
    /// first, then extra-turn moves (computed without cloning the board).
    fn order_moves(&self, b: &Board, moves: &mut [usize], tt_move: Option<usize>) {
        moves.sort_by_key(|&mv| {
            let mut rank = 0i32;
            if Some(mv) == tt_move {
                rank -= 1000;
            }
            if b.grants_extra_turn(mv) {
                rank -= 100;
            }
            rank
        });
    }

    fn heuristic(&self, b: &Board) -> i32 {
        let p = b.turn();
        let o = p.other();
        let store_diff = b.store(p) as i32 - b.store(o) as i32;
        let mut pit_self = 0i32;
        let mut pit_opp = 0i32;
        for i in 0..b.pits_per_side() {
            pit_self += b.cells()[b.pit_global(p, i)] as i32;
            pit_opp += b.cells()[b.pit_global(o, i)] as i32;
        }
        store_diff * STORE_WEIGHT + (pit_self - pit_opp)
    }

    /// Negamax + alpha-beta. `depth == None` searches to terminal (exact);
    /// `depth == Some(d)` searches `d` plies then applies the heuristic.
    fn search(&mut self, b: &Board, mut alpha: i32, mut beta: i32, depth: Option<u32>) -> i32 {
        self.nodes += 1;
        if depth.is_none() && self.nodes > self.budget {
            self.aborted = true;
            return 0;
        }

        if b.is_terminal() {
            let m = b.terminal_margin(b.turn());
            return match depth {
                None => m,
                Some(_) => m * STORE_WEIGHT,
            };
        }
        if depth == Some(0) {
            return self.heuristic(b);
        }

        let key = Self::key(b, depth);
        let alpha_orig = alpha;
        let mut tt_move = None;
        if let Some(e) = self.tt.get(&key) {
            match e.flag {
                Flag::Exact => return e.value,
                Flag::Lower => alpha = alpha.max(e.value),
                Flag::Upper => beta = beta.min(e.value),
            }
            if alpha >= beta {
                return e.value;
            }
            tt_move = if e.best == 255 { None } else { Some(e.best as usize) };
        }

        let mut moves = b.legal_moves(b.turn());
        self.order_moves(b, &mut moves, tt_move);
        let child_depth = depth.map(|d| d - 1);

        let mut best = i32::MIN;
        let mut best_move = None;
        for mv in moves {
            let r = b.apply(&self.rules, mv);
            let v = if r.extra_turn {
                // Same player continues: keep perspective and window.
                self.search(&r.board, alpha, beta, child_depth)
            } else {
                -self.search(&r.board, -beta, -alpha, child_depth)
            };
            if self.aborted {
                return 0;
            }
            if v > best {
                best = v;
                best_move = Some(mv);
            }
            alpha = alpha.max(v);
            if alpha >= beta {
                break;
            }
        }

        let flag = if best <= alpha_orig {
            Flag::Upper
        } else if best >= beta {
            Flag::Lower
        } else {
            Flag::Exact
        };
        // Stop caching once the table is full to keep memory bounded; existing
        // entries are retained so most of the cache stays useful.
        if self.tt.len() < self.tt_cap || self.tt.contains_key(&key) {
            self.tt.insert(
                key,
                TtEntry {
                    value: best,
                    flag,
                    best: best_move.map_or(255, |m| m as u8),
                },
            );
        }
        best
    }

    /// Follow stored best moves to reconstruct the principal variation, starting
    /// *from* `start`. Capped to avoid pathological loops.
    fn follow_pv(&self, start: &Board, depth: Option<u32>, cap: usize) -> Vec<usize> {
        let mut pv = Vec::new();
        let mut cur = start.clone();
        let mut d = depth;
        while !cur.is_terminal() && pv.len() < cap {
            if d == Some(0) {
                break;
            }
            let key = Self::key(&cur, d);
            let Some(entry) = self.tt.get(&key) else { break };
            if entry.best == 255 {
                break;
            }
            let mv = entry.best as usize;
            pv.push(mv);
            let r = cur.apply(&self.rules, mv);
            cur = r.board;
            d = d.map(|x| x - 1);
        }
        pv
    }

    /// Evaluate every legal move from the root with a full window so the
    /// per-move values are exact (no root pruning), and assemble the analysis.
    fn analyze_root(&mut self, b: &Board, depth: Option<u32>) -> Analysis {
        if b.is_terminal() {
            let value = b.terminal_margin(b.turn());
            return Analysis {
                exact: depth.is_none(),
                side_to_move: b.turn(),
                value: if depth.is_none() { value } else { value * STORE_WEIGHT },
                best_move: None,
                move_evals: Vec::new(),
                pv: Vec::new(),
                nodes: self.nodes,
                depth,
            };
        }

        let mut moves = b.legal_moves(b.turn());
        self.order_moves(b, &mut moves, None);

        let mut evals = Vec::with_capacity(moves.len());
        let mut best = i32::MIN;
        let mut best_move = None;
        let child_depth = depth.map(|d| d - 1);

        for mv in moves {
            let r = b.apply(&self.rules, mv);
            let v = if r.extra_turn {
                self.search(&r.board, -INF, INF, child_depth)
            } else {
                -self.search(&r.board, -INF, INF, child_depth)
            };
            if self.aborted {
                // Caller will discard this partial result.
                break;
            }
            evals.push(MoveEval {
                pit: mv,
                value: v,
                extra_turn: r.extra_turn,
                captured: r.captured,
            });
            if v > best {
                best = v;
                best_move = Some(mv);
            }
        }

        evals.sort_by(|a, c| c.value.cmp(&a.value));

        let pv = match best_move {
            Some(mv) => {
                let mut pv = vec![mv];
                let child = b.apply(&self.rules, mv).board;
                let cap = 4 * (b.total_cells() + b.cells().iter().map(|&c| c as usize).sum::<usize>());
                pv.extend(self.follow_pv(&child, child_depth, cap));
                pv
            }
            None => Vec::new(),
        };

        Analysis {
            exact: depth.is_none(),
            side_to_move: b.turn(),
            value: best,
            best_move,
            move_evals: evals,
            pv,
            nodes: self.nodes,
            depth,
        }
    }
}

/// Analyze `board`: attempt an exact solve within `node_budget`; if that budget
/// is exceeded, fall back to a depth-limited heuristic search of
/// `fallback_depth` plies.
pub fn analyze(board: &Board, rules: Rules, node_budget: u64, fallback_depth: u32) -> Analysis {
    let mut exact = Searcher::new(rules, node_budget);
    let result = exact.analyze_root(board, None);
    if !exact.aborted {
        return result;
    }

    // Exact search ran out of budget — fall back to a heuristic search.
    let mut limited = Searcher::new(rules, u64::MAX);
    limited.analyze_root(board, Some(fallback_depth))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve_exact(b: &Board) -> Analysis {
        analyze(b, Rules::default(), u64::MAX, 12)
    }

    #[test]
    fn tiny_terminal_position_is_exact_draw() {
        // 1 pit per side, both empty, equal stores -> draw.
        let cells = vec![0, 5, 0, 5];
        let b = Board::new(1, cells, Player::P0).unwrap();
        let a = solve_exact(&b);
        assert!(a.exact);
        assert_eq!(a.value, 0);
        assert_eq!(a.outcome(), Outcome::Draw);
        assert_eq!(a.best_move, None);
    }

    #[test]
    fn forced_capture_win_is_exact() {
        // Kalah(2,*). P0 plays pit 0 (1 seed): it lands in P0's empty pit 1
        // (global 1), whose opposite pit (global 3) holds 5 seeds -> capture of
        // 6 into P0's store. The turn passes to P1, whose side is then empty, so
        // the game ends with P0 ahead by 5 (6 vs 1).
        let cells = vec![1, 0, 0, 5, 1, 0];
        let b = Board::new(2, cells, Player::P0).unwrap();
        let a = solve_exact(&b);
        assert!(a.exact);
        assert_eq!(a.best_move, Some(0));
        assert_eq!(a.value, 5);
        assert_eq!(a.outcome(), Outcome::Win);
    }

    #[test]
    fn kalah_3_3_solves_exactly_and_deterministically() {
        // A small but non-trivial board (3 pits, 3 seeds = 18 seeds) that the
        // exact solver handles quickly. Guards the negamax/extra-turn logic
        // beyond the one-ply tests, and pins the (deterministic) result.
        let b = Board::start(3, 3, Player::P0).unwrap();
        let a = solve_exact(&b);
        assert!(a.exact);
        // For (3,3) the opening that lands its last seed in the store (distance
        // to store == seeds == 3) is pit index 0, and that extra-turn opening is
        // the engine's first choice. The exact game value is a first-player win
        // by 2 seeds (verified empirically).
        assert_eq!(a.best_move, Some(0));
        assert_eq!(a.value, 2);
        assert_eq!(a.outcome(), Outcome::Win);
        assert!(!a.pv.is_empty());
    }

    #[test]
    #[ignore = "exact solve of Kalah(6,2) takes tens of seconds; run with --ignored"]
    fn kalah_6_2_exact_value_regression() {
        // A larger board that still solves exactly in a reasonable time (~70M
        // nodes). Pins the engine's exact result as a regression check.
        let b = Board::start(6, 2, Player::P0).unwrap();
        let a = analyze(&b, Rules::default(), u64::MAX, 12);
        assert!(a.exact, "Kalah(6,2) should solve exactly");
        assert_eq!(a.value, 6, "Kalah(6,2) is a first-player win by 6 seeds");
        assert_eq!(a.outcome(), Outcome::Win);
        assert_eq!(a.best_move, Some(4));
        assert!(!a.pv.is_empty());
    }

    #[test]
    fn budget_forces_heuristic_fallback() {
        let b = Board::start(6, 4, Player::P0).unwrap();
        let a = analyze(&b, Rules::default(), 100, 6); // tiny budget
        assert!(!a.exact, "tiny budget must trigger heuristic fallback");
        assert_eq!(a.depth, Some(6));
        assert!(a.best_move.is_some());
    }

    #[test]
    fn move_evals_cover_all_legal_moves() {
        let b = Board::start(3, 3, Player::P0).unwrap();
        let a = solve_exact(&b);
        assert_eq!(a.move_evals.len(), 3);
        // Sorted best-first.
        for w in a.move_evals.windows(2) {
            assert!(w[0].value >= w[1].value);
        }
    }
}
