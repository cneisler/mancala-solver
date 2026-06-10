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

use crate::board::{Board, Player, Rules, MAX_PITS};
use crate::endgame::Endgame;
use crate::hash::U128Map;
use crate::tablebase::Tablebase;

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
    /// When `bound` is `true` this is only an upper bound (the move is provably
    /// no better than the best move, so it was not searched to an exact value).
    pub value: i32,
    /// If `true`, `value` is an upper bound (`≤ value`) rather than exact. This
    /// happens for inferior moves under the single-pass root search, which
    /// avoids the cost of proving an exact value for moves that cannot win.
    pub bound: bool,
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

/// Number of bits used to encode a single cell in the packed key. Cells holding
/// 64 or more seeds (or boards too wide to fit) skip the transposition table.
const CELL_BITS: u32 = 6;

/// Maximum number of transposition-table entries. Caps memory so an unbounded
/// exact search can never exhaust RAM: once the table is full we simply stop
/// caching new positions (they are recomputed if revisited), keeping the search
/// correct, only slower. At ~48 bytes per stored entry this is ≈8.5 GB, sized
/// to let a full Kalah(6,4) solve fit in memory on a 16 GB host.
const TT_MAX_ENTRIES: usize = 180_000_000;

type TtMap = U128Map<TtEntry>;

/// Seeds-in-play threshold at or below which the exact search consults the
/// store-independent endgame table instead of recursing. Chosen empirically as
/// a balance between endgame-table size and how much of the forward tree it
/// prunes.
const ENDGAME_CUTOFF: u32 = 14;

struct Searcher<'a> {
    rules: Rules,
    tt: TtMap,
    tt_cap: usize,
    /// History heuristic: cumulative beta-cutoff weight per `[side][pit]`, used
    /// to order quiet moves. Coarse (only 2 × MAX_PITS buckets) but cheap.
    history: [[u32; MAX_PITS]; 2],
    /// Lazily-built in-memory endgame table (used in exact mode when no
    /// precomputed tablebase is supplied).
    endgame: Endgame,
    /// Optional precomputed offline tablebase; when present it overrides the
    /// lazy endgame for positions within its seed cap.
    tb: Option<&'a Tablebase>,
    nodes: u64,
    budget: u64,
    aborted: bool,
}

impl<'a> Searcher<'a> {
    fn new(rules: Rules, budget: u64, endgame_cutoff: u32, tb: Option<&'a Tablebase>) -> Searcher<'a> {
        Searcher {
            rules,
            tt: TtMap::default(),
            tt_cap: TT_MAX_ENTRIES,
            history: [[0; MAX_PITS]; 2],
            endgame: Endgame::new(rules, endgame_cutoff),
            tb,
            nodes: 0,
            budget,
            aborted: false,
        }
    }

    /// Transposition-table key: the seed counts packed **mover-first**, plus a
    /// depth marker (255 for exact searches so they never collide with
    /// depth-limited entries).
    ///
    /// Packing the side to move's cells first (instead of P0-then-P1 plus a turn
    /// bit) canonicalizes **mirror positions**: Kalah is player-symmetric, so a
    /// position where the mover holds pits `A` against `B` has the same value no
    /// matter which player the mover is. Both mirrors collapse to one entry —
    /// stored values are already mover-perspective (negamax) and best moves are
    /// mover-local pit indices, so entries transfer between mirrors verbatim.
    ///
    /// In exact mode the key packs **only the pits**, not the stores: the stored
    /// value is the store-independent future differential `g` (see the
    /// `±store_diff` transform in [`Self::search`]). In depth-limited mode the
    /// heuristic depends on the stores, so each side's store is included after
    /// its pits.
    ///
    /// Returns `None` when the packed key would not fit in a `u128` (only very
    /// large boards or extremely full pits).
    fn key(b: &Board, depth: Option<u32>) -> Option<u128> {
        let n = b.pits_per_side();
        let cells = b.cells();
        let store_independent = depth.is_none();
        let included = if store_independent { 2 * n } else { 2 * n + 2 };
        // Bits: cells (CELL_BITS each) + 8 depth-marker bits.
        if included as u32 * CELL_BITS + 8 > 128 {
            return None;
        }
        let mut packed: u128 = 0;
        let mut push = |c: u8| -> bool {
            if (c as u32) >= (1 << CELL_BITS) {
                return false;
            }
            packed = (packed << CELL_BITS) | c as u128;
            true
        };
        let mover = b.turn();
        for p in [mover, mover.other()] {
            for i in 0..n {
                if !push(cells[b.pit_global(p, i)]) {
                    return None;
                }
            }
            if !store_independent && !push(b.store(p)) {
                return None;
            }
        }
        let depth_marker: u128 = match depth {
            None => 255,
            Some(d) => d.min(254) as u128,
        };
        packed = (packed << 8) | depth_marker;
        Some(packed)
    }

    /// Generate legal moves for the side to move into `buf`, ordered to improve
    /// alpha-beta pruning, and return the count. Ordering (all O(1) per move, no
    /// board cloning or sow simulation):
    ///   1. the transposition-table move,
    ///   2. moves that land their last seed in the store on the first lap
    ///      (an extra turn): `seeds == distance_to_store`,
    ///   3. remaining pits, those nearer the store first.
    ///
    /// The single-lap extra-turn test is exact for the common case and only ever
    /// affects ordering (never correctness), so multi-lap cases are ignored.
    fn ordered_moves(&self, b: &Board, tt_move: Option<usize>, buf: &mut [u8; MAX_PITS]) -> usize {
        let p = b.turn();
        let n = b.pits_per_side();
        let cells = b.cells();
        let mut count = 0;
        for i in 0..n {
            if cells[b.pit_global(p, i)] > 0 {
                buf[count] = i as u8;
                count += 1;
            }
        }
        let hist = &self.history[Self::pidx(p)];
        let moves = &mut buf[..count];
        moves.sort_by_key(|&mv| {
            let i = mv as usize;
            let seeds = cells[b.pit_global(p, i)] as usize;
            let mut rank = 0i32;
            if Some(i) == tt_move {
                rank -= 1_000_000;
            }
            // Distance from this pit to the player's own store is `n - i` for
            // both players; a single-lap sow lands in the store when equal.
            if seeds == n - i {
                rank -= 10_000;
            }
            // History: quiet moves that have caused cutoffs sort earlier. Capped
            // so it never outranks the TT or extra-turn moves.
            rank -= hist[i].min(8_000) as i32;
            rank -= i as i32; // tie-break: prefer pits nearer the store
            rank
        });
        count
    }

    fn pidx(p: Player) -> usize {
        match p {
            Player::P0 => 0,
            Player::P1 => 1,
        }
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
        // Exact-mode endgame cutoff: once few seeds remain in play, the exact
        // margin is the banked store difference plus the store-independent `g`.
        // A precomputed tablebase (O(1) lookup) takes priority over the lazy one.
        if depth.is_none() {
            let t = b.seeds_in_play();
            if let Some(tb) = self.tb {
                if t <= tb.cap() {
                    let p = b.turn();
                    let sd = b.store(p) as i32 - b.store(p.other()) as i32;
                    return sd + tb.lookup(b).expect("position within tablebase cap") as i32;
                }
            } else if t <= self.endgame.cutoff() {
                let p = b.turn();
                let sd = b.store(p) as i32 - b.store(p.other()) as i32;
                return sd + self.endgame.g(b) as i32;
            }
        }
        if depth == Some(0) {
            return self.heuristic(b);
        }

        // In exact mode the TT stores the store-independent value `g = M - sd`
        // (`sd` = banked store difference). Adding `sd` back reconstructs this
        // position's true margin; in depth-limited mode `sd` is 0 (no transform).
        let sd = if depth.is_none() {
            let p = b.turn();
            b.store(p) as i32 - b.store(p.other()) as i32
        } else {
            0
        };

        let key = Self::key(b, depth);
        let alpha_orig = alpha;
        let mut tt_move = None;
        if let Some(k) = key {
            if let Some(e) = self.tt.get(&k) {
                let val = e.value + sd;
                match e.flag {
                    Flag::Exact => return val,
                    Flag::Lower => alpha = alpha.max(val),
                    Flag::Upper => beta = beta.min(val),
                }
                if alpha >= beta {
                    return val;
                }
                tt_move = if e.best == 255 { None } else { Some(e.best as usize) };
            }
        }

        // Generate and order moves on the stack (no per-node allocation).
        let mut buf = [0u8; MAX_PITS];
        let count = self.ordered_moves(b, tt_move, &mut buf);
        let child_depth = depth.map(|d| d - 1);
        let p = b.turn();

        let mut best = i32::MIN;
        let mut best_move = None;
        for (idx, &mv) in buf[..count].iter().enumerate() {
            let mv = mv as usize;
            let r = b.apply(&self.rules, mv);

            // Principal Variation Search: search the first move with the full
            // window; probe the rest with a null window and only re-search the
            // few that beat alpha. For extra-turn moves the same player
            // continues, so the perspective and window are not negated.
            let v = if r.extra_turn {
                if idx == 0 {
                    self.search(&r.board, alpha, beta, child_depth)
                } else {
                    let probe = self.search(&r.board, alpha, alpha + 1, child_depth);
                    if probe > alpha && probe < beta {
                        self.search(&r.board, alpha, beta, child_depth)
                    } else {
                        probe
                    }
                }
            } else if idx == 0 {
                -self.search(&r.board, -beta, -alpha, child_depth)
            } else {
                let probe = -self.search(&r.board, -alpha - 1, -alpha, child_depth);
                if probe > alpha && probe < beta {
                    -self.search(&r.board, -beta, -alpha, child_depth)
                } else {
                    probe
                }
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
                // Beta cutoff: reward this move in the history table.
                self.history[Self::pidx(p)][mv] += 1;
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
        // entries are retained (and overwritten) so most of the cache stays useful.
        if let Some(k) = key {
            if self.tt.len() < self.tt_cap || self.tt.contains_key(&k) {
                self.tt.insert(
                    k,
                    TtEntry {
                        // Store the store-independent value (subtract `sd`).
                        value: best - sd,
                        flag,
                        best: best_move.map_or(255, |m| m as u8),
                    },
                );
            }
        }
        best
    }

    /// Follow stored best moves to reconstruct the principal variation, starting
    /// *from* `start`. Uses the main transposition table above the endgame cutoff
    /// and the tablebase (when present) within it, so the line extends to the end
    /// of the game. Capped to avoid pathological loops.
    fn follow_pv(&self, start: &Board, depth: Option<u32>, cap: usize) -> Vec<usize> {
        let mut pv = Vec::new();
        let mut cur = *start;
        let mut d = depth;
        while !cur.is_terminal() && pv.len() < cap {
            if d == Some(0) {
                break;
            }
            // Prefer a best move stored in the main TT; otherwise (inside the
            // endgame tablebase) derive it from the table.
            let from_tt = Self::key(&cur, d)
                .and_then(|k| self.tt.get(&k))
                .and_then(|e| (e.best != 255).then_some(e.best as usize));
            let mv = match from_tt.or_else(|| self.best_move_via_tb(&cur)) {
                Some(mv) => mv,
                None => break,
            };
            pv.push(mv);
            let r = cur.apply(&self.rules, mv);
            cur = r.board;
            d = d.map(|x| x - 1);
        }
        pv
    }

    /// Best move at a position covered by the endgame tablebase, by evaluating
    /// each child's `store_diff ± g`. `None` if there is no tablebase, the
    /// position is outside its cap, or there are no moves.
    fn best_move_via_tb(&self, b: &Board) -> Option<usize> {
        let tb = self.tb?;
        if b.seeds_in_play() > tb.cap() {
            return None;
        }
        let p = b.turn();
        let o = p.other();
        let mut best_v = i32::MIN;
        let mut best_m = None;
        for i in 0..b.pits_per_side() {
            if b.cells()[b.pit_global(p, i)] == 0 {
                continue;
            }
            let r = b.apply(&self.rules, i);
            let g = tb.lookup(&r.board)? as i32;
            let sd = r.board.store(p) as i32 - r.board.store(o) as i32;
            let v = if r.extra_turn { sd + g } else { sd - g };
            if v > best_v {
                best_v = v;
                best_m = Some(i);
            }
        }
        best_m
    }

    /// Solve the position with a single alpha-beta pass over the root moves.
    ///
    /// Moves are tried best-ordered first; `alpha` rises as better moves are
    /// found. Once a move establishes the best value, inferior moves fail low
    /// against the raised window and yield an *upper bound* rather than an exact
    /// value — they are flagged `bound` in the table. This avoids the ~N× cost
    /// of proving an exact value for every move while keeping the game value,
    /// best move, and principal variation exact.
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

        // Search the likely-best move first so its subtree warms the TT and the
        // raised alpha lets the siblings fail low quickly.
        let mut buf = [0u8; MAX_PITS];
        let count = self.ordered_moves(b, None, &mut buf);

        let mut evals = Vec::with_capacity(count);
        let mut alpha = -INF;
        let mut best = i32::MIN;
        let mut best_move = None;
        let child_depth = depth.map(|d| d - 1);

        for &mv in &buf[..count] {
            let mv = mv as usize;
            let r = b.apply(&self.rules, mv);
            // beta stays at +INF (no cutoff at the root), so a child only ever
            // fails low; a returned value `<= alpha` is an upper bound.
            let v = if r.extra_turn {
                self.search(&r.board, alpha, INF, child_depth)
            } else {
                -self.search(&r.board, -INF, -alpha, child_depth)
            };
            if self.aborted {
                // Caller will discard this partial result.
                break;
            }
            let is_bound = v <= alpha && best_move.is_some();
            evals.push(MoveEval {
                pit: mv,
                value: v,
                bound: is_bound,
                extra_turn: r.extra_turn,
                captured: r.captured,
            });
            if v > best {
                best = v;
                best_move = Some(mv);
            }
            alpha = alpha.max(v);
        }

        // Exact values first, then by value; the best move sorts to the top.
        evals.sort_by(|a, c| a.bound.cmp(&c.bound).then(c.value.cmp(&a.value)));

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

/// Largest board (pits per side) for which the endgame table is worthwhile: its
/// `≤ ENDGAME_CUTOFF`-seed config space stays bounded. Bigger boards have an
/// explosive endgame space, so they skip it.
const ENDGAME_MAX_PITS: usize = 6;

/// Only build the endgame table when the caller has committed to a substantial
/// exact search; small-budget probing attempts skip it so they stay cheap.
const ENDGAME_MIN_BUDGET: u64 = 50_000_000;

/// Analyze `board`: attempt an exact solve within `node_budget`; if that budget
/// is exceeded, fall back to a depth-limited heuristic search of
/// `fallback_depth` plies.
pub fn analyze(board: &Board, rules: Rules, node_budget: u64, fallback_depth: u32) -> Analysis {
    analyze_with_tb(board, rules, node_budget, fallback_depth, None)
}

/// Like [`analyze`], but consulting a precomputed offline [`Tablebase`] for the
/// endgame. The tablebase is used only if its board size matches `board`.
pub fn analyze_with_tb(
    board: &Board,
    rules: Rules,
    node_budget: u64,
    fallback_depth: u32,
    tb: Option<&Tablebase>,
) -> Analysis {
    let tb = tb.filter(|t| t.pits_per_side() == board.pits_per_side());

    // With a tablebase, the endgame cutoff is the tablebase's own cap. Otherwise
    // enable the lazy endgame only for small boards under a real exact budget
    // (a cutoff of 0 disables it: no non-terminal position has 0 seeds in play).
    let endgame_cutoff = if tb.is_some() {
        0
    } else if board.pits_per_side() <= ENDGAME_MAX_PITS && node_budget >= ENDGAME_MIN_BUDGET {
        ENDGAME_CUTOFF
    } else {
        0
    };

    let mut exact = Searcher::new(rules, node_budget, endgame_cutoff, tb);
    let result = exact.analyze_root(board, None);
    if !exact.aborted {
        return result;
    }

    // Exact search ran out of budget — fall back to a heuristic search.
    let mut limited = Searcher::new(rules, u64::MAX, 0, None);
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
    fn kalah_6_2_exact_value_regression() {
        // Solves exactly in well under a second with the optimized search.
        let b = Board::start(6, 2, Player::P0).unwrap();
        let a = analyze(&b, Rules::default(), u64::MAX, 12);
        assert!(a.exact, "Kalah(6,2) should solve exactly");
        assert_eq!(a.value, 6, "Kalah(6,2) is a first-player win by 6 seeds");
        assert_eq!(a.outcome(), Outcome::Win);
        assert_eq!(a.best_move, Some(4));
        assert!(!a.pv.is_empty());
    }

    #[test]
    fn kalah_5_3_exact_value_regression() {
        // Previously timed out (>60s); now ~2M nodes / ~1s.
        let b = Board::start(5, 3, Player::P0).unwrap();
        let a = analyze(&b, Rules::default(), u64::MAX, 12);
        assert!(a.exact);
        assert_eq!(a.value, 6, "Kalah(5,3) is a first-player win by 6 seeds");
        assert_eq!(a.best_move, Some(2));
    }

    #[test]
    #[ignore = "exact solve of Kalah(6,3) takes ~1 minute; run with --ignored"]
    fn kalah_6_3_exact_value_regression() {
        // A heavier board (~133M nodes). Pins the exact result as a regression.
        let b = Board::start(6, 3, Player::P0).unwrap();
        let a = analyze(&b, Rules::default(), u64::MAX, 12);
        assert!(a.exact, "Kalah(6,3) should solve exactly");
        assert_eq!(a.value, 2, "Kalah(6,3) is a first-player win by 2 seeds");
        assert_eq!(a.outcome(), Outcome::Win);
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
        // The best move sorts first and is reported exactly (not a bound).
        let top = a.move_evals[0];
        assert_eq!(Some(top.pit), a.best_move);
        assert!(!top.bound);
        assert_eq!(top.value, a.value);
        // Exact evaluations are listed before bounded ones.
        let first_bound = a.move_evals.iter().position(|m| m.bound);
        if let Some(idx) = first_bound {
            assert!(a.move_evals[idx..].iter().all(|m| m.bound));
        }
    }
}
