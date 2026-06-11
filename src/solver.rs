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

/// A decoded transposition-table hit.
#[derive(Clone, Copy)]
struct TtProbe {
    value: i32,
    flag: Flag,
    best: Option<usize>,
}

/// Number of bits used to encode a single cell in the packed key. Cells holding
/// 64 or more seeds (or boards too wide to fit) skip the transposition table.
const CELL_BITS: u32 = 6;

/// Bits of payload packed into the low end of each `u128` slot alongside the
/// board key (high bits): value (`i16`, 16 bits) + flag (2 bits) + best-move
/// (4 bits). Keys needing more than `128 - PAYLOAD_BITS` bits are uncacheable.
const PAYLOAD_BITS: u32 = 22;
const PAYLOAD_MASK: u128 = (1 << PAYLOAD_BITS) - 1;

/// Maximum slots for the exact table (each is 16 B). On 64-bit hosts 2^28 slots
/// is 4.3 GB and holds well over the std map's 180M-entry cap — in half the
/// memory — so far fewer positions are dropped and re-searched. (Peak during the
/// final doubling is ≈6.4 GB, within budget.)
#[cfg(target_pointer_width = "64")]
const TT_MAX_SLOTS_EXACT: usize = 1 << 28;

/// On 32-bit hosts (notably wasm32 in the browser, where the whole address space
/// is ≤ 4 GB and over-allocating crashes the page) the table is capped at 2^24
/// slots = 256 MB. Beyond it the search just caches less — slower, never OOM.
#[cfg(not(target_pointer_width = "64"))]
const TT_MAX_SLOTS_EXACT: usize = 1 << 24;

/// Grow (while below the cap) when entries reach this fraction (×/20) of slots,
/// keeping probe chains short and the table proportional to the working set so
/// light searches stay compact and cache-resident.
const TT_GROW_NUM: usize = 12; // 12/20 = 0.60

/// Linear-probe bound. Below the grow threshold chains are a few slots; this is
/// a safety valve once the capped table fills (a position past it goes uncached
/// — correct, just not cached — instead of scanning the whole array).
const TT_MAX_PROBE: usize = 48;

/// Mix a packed key to a `u64` for slot selection (same constants as the shared
/// `U128Hasher`).
#[inline]
fn tt_mix(key: u128) -> u64 {
    let mut h = (key as u64) ^ ((key >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 29;
    h
}

/// Advise the kernel to back `buf` with transparent huge pages. The multi-GB TT
/// is accessed randomly; 2 MB pages cut TLB misses by ~512× versus 4 KB pages,
/// which the std allocator cannot arrange on a `madvise`-only THP system. Raw
/// syscall to avoid any external dependency; best-effort (ignored on failure or
/// non-Linux/x86-64).
#[inline]
fn advise_hugepages(buf: &[u128]) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    unsafe {
        const SYS_MADVISE: usize = 28;
        const MADV_HUGEPAGE: usize = 14;
        const HP: usize = 2 * 1024 * 1024; // 2 MB huge page
        // `madvise(MADV_HUGEPAGE)` needs a page-aligned range, and huge pages only
        // back 2 MB-aligned spans. The allocator's pointer carries a small header
        // offset, so align the start up and the length down to 2 MB.
        let start = buf.as_ptr() as usize;
        let end = start + std::mem::size_of_val(buf);
        let aligned = (start + HP - 1) & !(HP - 1);
        if end <= aligned {
            return;
        }
        let alen = (end - aligned) & !(HP - 1);
        if alen == 0 {
            return;
        }
        let _ret: isize;
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MADVISE => _ret,
            in("rdi") aligned,
            in("rsi") alen,
            in("rdx") MADV_HUGEPAGE,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    let _ = buf;
}

/// A dense, open-addressing (linear-probing) transposition table.
///
/// A flat `Vec<u128>`; each slot fuses the board key (high bits) with a 22-bit
/// payload (value/flag/best). At 16 B/slot it is ~2× denser than the std map's
/// 16-aligned `(u128, TtEntry)` slots, so more of the working set fits in the
/// 260 MB L3 and far more positions are cached before the table fills (in half
/// the memory). A lookup reads only the entry array (no separate control bytes)
/// and — because four slots share a 64-byte line — the short probe chains at the
/// table's working load usually stay on one line; a child's line is prefetched
/// before recursing into it. The table doubles (rehash) up to `max_slots`,
/// staying proportional to the working set so small searches keep good locality.
/// It never overwrites a different key, so a cached subtree is never recomputed.
/// The buffer is `madvise`d for huge pages (a no-op where THP is unavailable).
/// An empty slot is `0`, which no real entry can equal (every key has a non-zero
/// depth marker in its low byte).
struct Tt {
    slots: Vec<u128>,
    mask: usize,
    occupancy: usize,
    max_slots: usize,
}

impl Tt {
    fn new(max_slots: usize) -> Tt {
        let n = (1usize << 14).min(max_slots); // start at 16K slots ≈ 256 KB
        let slots = vec![0u128; n];
        advise_hugepages(&slots);
        Tt {
            slots,
            mask: n - 1,
            occupancy: 0,
            max_slots,
        }
    }

    #[inline]
    fn encode(key: u128, value: i32, flag: Flag, best: Option<usize>) -> u128 {
        let v = (value.clamp(i16::MIN as i32, i16::MAX as i32) as i16) as u16 as u128;
        let f: u128 = match flag {
            Flag::Exact => 1,
            Flag::Lower => 2,
            Flag::Upper => 3,
        };
        let bm = best.map_or(15u128, |m| (m as u128) & 15);
        (key << PAYLOAD_BITS) | v | (f << 16) | (bm << 18)
    }

    #[inline]
    fn decode(entry: u128) -> TtProbe {
        let payload = entry & PAYLOAD_MASK;
        TtProbe {
            value: (payload as u16) as i16 as i32,
            flag: match (payload >> 16) & 3 {
                1 => Flag::Exact,
                2 => Flag::Lower,
                _ => Flag::Upper,
            },
            best: match (payload >> 18) & 15 {
                15 => None,
                b => Some(b as usize),
            },
        }
    }

    /// Prefetch the cache line for `key`'s home slot (issued for a child before
    /// recursing, to overlap the DRAM fetch with the child's entry work).
    #[inline]
    fn prefetch(&self, key: u128) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let idx = (tt_mix(key) as usize) & self.mask;
            core::arch::x86_64::_mm_prefetch(
                self.slots.as_ptr().add(idx) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    #[inline]
    fn get(&self, key: u128) -> Option<TtProbe> {
        let mut i = (tt_mix(key) as usize) & self.mask;
        for _ in 0..TT_MAX_PROBE {
            let e = self.slots[i];
            if e == 0 {
                return None;
            }
            if e >> PAYLOAD_BITS == key {
                return Some(Self::decode(e));
            }
            i = (i + 1) & self.mask;
        }
        None
    }

    fn store(&mut self, key: u128, value: i32, flag: Flag, best: Option<usize>) {
        if self.occupancy * 20 >= self.slots.len() * TT_GROW_NUM && self.slots.len() < self.max_slots
        {
            self.grow();
        }
        let entry = Self::encode(key, value, flag, best);
        let mut i = (tt_mix(key) as usize) & self.mask;
        for _ in 0..TT_MAX_PROBE {
            let e = self.slots[i];
            if e == 0 {
                self.slots[i] = entry;
                self.occupancy += 1;
                return;
            }
            if e >> PAYLOAD_BITS == key {
                self.slots[i] = entry; // refresh in place
                return;
            }
            i = (i + 1) & self.mask;
        }
        // Probe chain saturated (only near a full capped table): leave uncached.
    }

    fn grow(&mut self) {
        let new_len = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![0u128; new_len]);
        advise_hugepages(&self.slots);
        self.mask = new_len - 1;
        for &e in &old {
            if e != 0 {
                let mut i = (tt_mix(e >> PAYLOAD_BITS) as usize) & self.mask;
                while self.slots[i] != 0 {
                    i = (i + 1) & self.mask;
                }
                self.slots[i] = e;
            }
        }
    }
}

/// Seeds-in-play threshold at or below which the exact search consults the
/// store-independent endgame table instead of recursing. Chosen empirically as
/// a balance between endgame-table size and how much of the forward tree it
/// prunes.
const ENDGAME_CUTOFF: u32 = 14;

/// Seeds-in-play cutoff for the **play** engine's lazy endgame oracle when no
/// precomputed tablebase is supplied. Smaller than the solver's cutoff because
/// the lazy `g` is recomputed per move during a game; this keeps it cheap while
/// still giving perfect endgames on any board size.
const PLAY_ENDGAME_CUTOFF: u32 = 12;

struct Searcher<'a> {
    rules: Rules,
    tt: Tt,
    /// History heuristic: cumulative beta-cutoff weight per `[side][pit]`, used
    /// to order quiet moves. Coarse (only 2 × MAX_PITS buckets) but cheap.
    history: [[u32; MAX_PITS]; 2],
    /// Lazily-built in-memory endgame table (used in exact mode when no
    /// precomputed tablebase is supplied).
    endgame: Endgame,
    /// Optional precomputed offline tablebase; when present it overrides the
    /// lazy endgame for positions within its seed cap.
    tb: Option<&'a Tablebase>,
    /// When true, the depth-limited search runs a quiescence search at the
    /// horizon (extending forcing moves) instead of evaluating immediately.
    quiesce: bool,
    nodes: u64,
    budget: u64,
    aborted: bool,
}

impl<'a> Searcher<'a> {
    fn new(
        rules: Rules,
        budget: u64,
        endgame_cutoff: u32,
        tb: Option<&'a Tablebase>,
        tt_max_slots: usize,
    ) -> Searcher<'a> {
        Searcher {
            rules,
            tt: Tt::new(tt_max_slots),
            history: [[0; MAX_PITS]; 2],
            endgame: Endgame::new(rules, endgame_cutoff),
            tb,
            quiesce: false,
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
        let hist = &self.history[Self::pidx(p)];
        // Player `p`'s pits are contiguous starting at this global index, so we
        // can index `base + i` and skip the per-access `pit_global` match.
        let base = b.pit_global(p, 0);
        let tt_mv = tt_move.map_or(255u8, |m| m as u8);

        // Score each legal move exactly once (lower score = tried earlier), then
        // insertion-sort the few moves by their precomputed scores. Computing the
        // score once — rather than inside a `sort_by_key` comparator that re-runs
        // it on every comparison — is a large per-node saving (move ordering ran
        // at every node).
        let mut count = 0;
        let mut scores = [0i32; MAX_PITS];
        for i in 0..n {
            let seeds = cells[base + i] as usize;
            if seeds == 0 {
                continue;
            }
            let mut score = -(i as i32); // tie-break: prefer pits nearer the store
            if i as u8 == tt_mv {
                score -= 1_000_000;
            }
            // Distance from this pit to the player's own store is `n - i` for
            // both players; a single-lap sow lands in the store when equal.
            if seeds == n - i {
                score -= 10_000;
            }
            // History: quiet moves that have caused cutoffs sort earlier. Capped
            // so it never outranks the TT or extra-turn moves.
            score -= hist[i].min(8_000) as i32;
            scores[count] = score;
            buf[count] = i as u8;
            count += 1;
        }
        for a in 1..count {
            let (smv, ssc) = (buf[a], scores[a]);
            let mut j = a;
            while j > 0 && scores[j - 1] > ssc {
                scores[j] = scores[j - 1];
                buf[j] = buf[j - 1];
                j -= 1;
            }
            scores[j] = ssc;
            buf[j] = smv;
        }
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

    /// If `b` falls within the tablebase (or lazy endgame) cutoff, return its
    /// exact margin from the side to move — `× STORE_WEIGHT` when `scaled` (to
    /// match the depth-limited heuristic's units), otherwise the raw seed margin.
    fn resolved(&mut self, b: &Board, scaled: bool) -> Option<i32> {
        let t = b.seeds_in_play();
        let g = if let Some(tb) = self.tb {
            (t <= tb.cap()).then(|| tb.lookup(b).expect("position within tablebase cap") as i32)
        } else if t <= self.endgame.cutoff() {
            Some(self.endgame.g(b) as i32)
        } else {
            None
        };
        g.map(|g| {
            let p = b.turn();
            let margin = (b.store(p) as i32 - b.store(p.other()) as i32) + g;
            if scaled {
                margin * STORE_WEIGHT
            } else {
                margin
            }
        })
    }

    /// Quiescence search: at the horizon, instead of trusting the static eval in
    /// the middle of a tactical sequence, keep searching the **forcing** moves —
    /// extra-turn moves (a free continuation) and captures (a material swing) —
    /// until the position is quiet. Both kinds strictly reduce the seeds in play,
    /// so this terminates. Quiet moves are not searched (the eval is taken as a
    /// stand-pat floor, as in standard quiescence).
    fn quiesce(&mut self, b: &Board, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if self.nodes > self.budget {
            self.aborted = true;
            return 0;
        }
        if b.is_terminal() {
            return b.terminal_margin(b.turn()) * STORE_WEIGHT;
        }
        if let Some(v) = self.resolved(b, true) {
            return v;
        }
        let stand = self.heuristic(b);
        if stand >= beta {
            return stand;
        }
        if stand > alpha {
            alpha = stand;
        }
        let p = b.turn();
        let n = b.pits_per_side();
        let base = b.pit_global(p, 0);
        for i in 0..n {
            if b.cells()[base + i] == 0 {
                continue;
            }
            let r = b.apply(&self.rules, i);
            if !r.extra_turn && !r.captured {
                continue; // quiet move — not part of the tactical sequence
            }
            // Extra-turn moves keep the side to move (no negation); captures pass.
            let v = if r.extra_turn {
                self.quiesce(&r.board, alpha, beta)
            } else {
                -self.quiesce(&r.board, -beta, -alpha)
            };
            if self.aborted {
                return 0;
            }
            if v >= beta {
                return v;
            }
            if v > alpha {
                alpha = v;
            }
        }
        alpha
    }

    /// Negamax + alpha-beta. `depth == None` searches to terminal (exact);
    /// `depth == Some(d)` searches `d` plies then applies the heuristic.
    fn search(&mut self, b: &Board, mut alpha: i32, mut beta: i32, depth: Option<u32>) -> i32 {
        self.nodes += 1;
        if self.nodes > self.budget {
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
        // Endgame cutoff (both modes): once few seeds remain, a tablebase-resolved
        // line is the proven outcome — return it (scaled in play mode), giving
        // perfect endgame play without recursing.
        if let Some(v) = self.resolved(b, depth.is_some()) {
            return v;
        }
        if depth == Some(0) {
            return if self.quiesce {
                self.quiesce(b, alpha, beta)
            } else {
                self.heuristic(b)
            };
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
            if let Some(e) = self.tt.get(k) {
                let val = e.value + sd;
                match e.flag {
                    Flag::Exact => return val,
                    Flag::Lower => alpha = alpha.max(val),
                    Flag::Upper => beta = beta.min(val),
                }
                if alpha >= beta {
                    return val;
                }
                tt_move = e.best;
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

            // Prefetch this child's TT slot now so its DRAM fetch overlaps the
            // child's own entry work (terminal/cutoff checks) before it probes.
            if let Some(ck) = Self::key(&r.board, child_depth) {
                self.tt.prefetch(ck);
            }

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
        // Cache the store-independent value (subtract `sd`).
        if let Some(k) = key {
            self.tt.store(k, best - sd, flag, best_move);
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
                .and_then(|k| self.tt.get(k))
                .and_then(|e| e.best);
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

    let mut exact = Searcher::new(rules, node_budget, endgame_cutoff, tb, TT_MAX_SLOTS_EXACT);
    let result = exact.analyze_root(board, None);
    if !exact.aborted {
        if std::env::var_os("MANCALA_DIAG").is_some() {
            eprintln!(
                "DIAG tt_entries={} tt_slots={} nodes={}",
                exact.tt.occupancy,
                exact.tt.slots.len(),
                exact.nodes
            );
        }
        return result;
    }
    // Release the (large) exact table before the fallback allocates its own.
    drop(exact);

    // Exact search ran out of budget — fall back to a heuristic search.
    let mut limited = Searcher::new(rules, u64::MAX, 0, None, 1 << 22);
    limited.analyze_root(board, Some(fallback_depth))
}

/// A thinking limit for the play search.
#[derive(Clone, Copy, Debug)]
pub enum Limit {
    /// Search exactly this many plies.
    Depth(u32),
    /// Iteratively deepen until this many search nodes have been visited,
    /// returning the deepest fully-completed iteration (an anytime search, so
    /// two engines can be compared on equal *work* rather than equal depth).
    Nodes(u64),
}

/// Play search: depth-limited / node-budgeted alpha-beta with the heuristic at
/// the horizon. `use_endgame` adds the exact endgame oracle (tablebase if given,
/// else the lazy endgame) for perfect endgames on any board size; `quiesce` adds
/// a quiescence search through forcing moves. With `use_endgame == false` and
/// `quiesce == false` this is the plain heuristic player.
pub fn play_search(
    board: &Board,
    rules: Rules,
    tb: Option<&Tablebase>,
    limit: Limit,
    use_endgame: bool,
    quiesce: bool,
) -> Analysis {
    let tb = if use_endgame {
        tb.filter(|t| t.pits_per_side() == board.pits_per_side())
    } else {
        None
    };
    let endgame_cutoff = if use_endgame && tb.is_none() {
        PLAY_ENDGAME_CUTOFF
    } else {
        0
    };
    let budget = match limit {
        Limit::Nodes(n) => n,
        Limit::Depth(_) => u64::MAX,
    };
    let mut s = Searcher::new(rules, budget, endgame_cutoff, tb, 1 << 24);
    s.quiesce = quiesce;

    match limit {
        Limit::Depth(d) => s.analyze_root(board, Some(d.max(1))),
        Limit::Nodes(_) => {
            // Iterative deepening: keep the deepest iteration that finished within
            // budget (nodes accumulate across iterations, so the budget bounds the
            // whole search). If even depth 1 overruns a tiny budget, keep it anyway.
            let mut best: Option<Analysis> = None;
            for depth in 1..=64u32 {
                let a = s.analyze_root(board, Some(depth));
                let completed = !s.aborted;
                if a.best_move.is_some() && (completed || best.is_none()) {
                    best = Some(a);
                }
                if !completed {
                    break;
                }
            }
            best.expect("at least one iteration runs")
        }
    }
}

/// A reusable exact solver that keeps one transposition table warm across many
/// root positions. Solving several nearby positions (e.g. an opening tree) in
/// turn is far cheaper than independent solves because each later search reuses
/// the table the earlier ones populated. Used to build opening books.
pub struct Solver<'a> {
    searcher: Searcher<'a>,
}

impl<'a> Solver<'a> {
    /// A solver that consults `tb` for the endgame (required — high-seed root
    /// solves are only tractable with a tablebase).
    pub fn new(rules: Rules, tb: &'a Tablebase) -> Solver<'a> {
        Solver {
            searcher: Searcher::new(rules, u64::MAX, 0, Some(tb), TT_MAX_SLOTS_EXACT),
        }
    }

    /// Exactly solve `board`, returning its game value (seed margin) and best
    /// move. Reuses the warm transposition table from previous calls.
    pub fn solve(&mut self, board: &Board) -> (i32, Option<usize>) {
        self.searcher.nodes = 0;
        self.searcher.aborted = false;
        let a = self.searcher.analyze_root(board, None);
        (a.value, a.best_move)
    }
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
