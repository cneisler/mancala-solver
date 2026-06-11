//! Offline endgame tablebase.
//!
//! Stores the store-independent future differential `g(pits, side)` (see
//! [`crate::endgame`]) for **every** pit layout with at most `cap` seeds in
//! play, on a fixed board size. Unlike the in-memory [`crate::endgame::Endgame`]
//! (built lazily during a single search), a tablebase is built once by a
//! separate command, written to disk, then memory-loaded and queried in O(1)
//! during any later search.
//!
//! # Indexing (mirror-canonical)
//!
//! A pit layout is the `2N` pit counts packed **mover-first** (the side to
//! move's pits, then the opponent's). Kalah is player-symmetric, so the value of
//! "the mover holds `A` against `B`" is independent of *which* player the mover
//! is — indexing mover-first makes the two mirror positions share one entry and
//! **halves the table** relative to keying on (P0, P1, side).
//!
//! Layouts with total ≤ `cap` are enumerated as integer compositions and mapped
//! to a dense index via the combinatorial number system, so the on-disk form is
//! just a flat `[i16]` — no keys stored. `data[rank(mover_pits ++ opp_pits)]`
//! is `g` from the mover's perspective.
//!
//! # File format (little-endian)
//!
//! ```text
//! magic        : 8 bytes  = b"KALATB02"
//! pits_per_side: u32
//! seeds_cap    : u32
//! num_layouts  : u64       (= C(cap + 2N, 2N))
//! data         : i16 * num_layouts
//! ```

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use crate::board::{Board, Rules, MAX_CELLS};
use crate::hash::U128Map;

const MAGIC: &[u8; 8] = b"KALATB02";

/// Number of shards in the concurrent build memo (low lock contention).
const BUILD_SHARDS: usize = 512;

/// Mix a packed key down to a `u64` for shard selection.
fn mix(key: u128) -> u64 {
    let mut h = (key as u64) ^ ((key >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 29;
    h
}

/// Store-independent, mirror-canonical key: the mover's pits then the
/// opponent's, 6 bits per pit (no turn bit — mirrors share the key, and `g` is
/// mover-perspective so values transfer verbatim).
fn g_key(b: &Board) -> u128 {
    let n = b.pits_per_side();
    let cells = b.cells();
    let mover = b.turn();
    let mut k = 0u128;
    for p in [mover, mover.other()] {
        for i in 0..n {
            k = (k << 6) | cells[b.pit_global(p, i)] as u128;
        }
    }
    k
}

/// Compute the store-independent future differential `g` for `b`, memoised into
/// a sharded concurrent table. Locks are held only for the brief get/insert, not
/// across recursion; computation is idempotent, so concurrent threads racing on
/// the same layout simply recompute the (identical) value.
fn solve_g(b: &Board, rules: &Rules, memo: &[Mutex<U128Map<i16>>]) -> i16 {
    if b.is_terminal() {
        let p = b.turn();
        return b.pit_seeds(p) as i16 - b.pit_seeds(p.other()) as i16;
    }
    let key = g_key(b);
    let shard = &memo[(mix(key) as usize) % memo.len()];
    if let Some(&v) = shard.lock().unwrap().get(&key) {
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
        let r = b.apply(rules, i);
        let gain = r.board.store(p) as i16 - store_before;
        let child = solve_g(&r.board, rules, memo);
        let v = if r.extra_turn { gain + child } else { gain - child };
        best = best.max(v);
    }
    shard.lock().unwrap().insert(key, best);
    best
}

/// Side length of the precomputed Pascal triangle. Ranking a layout only ever
/// indexes `C(n, k)` with `n ≤ cap + 2·pits`; for the supported caps (a pit
/// holds < 64 seeds) and board sizes this stays well under 128.
const PASCAL_N: usize = 128;

/// Lazily-built Pascal triangle `C(n, k)` for `n, k < PASCAL_N`, row-major.
/// Middle entries that overflow `u64` saturate — those huge binomials are never
/// read (rank only uses small-`k` columns), so saturation only avoids a panic
/// during construction.
static PASCAL: OnceLock<Vec<u64>> = OnceLock::new();

fn pascal() -> &'static [u64] {
    PASCAL
        .get_or_init(|| {
            let mut t = vec![0u64; PASCAL_N * PASCAL_N];
            for n in 0..PASCAL_N {
                t[n * PASCAL_N] = 1; // C(n, 0)
                for k in 1..=n {
                    let above = t[(n - 1) * PASCAL_N + k];
                    let left = t[(n - 1) * PASCAL_N + (k - 1)];
                    t[n * PASCAL_N + k] = above.saturating_add(left);
                }
            }
            t
        })
        .as_slice()
}

/// `C(n, k)`. O(1) via the Pascal table for `n < PASCAL_N`; falls back to a
/// `u128` accumulator for the rare larger `n` (only hit by huge custom caps).
#[inline]
fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if (n as usize) < PASCAL_N {
        return pascal()[n as usize * PASCAL_N + k as usize];
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    for i in 0..k {
        num = num * (n - i) as u128 / (i + 1) as u128;
    }
    num as u64
}

/// Number of compositions of `t` into `parts` non-negative parts.
fn comps(t: u64, parts: u64) -> u64 {
    if parts == 0 {
        return if t == 0 { 1 } else { 0 };
    }
    binom(t + parts - 1, parts - 1)
}

/// Number of layouts (compositions into `m` parts) with total **strictly less
/// than** `t`.
fn offset(t: u64, m: u64) -> u64 {
    if t == 0 {
        0
    } else {
        binom(t - 1 + m, m)
    }
}

/// Total number of layouts with total ≤ `cap` over `m` parts: `C(cap + m, m)`.
fn num_layouts(cap: u64, m: u64) -> u64 {
    binom(cap + m, m)
}

/// Dense rank of a pit layout (composition) among all layouts with total ≤ cap.
///
/// Hot path: this runs at every endgame-frontier leaf of an exact search, so it
/// reads the Pascal table directly (one O(1) row lookup per term) rather than
/// going through `comps`/`binom`.
fn rank(pits: &[u8]) -> u64 {
    let p = pascal();
    let m = pits.len() as u64;
    let total: u64 = pits.iter().map(|&x| x as u64).sum();
    let mut r = offset(total, m);
    let mut rem = total;
    for (j, &c) in pits.iter().enumerate() {
        let parts_left = m - 1 - j as u64; // parts after position j
        if parts_left == 0 {
            break;
        }
        // Sum over v in 0..c of comps(rem - v, parts_left)
        //   = C(rem - v + parts_left - 1, parts_left - 1).
        // `parts_left >= 1` here, so the binomial column is fixed at
        // `parts_left - 1` and we only need the table when n < PASCAL_N.
        let col = (parts_left - 1) as usize;
        for v in 0..c as u64 {
            let n = (rem - v + parts_left - 1) as usize;
            r += if n < PASCAL_N {
                p[n * PASCAL_N + col]
            } else {
                comps(rem - v, parts_left)
            };
        }
        rem -= c as u64;
    }
    r
}

/// An endgame tablebase for one board size.
pub struct Tablebase {
    pits_per_side: usize,
    cap: u32,
    num_layouts: u64,
    /// `g` values: `data[side * num_layouts + rank]`.
    data: Vec<i16>,
}

impl Tablebase {
    pub fn pits_per_side(&self) -> usize {
        self.pits_per_side
    }

    pub fn cap(&self) -> u32 {
        self.cap
    }

    /// Number of `g` entries the table would hold for `(pits_per_side, cap)`,
    /// useful for sizing checks before building. Thanks to mirror canonicalization
    /// one entry serves both sides, so this is just the layout count.
    pub fn entry_count(pits_per_side: usize, cap: u32) -> u64 {
        num_layouts(cap as u64, 2 * pits_per_side as u64)
    }

    /// Build the tablebase for every layout with ≤ `cap` seeds in play, using
    /// all available cores. Each enumerated layout is read as (mover pits,
    /// opponent pits); mirror symmetry means no per-side pass is needed.
    pub fn build(rules: Rules, pits_per_side: usize, cap: u32) -> Tablebase {
        let m = 2 * pits_per_side;
        let total = num_layouts(cap as u64, m as u64) as usize;

        // Fill a sharded, store-independent `g` memo in parallel. Each thread
        // owns the layouts whose enumeration index is `≡ t`.
        let memo: Vec<Mutex<U128Map<i16>>> =
            (0..BUILD_SHARDS).map(|_| Mutex::new(U128Map::default())).collect();
        let nthreads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 8);

        std::thread::scope(|scope| {
            let memo = &memo;
            for t in 0..nthreads {
                scope.spawn(move || {
                    let mut pits = vec![0u8; m];
                    let mut idx = 0usize;
                    Self::for_each_layout(cap, &mut pits, &mut |layout| {
                        if idx % nthreads == t {
                            let board = layout_board(pits_per_side, layout);
                            solve_g(&board, &rules, memo);
                        }
                        idx += 1;
                    });
                });
            }
        });

        // Transcribe the memo into the dense, rank-indexed array.
        let mut data = vec![0i16; total];
        let mut pits = vec![0u8; m];
        Self::for_each_layout(cap, &mut pits, &mut |layout| {
            let board = layout_board(pits_per_side, layout);
            // Terminal layouts return before being memoised (their value is
            // the immediate sweep); compute them directly here.
            let v = if board.is_terminal() {
                let p = board.turn();
                board.pit_seeds(p) as i16 - board.pit_seeds(p.other()) as i16
            } else {
                let key = g_key(&board);
                let shard = &memo[(mix(key) as usize) % BUILD_SHARDS];
                *shard.lock().unwrap().get(&key).expect("non-terminal memo filled")
            };
            data[rank(layout) as usize] = v;
        });

        Tablebase {
            pits_per_side,
            cap,
            num_layouts: total as u64,
            data,
        }
    }

    /// Enumerate every composition of total ≤ `cap` into `m` parts, invoking `f`
    /// with the filled `pits` buffer each time.
    fn for_each_layout(cap: u32, pits: &mut [u8], f: &mut impl FnMut(&[u8])) {
        fn rec(idx: usize, remaining: u32, pits: &mut [u8], f: &mut impl FnMut(&[u8])) {
            if idx == pits.len() - 1 {
                // Last pit takes anything from 0..=remaining (≤ cap overall).
                for v in 0..=remaining {
                    pits[idx] = v as u8;
                    f(pits);
                }
                return;
            }
            for v in 0..=remaining {
                pits[idx] = v as u8;
                rec(idx + 1, remaining - v, pits, f);
            }
        }
        // `remaining` is the budget for the whole layout (≤ cap).
        rec(0, cap, pits, f);
    }

    /// Look up `g` (mover-perspective) for a position, if its seed count is
    /// within the table and the board size matches. Returns `None` otherwise.
    /// Mirror-canonical: works for either side to move via the same entry.
    pub fn lookup(&self, b: &Board) -> Option<i16> {
        if b.pits_per_side() != self.pits_per_side || b.seeds_in_play() > self.cap {
            return None;
        }
        let n = self.pits_per_side;
        let cells = b.cells();
        let mover = b.turn();
        let mut pits = [0u8; MAX_CELLS];
        for i in 0..n {
            pits[i] = cells[b.pit_global(mover, i)];
            pits[n + i] = cells[b.pit_global(mover.other(), i)];
        }
        Some(self.data[rank(&pits[..2 * n]) as usize])
    }

    /// Serialize to `path`.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&(self.pits_per_side as u32).to_le_bytes())?;
        w.write_all(&self.cap.to_le_bytes())?;
        w.write_all(&self.num_layouts.to_le_bytes())?;
        for &v in &self.data {
            w.write_all(&v.to_le_bytes())?;
        }
        w.flush()
    }

    /// Load a tablebase from `path`.
    pub fn load(path: &Path) -> io::Result<Tablebase> {
        Self::from_reader(BufReader::new(File::open(path)?))
    }

    /// Load a tablebase from an in-memory byte slice (e.g. a `fetch`ed file in
    /// the browser, where there is no filesystem).
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Tablebase> {
        Self::from_reader(std::io::Cursor::new(bytes))
    }

    /// Parse a tablebase from any reader (see the file-format docs above).
    pub fn from_reader<R: Read>(mut r: R) -> io::Result<Tablebase> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a Kalah tablebase file",
            ));
        }
        let pits_per_side = read_u32(&mut r)? as usize;
        let cap = read_u32(&mut r)?;
        let num_layouts = read_u64(&mut r)?;
        let count = num_layouts as usize;
        let mut data = vec![0i16; count];
        let mut buf = [0u8; 2];
        for slot in data.iter_mut() {
            r.read_exact(&mut buf)?;
            *slot = i16::from_le_bytes(buf);
        }
        Ok(Tablebase {
            pits_per_side,
            cap,
            num_layouts,
            data,
        })
    }
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Build a zero-store board from a canonical layout (mover pits then opponent
/// pits), with `P0` as the mover. By mirror symmetry this one orientation covers
/// both sides.
fn layout_board(pits_per_side: usize, layout: &[u8]) -> Board {
    let n = pits_per_side;
    let mut cells = vec![0u8; 2 * n + 2];
    cells[..n].copy_from_slice(&layout[..n]); // mover (P0) pits
    cells[n + 1..2 * n + 1].copy_from_slice(&layout[n..2 * n]); // opponent (P1) pits
    Board::new(n, cells, crate::board::Player::P0).expect("valid layout board")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Player;
    use crate::endgame::Endgame;

    /// Board with the given mover/opponent pits and `turn` to move (zero stores).
    fn board_for(n: usize, mover: &[u8], opp: &[u8], turn: Player) -> Board {
        let mut cells = vec![0u8; 2 * n + 2];
        let (p0, p1) = match turn {
            Player::P0 => (mover, opp),
            Player::P1 => (opp, mover),
        };
        cells[..n].copy_from_slice(p0);
        cells[n + 1..2 * n + 1].copy_from_slice(p1);
        Board::new(n, cells, turn).unwrap()
    }

    #[test]
    fn rank_is_a_bijection() {
        // For m parts and cap C, ranks of all layouts must be exactly 0..total.
        for (m, cap) in [(2u64, 4u32), (3, 5), (4, 6)] {
            let total = num_layouts(cap as u64, m) as usize;
            let mut seen = vec![false; total];
            let mut pits = vec![0u8; m as usize];
            Tablebase::for_each_layout(cap, &mut pits, &mut |layout| {
                let r = rank(layout) as usize;
                assert!(r < total, "rank {r} out of range {total}");
                assert!(!seen[r], "duplicate rank {r}");
                seen[r] = true;
            });
            assert!(seen.into_iter().all(|s| s), "every index covered");
        }
    }

    #[test]
    fn tablebase_matches_lazy_g() {
        // Built values must equal the lazy endgame `g` for every covered layout,
        // with either player as the mover (exercising mirror canonicalization).
        let rules = Rules::default();
        let tb = Tablebase::build(rules, 2, 6);
        let mut eg = Endgame::new(rules, 6);
        let mut pits = vec![0u8; 4];
        for turn in [Player::P0, Player::P1] {
            Tablebase::for_each_layout(6, &mut pits, &mut |layout| {
                let board = board_for(2, &layout[..2], &layout[2..], turn);
                assert_eq!(tb.lookup(&board), Some(eg.g(&board)));
            });
        }
    }

    #[test]
    fn lookup_is_mirror_symmetric() {
        // The same (mover, opponent) pits must hit the same entry regardless of
        // which player is the mover.
        let tb = Tablebase::build(Rules::default(), 2, 6);
        let mut pits = vec![0u8; 4];
        Tablebase::for_each_layout(6, &mut pits, &mut |layout| {
            let as_p0 = board_for(2, &layout[..2], &layout[2..], Player::P0);
            let as_p1 = board_for(2, &layout[..2], &layout[2..], Player::P1);
            assert_eq!(tb.lookup(&as_p0), tb.lookup(&as_p1));
        });
    }

    #[test]
    fn save_and_load_round_trips() {
        let tb = Tablebase::build(Rules::default(), 2, 5);
        let path = std::env::temp_dir().join("mancala_test_tb.bin");
        tb.save(&path).unwrap();
        let loaded = Tablebase::load(&path).unwrap();
        assert_eq!(loaded.pits_per_side(), 2);
        assert_eq!(loaded.cap(), 5);
        // Spot-check a couple of layouts.
        let b = Board::new(2, vec![1, 2, 0, 1, 1, 0], Player::P0).unwrap();
        assert_eq!(tb.lookup(&b), loaded.lookup(&b));
        std::fs::remove_file(&path).ok();
    }
}
