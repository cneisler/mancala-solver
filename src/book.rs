//! Opening book: a sparse, store-independent endgame-style table for the
//! *early* game.
//!
//! The [`crate::tablebase`] cuts the forward search off at the **bottom** (few
//! seeds left); an opening book cuts it off at the **top** (the first few plies
//! from the start), so together they sandwich the search. Unlike the endgame —
//! whose ≤cap positions form a small closed set solvable in isolation — an
//! early position's exact value depends on the *whole* tree beneath it, so a
//! *proven* book can't be cheaper to build than solving the game. It is instead
//! a **byproduct** of solving once: we enumerate the layouts reachable within
//! `plies` moves of the opening and solve each (reusing one warm transposition
//! table, so the later solves are mostly cache hits). The payoff is downstream —
//! those positions become O(1) exact lookups, which makes opening / early-game
//! analysis instant (including in the browser, which can't otherwise solve the
//! opening at all).
//!
//! Entries are **mirror-canonical** and **store-independent**, exactly like the
//! tablebase: the key is the pit layout packed mover-first (6 bits per pit), and
//! the value is `g` (the optimal future store differential), so one entry serves
//! both mirrors and every store filling — `margin = (my_store − opp_store) + g`.

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashSet;
use std::io::{self, Read, Write};

use crate::board::{Board, Rules};
#[cfg(not(target_arch = "wasm32"))]
use crate::board::Player;
use crate::hash::U128Map;
use crate::solver::{Analysis, MoveEval};
use crate::tablebase::Tablebase;

const MAGIC: &[u8; 8] = b"KALABK01";

/// Mover-canonical, store-independent key: the mover's pits then the opponent's,
/// 6 bits per pit (no stores, no turn bit — identical scheme to the tablebase).
fn layout_key(b: &Board) -> u128 {
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

/// A zero-store board with `b`'s pit layout and side to move (so its game value
/// equals the store-independent `g` for the layout).
#[cfg(not(target_arch = "wasm32"))]
fn zero_store(b: &Board) -> Board {
    let n = b.pits_per_side();
    let mut cells = vec![0u8; 2 * n + 2];
    for i in 0..n {
        cells[b.pit_global(Player::P0, i)] = b.cells()[b.pit_global(Player::P0, i)];
        cells[b.pit_global(Player::P1, i)] = b.cells()[b.pit_global(Player::P1, i)];
    }
    Board::new(n, cells, b.turn()).expect("valid zero-store layout")
}

/// Best move at a tablebase-covered position, by evaluating each child's
/// `store_diff ± g`. `None` outside the cap or with no tablebase.
fn tb_best_move(b: &Board, rules: Rules, tb: Option<&Tablebase>) -> Option<usize> {
    let tb = tb?;
    if b.pits_per_side() != tb.pits_per_side() || b.seeds_in_play() > tb.cap() {
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
        let r = b.apply(&rules, i);
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

/// A proven opening book for one board size.
pub struct Book {
    pits_per_side: usize,
    /// `layout_key -> (g, best_move)`; best `255` means none recorded.
    map: U128Map<(i16, u8)>,
}

impl Book {
    pub fn pits_per_side(&self) -> usize {
        self.pits_per_side
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Build a book covering every layout reachable within `plies` moves of the
    /// standard `pits`×`seeds` opening, solving each exactly with a shared warm
    /// table. `progress(done, total)` is called as layouts are solved.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn build(
        rules: Rules,
        pits: usize,
        seeds: u8,
        plies: u32,
        tb: &Tablebase,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Book, String> {
        use crate::solver::Solver;

        // Breadth-first enumeration of the distinct (mirror-canonical) pit
        // layouts reachable within `plies` moves; collect a zero-store
        // representative of each to solve.
        let start = Board::start(pits, seeds, Player::P0)?;
        let mut seen: HashSet<u128> = HashSet::new();
        let mut layouts: Vec<Board> = Vec::new();
        let mut current = vec![start];
        for ply in 0..=plies {
            let mut next = Vec::new();
            for b in &current {
                if b.is_terminal() {
                    continue;
                }
                if seen.insert(layout_key(b)) {
                    layouts.push(zero_store(b));
                }
                if ply < plies {
                    for mv in b.legal_moves(b.turn()) {
                        next.push(b.apply(&rules, mv).board);
                    }
                }
            }
            current = next;
            if current.is_empty() {
                break;
            }
        }

        let mut solver = Solver::new(rules, tb);
        let mut map = U128Map::default();
        let total = layouts.len();
        for (i, lb) in layouts.iter().enumerate() {
            let (value, best) = solver.solve(lb);
            map.insert(layout_key(lb), (value as i16, best.map_or(255, |m| m as u8)));
            progress(i + 1, total);
        }
        Ok(Book {
            pits_per_side: pits,
            map,
        })
    }

    /// Look up a position: `(g, best_move)` if its layout is in the book.
    pub fn lookup(&self, b: &Board) -> Option<(i16, Option<usize>)> {
        if b.pits_per_side() != self.pits_per_side {
            return None;
        }
        self.map.get(&layout_key(b)).map(|&(g, best)| {
            (g, if best == 255 { None } else { Some(best as usize) })
        })
    }

    /// `g` for a child position from book or tablebase (child-mover perspective).
    fn child_g(&self, child: &Board, tb: Option<&Tablebase>) -> Option<i32> {
        if let Some((g, _)) = self.lookup(child) {
            return Some(g as i32);
        }
        if let Some(t) = tb {
            if child.pits_per_side() == t.pits_per_side() && child.seeds_in_play() <= t.cap() {
                return t.lookup(child).map(|g| g as i32);
            }
        }
        None
    }

    /// Full exact analysis of `board` from the book, or `None` if it isn't a
    /// book position. Per-move values and the principal variation are recovered
    /// from the book and tablebase (`±store_diff ± g`); moves whose child can't
    /// be resolved are reported as a bound (`≤` the best value).
    pub fn analyze(&self, board: &Board, rules: Rules, tb: Option<&Tablebase>) -> Option<Analysis> {
        let (g, best) = self.lookup(board)?;
        let p = board.turn();
        let o = p.other();
        let sd = board.store(p) as i32 - board.store(o) as i32;
        let value = sd + g as i32;

        let mut evals = Vec::new();
        for mv in 0..board.pits_per_side() {
            if board.cells()[board.pit_global(p, mv)] == 0 {
                continue;
            }
            let r = board.apply(&rules, mv);
            let (v, bound) = match self.child_g(&r.board, tb) {
                Some(cg) => {
                    let csd = r.board.store(p) as i32 - r.board.store(o) as i32;
                    (if r.extra_turn { csd + cg } else { csd - cg }, false)
                }
                None => (value, true), // exact value unknown, but provably ≤ best
            };
            evals.push(MoveEval {
                pit: mv,
                value: v,
                bound,
                extra_turn: r.extra_turn,
                captured: r.captured,
            });
        }
        evals.sort_by(|a, c| a.bound.cmp(&c.bound).then(c.value.cmp(&a.value)));

        let pv = self.principal_variation(board, rules, tb);
        Some(Analysis {
            exact: true,
            side_to_move: p,
            value,
            best_move: best,
            move_evals: evals,
            pv,
            nodes: 0,
            depth: None,
        })
    }

    /// Walk best moves through the book, then the tablebase, to the game end.
    fn principal_variation(&self, board: &Board, rules: Rules, tb: Option<&Tablebase>) -> Vec<usize> {
        let mut pv = Vec::new();
        let mut cur = *board;
        let mut guard = 0;
        while !cur.is_terminal() && guard < 4000 {
            guard += 1;
            let mv = match self
                .lookup(&cur)
                .and_then(|(_, b)| b)
                .or_else(|| tb_best_move(&cur, rules, tb))
            {
                Some(m) => m,
                None => break,
            };
            pv.push(mv);
            cur = cur.apply(&rules, mv).board;
        }
        pv
    }

    /// Parse a book from any reader (see [`Self::save`] for the format).
    pub fn from_reader<R: Read>(mut r: R) -> io::Result<Book> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a Kalah book file"));
        }
        let pits_per_side = read_u32(&mut r)? as usize;
        let count = read_u64(&mut r)? as usize;
        let mut map = U128Map::default();
        let mut kbuf = [0u8; 16];
        let mut gbuf = [0u8; 2];
        let mut bbuf = [0u8; 1];
        for _ in 0..count {
            r.read_exact(&mut kbuf)?;
            r.read_exact(&mut gbuf)?;
            r.read_exact(&mut bbuf)?;
            map.insert(u128::from_le_bytes(kbuf), (i16::from_le_bytes(gbuf), bbuf[0]));
        }
        Ok(Book { pits_per_side, map })
    }

    /// Load a book from an in-memory byte slice (e.g. a `fetch`ed file).
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Book> {
        Self::from_reader(std::io::Cursor::new(bytes))
    }

    /// Serialize to a writer. Format (little-endian): magic `KALABK01`,
    /// `pits_per_side: u32`, `count: u64`, then `count × (key u128, g i16, best u8)`.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&(self.pits_per_side as u32).to_le_bytes())?;
        w.write_all(&(self.map.len() as u64).to_le_bytes())?;
        for (&key, &(g, best)) in &self.map {
            w.write_all(&key.to_le_bytes())?;
            w.write_all(&g.to_le_bytes())?;
            w.write_all(&[best])?;
        }
        Ok(())
    }

    /// Save to a file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        self.write_to(&mut w)?;
        w.flush()
    }

    /// Load from a file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> io::Result<Book> {
        Self::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::solver;

    #[test]
    fn book_matches_direct_solve_and_round_trips() {
        let rules = Rules::default();
        let tb = Tablebase::build(rules, 4, 10);
        let book = Book::build(rules, 4, 3, 2, &tb, |_, _| {}).unwrap();
        assert!(book.len() > 1, "book should cover several layouts");

        // The book's exact opening result must match a direct exact solve.
        let start = Board::start(4, 3, Player::P0).unwrap();
        let direct = solver::analyze_with_tb(&start, rules, u64::MAX, 12, Some(&tb));
        let booked = book.analyze(&start, rules, Some(&tb)).unwrap();
        assert!(booked.exact);
        assert_eq!(booked.value, direct.value, "book value must equal solved value");
        assert_eq!(booked.best_move, direct.best_move);
        assert!(!booked.pv.is_empty());

        // A position one move in is also a book hit with the matching value.
        let next = start.apply(&rules, direct.best_move.unwrap()).board;
        if let Some(b) = book.analyze(&next, rules, Some(&tb)) {
            let d = solver::analyze_with_tb(&next, rules, u64::MAX, 12, Some(&tb));
            assert_eq!(b.value, d.value);
        }

        // Serialization round-trips.
        let mut buf = Vec::new();
        book.write_to(&mut buf).unwrap();
        let loaded = Book::from_bytes(&buf).unwrap();
        assert_eq!(loaded.len(), book.len());
        assert_eq!(loaded.lookup(&start), book.lookup(&start));
    }
}
