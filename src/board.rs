//! Board representation and the rules of Kalah-style Mancala.
//!
//! # Layout
//!
//! The board is stored as a flat `Vec<u8>` of `2 * pits_per_side + 2` cells:
//!
//! ```text
//! index:  0 .. N-1      N        N+1 .. 2N     2N+1
//!        [ P0 pits ] [P0 store] [ P1 pits ]  [P1 store]
//! ```
//!
//! Seeds are sown counterclockwise (increasing index, wrapping around),
//! skipping the *opponent's* store. We use `u8` per cell, which is sufficient
//! as long as the total number of seeds fits in a `u8`; [`Board::new`] and the
//! notation parser reject boards that would overflow.

use std::fmt;

/// Maximum number of board cells (`2 * pits_per_side + 2`). Boards are stored in
/// a fixed-size stack array so positions are cheap to copy during search (no
/// heap allocation per node). This caps the board at [`MAX_PITS`] pits per side.
pub const MAX_CELLS: usize = 32;

/// Maximum pits per side, derived from [`MAX_CELLS`].
pub const MAX_PITS: usize = (MAX_CELLS - 2) / 2;

/// The two players. `P0` is conventionally rendered as the South (bottom) side,
/// `P1` as the North (top) side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Player {
    P0,
    P1,
}

impl Player {
    /// The opposing player.
    pub fn other(self) -> Player {
        match self {
            Player::P0 => Player::P1,
            Player::P1 => Player::P0,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::P0 => write!(f, "P0 (South)"),
            Player::P1 => write!(f, "P1 (North)"),
        }
    }
}

/// Tunable rule options for the Kalah variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// If `true`, a capture only happens when the *opposite* pit is non-empty
    /// (the variant requested for this project). If `false`, landing the last
    /// seed in one of your own previously-empty pits always captures (you take
    /// at least your own seed).
    pub capture_requires_nonempty_opposite: bool,
}

impl Default for Rules {
    fn default() -> Self {
        Rules {
            capture_requires_nonempty_opposite: true,
        }
    }
}

/// The result of applying a move.
#[derive(Clone, Debug)]
pub struct MoveResult {
    /// The resulting board, with `turn` already updated.
    pub board: Board,
    /// Whether the move granted an extra turn (last seed landed in own store),
    /// in which case `board.turn` is unchanged from the mover.
    pub extra_turn: bool,
    /// Whether the move performed a capture.
    pub captured: bool,
}

/// A Mancala position: the seed counts, the board size, and the side to move.
///
/// `cells` is a fixed-size array; only the first `2 * pits_per_side + 2` entries
/// are meaningful and the remainder are kept zero, so the type is `Copy` and
/// can be cloned during search without allocating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Board {
    pits_per_side: usize,
    cells: [u8; MAX_CELLS],
    turn: Player,
}

impl Board {
    /// Build a board from explicit cell counts.
    ///
    /// `cells` must have length `2 * pits_per_side + 2`, `pits_per_side` must be
    /// in `1..=MAX_PITS`, and the total number of seeds must fit in a `u8`
    /// (≤ 255). Returns an error otherwise.
    pub fn new(pits_per_side: usize, cells: Vec<u8>, turn: Player) -> Result<Board, String> {
        if pits_per_side == 0 {
            return Err("pits_per_side must be at least 1".to_string());
        }
        if pits_per_side > MAX_PITS {
            return Err(format!(
                "pits_per_side {pits_per_side} exceeds the supported maximum of {MAX_PITS}"
            ));
        }
        let expected = 2 * pits_per_side + 2;
        if cells.len() != expected {
            return Err(format!(
                "expected {expected} cells for a board with {pits_per_side} pits per side, got {}",
                cells.len()
            ));
        }
        let total: u32 = cells.iter().map(|&c| c as u32).sum();
        if total > u8::MAX as u32 {
            return Err(format!(
                "total seed count {total} exceeds the supported maximum of {}",
                u8::MAX
            ));
        }
        let mut buf = [0u8; MAX_CELLS];
        buf[..expected].copy_from_slice(&cells);
        Ok(Board {
            pits_per_side,
            cells: buf,
            turn,
        })
    }

    /// The standard opening position: every pit holds `seeds_per_pit` seeds,
    /// both stores empty.
    pub fn start(pits_per_side: usize, seeds_per_pit: u8, turn: Player) -> Result<Board, String> {
        let mut cells = vec![seeds_per_pit; 2 * pits_per_side + 2];
        cells[pits_per_side] = 0; // P0 store
        cells[2 * pits_per_side + 1] = 0; // P1 store
        Board::new(pits_per_side, cells, turn)
    }

    /// Number of pits on each side.
    pub fn pits_per_side(&self) -> usize {
        self.pits_per_side
    }

    /// The side to move.
    pub fn turn(&self) -> Player {
        self.turn
    }

    /// Raw cell counts (see the module docs for the layout).
    pub fn cells(&self) -> &[u8] {
        &self.cells[..self.total_cells()]
    }

    /// Total number of cells, `2 * pits_per_side + 2`.
    pub fn total_cells(&self) -> usize {
        2 * self.pits_per_side + 2
    }

    /// Global cell index of player `p`'s store.
    pub fn store_index(&self, p: Player) -> usize {
        match p {
            Player::P0 => self.pits_per_side,
            Player::P1 => 2 * self.pits_per_side + 1,
        }
    }

    /// Global cell index of player `p`'s pit number `i` (0-based within the row).
    pub fn pit_global(&self, p: Player, i: usize) -> usize {
        match p {
            Player::P0 => i,
            Player::P1 => self.pits_per_side + 1 + i,
        }
    }

    /// Seeds currently in player `p`'s store.
    pub fn store(&self, p: Player) -> u8 {
        self.cells[self.store_index(p)]
    }

    /// Whether global index `g` is one of player `p`'s playing pits.
    fn is_own_pit(&self, p: Player, g: usize) -> bool {
        match p {
            Player::P0 => g < self.pits_per_side,
            Player::P1 => g > self.pits_per_side && g < 2 * self.pits_per_side + 1,
        }
    }

    /// The pit directly opposite global pit index `g` (capture target).
    fn opposite(&self, g: usize) -> usize {
        2 * self.pits_per_side - g
    }

    /// Whether player `p`'s entire row of pits is empty.
    pub fn side_empty(&self, p: Player) -> bool {
        (0..self.pits_per_side).all(|i| self.cells[self.pit_global(p, i)] == 0)
    }

    /// A position is terminal when either side has no seeds in its pits.
    pub fn is_terminal(&self) -> bool {
        self.side_empty(Player::P0) || self.side_empty(Player::P1)
    }

    /// Legal moves for player `p`: indices `0..pits_per_side` of non-empty pits.
    pub fn legal_moves(&self, p: Player) -> Vec<usize> {
        (0..self.pits_per_side)
            .filter(|&i| self.cells[self.pit_global(p, i)] > 0)
            .collect()
    }

    /// Global index where the last seed of move `mv` (for the side to move)
    /// would land, accounting for the skipped opponent store. Does not mutate
    /// or clone the board — used for cheap move ordering.
    pub fn sow_landing(&self, mv: usize) -> usize {
        let p = self.turn;
        let start = self.pit_global(p, mv);
        let total = self.total_cells();
        let opp_store = self.store_index(p.other());
        let mut remaining = self.cells[start];
        let mut g = start;
        while remaining > 0 {
            g = (g + 1) % total;
            if g == opp_store {
                continue;
            }
            remaining -= 1;
        }
        g
    }

    /// Whether move `mv` (for the side to move) would grant an extra turn.
    pub fn grants_extra_turn(&self, mv: usize) -> bool {
        self.sow_landing(mv) == self.store_index(self.turn)
    }

    /// Apply `mv` (a 0-based pit index for the side to move) and return the
    /// resulting position. Panics if `mv` is out of range or names an empty pit.
    pub fn apply(&self, rules: &Rules, mv: usize) -> MoveResult {
        let p = self.turn;
        assert!(mv < self.pits_per_side, "move {mv} out of range");
        let start = self.pit_global(p, mv);
        assert!(self.cells[start] > 0, "cannot play from an empty pit");

        let mut b = *self;
        let total = b.total_cells();
        let own_store = b.store_index(p);
        let opp_store = b.store_index(p.other());

        let mut seeds = b.cells[start];
        b.cells[start] = 0;

        let mut g = start;
        while seeds > 0 {
            g = (g + 1) % total;
            if g == opp_store {
                continue; // never sow into the opponent's store
            }
            b.cells[g] += 1;
            seeds -= 1;
        }

        let mut extra_turn = false;
        let mut captured = false;
        if g == own_store {
            extra_turn = true;
        } else if b.is_own_pit(p, g) && b.cells[g] == 1 {
            // The last seed landed in one of our pits that was empty before this
            // seed dropped (it now holds exactly 1) — a potential capture.
            let opp = b.opposite(g);
            let do_capture = if rules.capture_requires_nonempty_opposite {
                b.cells[opp] > 0
            } else {
                true
            };
            if do_capture {
                let taken = b.cells[g] + b.cells[opp];
                b.cells[g] = 0;
                b.cells[opp] = 0;
                b.cells[own_store] += taken;
                captured = true;
            }
        }

        if !extra_turn {
            b.turn = p.other();
        }

        MoveResult {
            board: b,
            extra_turn,
            captured,
        }
    }

    /// Final scores `(p0, p1)` for a terminal position: each side's remaining
    /// pit seeds are swept into its own store. Safe to call on any position
    /// (only the side with seeds contributes), but only meaningful at terminal.
    pub fn final_scores(&self) -> (u32, u32) {
        // No mutation needed; sum each side's pits plus its store directly.
        let mut score = [0u32; 2];
        for (idx, p) in [Player::P0, Player::P1].into_iter().enumerate() {
            let mut total = self.cells[self.store_index(p)] as u32;
            for i in 0..self.pits_per_side {
                total += self.cells[self.pit_global(p, i)] as u32;
            }
            score[idx] = total;
        }
        (score[0], score[1])
    }

    /// Final score margin from `p`'s perspective (positive = `p` wins by that
    /// many seeds). Meaningful at terminal positions.
    pub fn terminal_margin(&self, p: Player) -> i32 {
        let (s0, s1) = self.final_scores();
        let (s0, s1) = (s0 as i32, s1 as i32);
        match p {
            Player::P0 => s0 - s1,
            Player::P1 => s1 - s0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> Rules {
        Rules::default()
    }

    #[test]
    fn start_position_has_no_seeds_in_stores() {
        let b = Board::start(6, 4, Player::P0).unwrap();
        assert_eq!(b.store(Player::P0), 0);
        assert_eq!(b.store(Player::P1), 0);
        assert_eq!(b.cells().iter().map(|&c| c as u32).sum::<u32>(), 48);
    }

    #[test]
    fn landing_in_store_grants_extra_turn() {
        // Kalah(6,4): playing pit index 2 sows into pits 3,4,5 and the store.
        let b = Board::start(6, 4, Player::P0).unwrap();
        let r = b.apply(&rules(), 2);
        assert!(r.extra_turn);
        assert_eq!(r.board.turn(), Player::P0); // same player moves again
        assert_eq!(r.board.store(Player::P0), 1);
    }

    #[test]
    fn non_store_move_passes_turn() {
        let b = Board::start(6, 4, Player::P0).unwrap();
        let r = b.apply(&rules(), 0); // 4 seeds -> pits 1,2,3,4
        assert!(!r.extra_turn);
        assert_eq!(r.board.turn(), Player::P1);
    }

    #[test]
    fn skips_opponent_store() {
        // A long sow from P0 must skip P1's store (index 13 on a 6-board).
        // P0 pit 0 is pre-filled so the wrap-around does not land in an empty
        // pit (which would trigger a capture and muddy this test).
        let mut cells = vec![0u8; 14];
        cells[0] = 5; // wrap lands here, making it 6 (no capture)
        cells[5] = 8; // P0 pit 5: sows past its store, around, to pit 0
        let b = Board::new(6, cells, Player::P0).unwrap();
        let r = b.apply(&rules(), 5);
        // P1's store (index 13) must remain empty.
        assert_eq!(r.board.cells()[13], 0);
        // P0's own store (index 6) should have received exactly one seed.
        assert_eq!(r.board.store(Player::P0), 1);
        assert!(!r.captured);
        // Seeds are conserved.
        assert_eq!(r.board.cells().iter().map(|&c| c as u32).sum::<u32>(), 13);
    }

    #[test]
    fn capture_takes_opposite_pit() {
        // Set up: P0 pit 0 has 1 seed, lands in empty pit 1; opposite of pit 1
        // (global 1) is global 2*6 - 1 = 11, which we stock with seeds.
        let mut cells = vec![0u8; 14];
        cells[0] = 1; // play this
        cells[1] = 0; // becomes 1 -> empty before
        cells[11] = 5; // opposite pit, non-empty
        let b = Board::new(6, cells, Player::P0).unwrap();
        let r = b.apply(&rules(), 0);
        assert!(r.captured);
        assert_eq!(r.board.store(Player::P0), 6); // 1 + 5 captured
        assert_eq!(r.board.cells()[1], 0);
        assert_eq!(r.board.cells()[11], 0);
    }

    #[test]
    fn no_capture_when_opposite_empty_under_default_rules() {
        let mut cells = vec![0u8; 14];
        cells[0] = 1;
        cells[11] = 0; // opposite empty
        let b = Board::new(6, cells, Player::P0).unwrap();
        let r = b.apply(&rules(), 0);
        assert!(!r.captured);
        assert_eq!(r.board.cells()[1], 1); // seed stays put
        assert_eq!(r.board.store(Player::P0), 0);
    }

    #[test]
    fn capture_when_opposite_empty_if_rule_disabled() {
        let mut cells = vec![0u8; 14];
        cells[0] = 1;
        cells[11] = 0;
        let b = Board::new(6, cells, Player::P0).unwrap();
        let lenient = Rules {
            capture_requires_nonempty_opposite: false,
        };
        let r = b.apply(&lenient, 0);
        assert!(r.captured);
        assert_eq!(r.board.store(Player::P0), 1); // captured own single seed
    }

    #[test]
    fn terminal_sweeps_remaining_seeds() {
        // P0 side empty -> terminal; P1's seeds sweep to P1 store.
        let mut cells = vec![0u8; 14];
        cells[6] = 10; // P0 store
        cells[7] = 3; // P1 pit
        cells[8] = 2; // P1 pit
        cells[13] = 5; // P1 store
        let b = Board::new(6, cells, Player::P1).unwrap();
        assert!(b.is_terminal());
        assert_eq!(b.final_scores(), (10, 10));
        assert_eq!(b.terminal_margin(Player::P0), 0);
    }
}
