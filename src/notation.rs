//! Parsing and formatting of compact board-notation strings.
//!
//! A position is written as two comma-separated groups separated by `|`:
//!
//! ```text
//! p0_0,p0_1,...,p0_{N-1},p0_store | p1_0,p1_1,...,p1_{N-1},p1_store
//! ```
//!
//! The left group is player 0's pits followed by player 0's store; the right
//! group is player 1's pits followed by player 1's store. Both groups must have
//! the same length, and `N` (pits per side) is inferred as `len - 1`.
//!
//! Example (Kalah(6,4) start):
//! `4,4,4,4,4,4,0 | 4,4,4,4,4,4,0`

use crate::board::{Board, Player};

/// Parse a board-notation string into a [`Board`] with the given side to move.
pub fn parse(s: &str, turn: Player) -> Result<Board, String> {
    let sides: Vec<&str> = s.split('|').collect();
    if sides.len() != 2 {
        return Err(format!(
            "expected exactly one '|' separating the two sides, found {}",
            sides.len().saturating_sub(1)
        ));
    }

    let p0 = parse_side(sides[0])?;
    let p1 = parse_side(sides[1])?;

    if p0.len() != p1.len() {
        return Err(format!(
            "both sides must list the same number of values (got {} and {})",
            p0.len(),
            p1.len()
        ));
    }
    if p0.len() < 2 {
        return Err("each side must list at least one pit plus a store".to_string());
    }

    let n = p0.len() - 1; // last value of each group is that side's store
    let mut cells = vec![0u8; 2 * n + 2];
    cells[..n].copy_from_slice(&p0[..n]); // P0 pits
    cells[n] = p0[n]; // P0 store
    cells[n + 1..2 * n + 1].copy_from_slice(&p1[..n]); // P1 pits
    cells[2 * n + 1] = p1[n]; // P1 store

    Board::new(n, cells, turn)
}

fn parse_side(s: &str) -> Result<Vec<u8>, String> {
    s.split(',')
        .map(|tok| tok.trim())
        .filter(|tok| !tok.is_empty())
        .map(|tok| {
            tok.parse::<u8>()
                .map_err(|_| format!("'{tok}' is not a valid seed count (0..=255)"))
        })
        .collect()
}

/// Format a [`Board`] back into notation (round-trips with [`parse`]).
pub fn format(b: &Board) -> String {
    let n = b.pits_per_side();
    let cells = b.cells();
    let p0: Vec<String> = (0..=n).map(|i| cells[i].to_string()).collect();
    let p1: Vec<String> = (n + 1..=2 * n + 1).map(|i| cells[i].to_string()).collect();
    format!("{} | {}", p0.join(","), p1.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_start_position() {
        let b = parse("4,4,4,4,4,4,0 | 4,4,4,4,4,4,0", Player::P0).unwrap();
        assert_eq!(b.pits_per_side(), 6);
        assert_eq!(b.store(Player::P0), 0);
        assert_eq!(b.store(Player::P1), 0);
        assert_eq!(b.cells().iter().map(|&c| c as u32).sum::<u32>(), 48);
    }

    #[test]
    fn round_trips() {
        let s = "1,2,3,0 | 4,5,6,7";
        let b = parse(s, Player::P1).unwrap();
        assert_eq!(format(&b), s);
    }

    #[test]
    fn rejects_mismatched_sides() {
        assert!(parse("1,2,0 | 3,4,5,0", Player::P0).is_err());
    }

    #[test]
    fn rejects_missing_separator() {
        assert!(parse("1,2,3,0", Player::P0).is_err());
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse("1,x,0 | 1,2,0", Player::P0).is_err());
    }
}
