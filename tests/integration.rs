//! End-to-end tests exercising the public library API the CLI is built on:
//! parse a position from notation, analyze it, and check the engine's verdict.

use mancala::board::{Player, Rules};
use mancala::notation;
use mancala::solver::{analyze, Outcome};

/// Analyze notation with default rules and an unbounded exact budget.
fn analyze_notation(s: &str, turn: Player) {
    let board = notation::parse(s, turn).unwrap();
    let a = analyze(&board, Rules::default(), u64::MAX, 12);
    assert!(a.exact);
    assert!(a.best_move.is_some() || board.is_terminal());
}

#[test]
fn parse_and_analyze_small_position() {
    analyze_notation("3,3,3,0 | 3,3,3,0", Player::P0);
}

#[test]
fn symmetric_position_other_side_to_move_is_mirror() {
    // A symmetric board should give the same exact margin regardless of which
    // (equivalent) side is to move.
    let s = "3,3,3,0 | 3,3,3,0";
    let a0 = analyze(&notation::parse(s, Player::P0).unwrap(), Rules::default(), u64::MAX, 12);
    let a1 = analyze(&notation::parse(s, Player::P1).unwrap(), Rules::default(), u64::MAX, 12);
    assert!(a0.exact && a1.exact);
    // Same value from the mover's perspective by symmetry.
    assert_eq!(a0.value, a1.value);
}

#[test]
fn terminal_position_reports_no_move() {
    // P0 side empty -> terminal; equal stores -> draw.
    let board = notation::parse("0,0,0,9 | 1,2,3,3", Player::P0).unwrap();
    assert!(board.is_terminal());
    let a = analyze(&board, Rules::default(), u64::MAX, 12);
    assert_eq!(a.best_move, None);
    // P0 store 9 vs P1 store 3 + pits 1+2+3 = 9 -> draw.
    assert_eq!(a.outcome(), Outcome::Draw);
}

#[test]
fn kalah_4_3_exact_value() {
    // A non-trivial board (~0.5M nodes, sub-second) with a verified exact value:
    // a first-player win by 6 seeds, best opening pit 1 (the extra-turn move).
    use mancala::board::Board;
    let b = Board::start(4, 3, Player::P0).unwrap();
    let a = analyze(&b, Rules::default(), u64::MAX, 12);
    assert!(a.exact);
    assert_eq!(a.value, 6);
    assert_eq!(a.best_move, Some(1));
    assert_eq!(a.outcome(), Outcome::Win);
}

#[test]
fn heuristic_fallback_still_produces_a_move() {
    // A larger board with a tiny budget must fall back to the heuristic search.
    let board = notation::parse("5,5,5,5,5,5,5,5,0 | 5,5,5,5,5,5,5,5,0", Player::P0).unwrap();
    let a = analyze(&board, Rules::default(), 5_000, 8);
    assert!(!a.exact);
    assert_eq!(a.depth, Some(8));
    assert!(a.best_move.is_some());
    assert_eq!(a.move_evals.len(), 8);
}
