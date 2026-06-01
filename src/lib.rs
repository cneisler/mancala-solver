//! `mancala` — a Kalah-style Mancala engine and position solver.
//!
//! The crate is split into a UI-independent core so that a future GUI/web
//! front-end can reuse the same engine the CLI uses today.
//!
//! * [`board`] — board representation, the rules of play, move generation and
//!   move application.
//! * [`solver`] — an exact alpha-beta solver with a node budget that falls back
//!   to a depth-limited heuristic search, plus position analysis.
//! * [`notation`] — parsing/formatting of the compact board-notation strings
//!   accepted on the command line.

pub mod board;
pub mod notation;
pub mod solver;

pub use board::{Board, MoveResult, Player, Rules};
pub use solver::{analyze, Analysis, MoveEval};
