//! Self-play strength testing.
//!
//! Strength in the *non-exact* regime can't be read off a single position — it
//! only shows up over many games. This module pits two engine configurations
//! ("players") against each other over a **diverse opening book** (each opening
//! played from both sides so colour/first-move bias cancels) and reports the
//! match score together with the **Elo difference, a 95% confidence interval,
//! and the likelihood-of-superiority** — the standard way engine changes are
//! graded (à la Fishtest).
//!
//! Players are configurations of this engine, so games run in-process with no
//! protocol overhead. Each player is a deterministic move-chooser; diversity
//! (and thus distinct games) comes entirely from the randomised opening book,
//! which keeps results reproducible for a given seed.

use std::collections::HashSet;

use crate::board::{Board, Player, Rules};
use crate::solver::{self, Limit};
use crate::tablebase::Tablebase;

/// A move-selecting engine configuration. The `Limit` is the thinking budget —
/// a fixed depth, or a node count (iterative deepening) so two engines can be
/// compared on equal *work* rather than equal depth.
#[derive(Clone, Copy, Debug)]
pub enum Engine {
    /// Plain heuristic search (no endgame oracle, no quiescence) — the baseline.
    Heuristic { limit: Limit },
    /// Tablebase/endgame-aware play (perfect endgame); `quiesce` extends forcing
    /// moves at the horizon; `tt` enables the transposition table (toggle for
    /// measuring its contribution).
    Play { limit: Limit, quiesce: bool, tt: bool },
    /// Try an exact solve within `budget` nodes; else a depth-`depth` heuristic.
    Analyze { budget: u64, depth: u32 },
}

impl Engine {
    /// Choose a move for the side to move, or `None` at a terminal position.
    pub fn choose(&self, b: &Board, rules: Rules, tb: Option<&Tablebase>) -> Option<usize> {
        if b.is_terminal() {
            return None;
        }
        let a = match *self {
            Engine::Heuristic { limit } => solver::play_search(b, rules, tb, limit, false, false, true),
            Engine::Play { limit, quiesce, tt } => solver::play_search(b, rules, tb, limit, true, quiesce, tt),
            Engine::Analyze { budget, depth } => solver::analyze_with_tb(b, rules, budget, depth, tb),
        };
        a.best_move
    }

    /// Parse a compact spec. The limit token is `<n>` or `d<n>` for a fixed
    /// depth, or `n<nodes>` for a node budget (iterative deepening):
    ///   `h:<lim>` heuristic · `p:<lim>` play · `q:<lim>` play+quiescence ·
    ///   `a:<budget>:<depth>` exact-then-heuristic.
    pub fn parse(spec: &str) -> Result<Engine, String> {
        let parts: Vec<&str> = spec.split(':').collect();
        match parts.as_slice() {
            ["h", l] => Ok(Engine::Heuristic { limit: parse_limit(l, spec)? }),
            ["p", l] => Ok(Engine::Play { limit: parse_limit(l, spec)?, quiesce: false, tt: true }),
            ["q", l] => Ok(Engine::Play { limit: parse_limit(l, spec)?, quiesce: true, tt: true }),
            // `q:<lim>:nott` disables the transposition table (for A/B testing it).
            ["q", l, "nott"] => Ok(Engine::Play { limit: parse_limit(l, spec)?, quiesce: true, tt: false }),
            ["a", b, d] => Ok(Engine::Analyze {
                budget: b.parse().map_err(|_| format!("bad budget in '{spec}'"))?,
                depth: d.parse().map_err(|_| format!("bad depth in '{spec}'"))?,
            }),
            _ => Err(format!(
                "bad engine spec '{spec}' (h/p/q:<lim> or a:<budget>:<depth>; <lim> = depth '6', \
                 nodes 'n200000', or time 't50'; 'q:<lim>:nott' disables the TT)"
            )),
        }
    }
}

/// Parse a limit token: `n<nodes>` → node budget; `t<ms>` → time budget;
/// `d<n>` or bare `<n>` → depth.
fn parse_limit(tok: &str, spec: &str) -> Result<Limit, String> {
    if let Some(n) = tok.strip_prefix('n') {
        Ok(Limit::Nodes(n.parse().map_err(|_| format!("bad node budget in '{spec}'"))?))
    } else if let Some(t) = tok.strip_prefix('t') {
        Ok(Limit::Time(t.parse().map_err(|_| format!("bad time budget (ms) in '{spec}'"))?))
    } else {
        let d = tok.strip_prefix('d').unwrap_or(tok);
        Ok(Limit::Depth(d.parse().map_err(|_| format!("bad depth in '{spec}'"))?))
    }
}

/// A tiny SplitMix64 PRNG — deterministic, dependency-free, good enough to
/// shuffle out a varied opening book.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Build `count` distinct, non-terminal opening positions by playing `plies`
/// random legal moves from the standard start. Deterministic for a given seed.
fn opening_book(
    pits: usize,
    seeds: u8,
    rules: Rules,
    plies: u32,
    count: usize,
    seed: u64,
) -> Result<Vec<Board>, String> {
    let mut rng = Rng(seed ^ 0xA5A5_5A5A_1234_5678);
    let mut seen: HashSet<Board> = HashSet::new();
    let mut book = Vec::with_capacity(count);
    let mut tries = 0usize;
    let max_tries = count.saturating_mul(64).max(4096);
    while book.len() < count && tries < max_tries {
        tries += 1;
        let mut b = Board::start(pits, seeds, Player::P0)?;
        for _ in 0..plies {
            if b.is_terminal() {
                break;
            }
            let moves = b.legal_moves(b.turn());
            let mv = moves[rng.below(moves.len())];
            b = b.apply(&rules, mv).board;
        }
        if !b.is_terminal() && seen.insert(b) {
            book.push(b);
        }
    }
    if book.is_empty() {
        return Err("could not generate any non-terminal openings".to_string());
    }
    Ok(book)
}

/// Play one game from `open` with engine `a` controlling P0 iff `a_is_p0`.
/// Returns A's score: 1.0 win, 0.5 draw, 0.0 loss.
fn play_game(open: Board, rules: Rules, a: Engine, b: Engine, a_is_p0: bool, tb: Option<&Tablebase>) -> f64 {
    let mut board = open;
    let mut plies = 0u32;
    while !board.is_terminal() && plies < 100_000 {
        let a_to_move = (board.turn() == Player::P0) == a_is_p0;
        let eng = if a_to_move { a } else { b };
        let mv = match eng.choose(&board, rules, tb) {
            Some(m) => m,
            None => break,
        };
        board = board.apply(&rules, mv).board;
        plies += 1;
    }
    let (s0, s1) = board.final_scores();
    let (a_score, b_score) = if a_is_p0 { (s0, s1) } else { (s1, s0) };
    match a_score.cmp(&b_score) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => 0.0,
        std::cmp::Ordering::Equal => 0.5,
    }
}

/// Aggregate result of a match from engine A's perspective.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatchResult {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl MatchResult {
    pub fn games(&self) -> u32 {
        self.wins + self.draws + self.losses
    }

    /// A's score fraction in `[0, 1]`.
    pub fn score(&self) -> f64 {
        let n = self.games();
        if n == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / n as f64
    }

    fn elo_of(score: f64) -> f64 {
        let s = score.clamp(1e-9, 1.0 - 1e-9);
        -400.0 * (1.0 / s - 1.0).log10()
    }

    /// Elo difference A − B implied by the score.
    pub fn elo(&self) -> f64 {
        Self::elo_of(self.score())
    }

    /// 95% confidence interval `(low, high)` on the Elo difference, from the
    /// per-game score variance (normal approximation).
    pub fn elo_ci(&self) -> (f64, f64) {
        let n = self.games() as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }
        let s = self.score();
        let w = self.wins as f64 / n;
        let d = self.draws as f64 / n;
        // Variance of a single game's score (values in {0, 0.5, 1}).
        let var = (w + 0.25 * d) - s * s;
        let se = (var / n).sqrt();
        (
            Self::elo_of(s - 1.96 * se),
            Self::elo_of(s + 1.96 * se),
        )
    }

    /// Likelihood (0..1) that A is genuinely stronger than B, from the
    /// win/loss counts (drawn games carry no directional information).
    pub fn los(&self) -> f64 {
        let wl = (self.wins + self.losses) as f64;
        if wl == 0.0 {
            return 0.5;
        }
        let x = (self.wins as f64 - self.losses as f64) / (2.0 * wl).sqrt();
        0.5 * (1.0 + erf(x))
    }
}

/// Abramowitz & Stegun 7.1.26 approximation of the error function.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// Settings for a match.
pub struct MatchConfig {
    pub pits: usize,
    pub seeds: u8,
    pub rules: Rules,
    pub openings: usize,
    pub opening_plies: u32,
    pub seed: u64,
    pub threads: usize,
}

/// Run a match between engines `a` and `b`: every opening is played twice (once
/// with each engine moving from each side). `progress(done, total)` is called as
/// games complete. Returns the aggregate from A's perspective.
pub fn run_match(
    a: Engine,
    b: Engine,
    cfg: &MatchConfig,
    tb: Option<&Tablebase>,
    mut progress: impl FnMut(usize, usize),
) -> Result<MatchResult, String> {
    let book = opening_book(cfg.pits, cfg.seeds, cfg.rules, cfg.opening_plies, cfg.openings, cfg.seed)?;
    let total = book.len() * 2;
    let rules = cfg.rules;
    let nthreads = cfg.threads.clamp(1, 64);

    if nthreads == 1 {
        let mut r = MatchResult::default();
        let mut done = 0;
        for open in &book {
            for a_is_p0 in [true, false] {
                tally(&mut r, play_game(*open, rules, a, b, a_is_p0, tb));
                done += 1;
                progress(done, total);
            }
        }
        return Ok(r);
    }

    // Split the book across threads; each game is independent and deterministic,
    // so the summed result is identical regardless of thread count.
    let done = std::sync::atomic::AtomicUsize::new(0);
    let chunk = book.len().div_ceil(nthreads);
    let result = std::thread::scope(|scope| {
        let handles: Vec<_> = book
            .chunks(chunk)
            .map(|slice| {
                let done = &done;
                scope.spawn(move || {
                    let mut r = MatchResult::default();
                    for open in slice {
                        for a_is_p0 in [true, false] {
                            tally(&mut r, play_game(*open, rules, a, b, a_is_p0, tb));
                            done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    r
                })
            })
            .collect();
        let mut total_r = MatchResult::default();
        for h in handles {
            let r = h.join().unwrap();
            total_r.wins += r.wins;
            total_r.draws += r.draws;
            total_r.losses += r.losses;
        }
        total_r
    });
    progress(done.load(std::sync::atomic::Ordering::Relaxed), total);
    Ok(result)
}

fn tally(r: &mut MatchResult, score: f64) {
    if score > 0.75 {
        r.wins += 1;
    } else if score < 0.25 {
        r.losses += 1;
    } else {
        r.draws += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_is_diverse_and_deterministic() {
        let r = Rules::default();
        let a = opening_book(6, 4, r, 4, 50, 7).unwrap();
        let b = opening_book(6, 4, r, 4, 50, 7).unwrap();
        assert_eq!(a, b, "same seed -> same book");
        let uniq: HashSet<_> = a.iter().collect();
        assert_eq!(uniq.len(), a.len(), "book has no duplicates");
        assert!(a.iter().all(|p| !p.is_terminal()));
    }

    #[test]
    fn deeper_heuristic_outscores_shallower() {
        // The harness must register that more search = stronger play: a depth-5
        // heuristic should beat a depth-1 one over a small match.
        let cfg = MatchConfig {
            pits: 6,
            seeds: 4,
            rules: Rules::default(),
            openings: 30,
            opening_plies: 4,
            seed: 1,
            threads: 1,
        };
        let deep = Engine::Heuristic { limit: Limit::Depth(5) };
        let shallow = Engine::Heuristic { limit: Limit::Depth(1) };
        let res = run_match(deep, shallow, &cfg, None, |_, _| {}).unwrap();
        assert!(
            res.score() > 0.5,
            "deeper should score >50% (got {:.1}% W{}-L{}-D{})",
            res.score() * 100.0,
            res.wins,
            res.losses,
            res.draws
        );
    }
}
