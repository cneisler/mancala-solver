//! `mancala-solver` — command-line front-end for the `mancala` engine.
//!
//! Subcommands:
//!   analyze --board "<notation>" [options]   Analyze a position from notation.
//!   start   --pits N --seeds M    [options]  Analyze the standard opening.
//!
//! See `--help` for the full option list.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mancala::board::{Board, Player, Rules};
use mancala::notation;
use mancala::solver::{analyze_with_tb, Analysis, MoveEval, Outcome};
use mancala::tablebase::Tablebase;

const DEFAULT_BUDGET: u64 = 20_000_000;
const DEFAULT_FALLBACK_DEPTH: u32 = 11;

/// Auto-tablebase is used by default for boards up to this many pits per side
/// (bigger boards have an explosive endgame space).
const AUTO_TB_MAX_PITS: usize = 6;
/// ...and only when at least this many seeds are in play (small/easy positions
/// solve instantly without one, so building a table would just be overhead).
const AUTO_TB_MIN_SEEDS: u32 = 34;
/// Default endgame seed cap for an auto-built tablebase.
const AUTO_TB_CAP: u32 = 14;
/// Refuse to auto-build a table larger than this many entries (~700 MB).
const AUTO_TB_MAX_ENTRIES: u64 = 350_000_000;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            eprintln!("\nRun `mancala-solver --help` for usage.");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let first = args.first().map(String::as_str);
    match first {
        None | Some("-h") | Some("--help") | Some("help") => {
            print_help();
            Ok(())
        }
        Some("analyze") => cmd_analyze(&args[1..]),
        Some("start") => cmd_start(&args[1..]),
        Some("gen-tb") => cmd_gen_tb(&args[1..]),
        Some(other) => Err(format!("unknown command '{other}'")),
    }
}

/// Options shared by the subcommands.
struct CommonOpts {
    turn: Player,
    budget: u64,
    fallback_depth: u32,
    rules: Rules,
    /// Explicit tablebase file (overrides auto).
    tb_path: Option<String>,
    /// Disable the automatic tablebase.
    no_tb: bool,
    /// Override the auto-tablebase seed cap.
    seeds_cap: Option<u32>,
}

impl Default for CommonOpts {
    fn default() -> Self {
        CommonOpts {
            turn: Player::P0,
            budget: DEFAULT_BUDGET,
            fallback_depth: DEFAULT_FALLBACK_DEPTH,
            rules: Rules::default(),
            tb_path: None,
            no_tb: false,
            seeds_cap: None,
        }
    }
}

/// Parse a `--flag value` style option, consuming the value. Returns the value
/// or an error if it is missing.
fn take_value<'a>(flag: &str, iter: &mut std::slice::Iter<'a, String>) -> Result<&'a str, String> {
    iter.next()
        .map(String::as_str)
        .ok_or_else(|| format!("flag '{flag}' requires a value"))
}

fn parse_common(flag: &str, iter: &mut std::slice::Iter<'_, String>, opts: &mut CommonOpts) -> Result<bool, String> {
    match flag {
        "--turn" => {
            opts.turn = parse_turn(take_value(flag, iter)?)?;
            Ok(true)
        }
        "--budget" => {
            opts.budget = take_value(flag, iter)?
                .parse()
                .map_err(|_| "--budget must be a non-negative integer".to_string())?;
            Ok(true)
        }
        "--depth" => {
            opts.fallback_depth = take_value(flag, iter)?
                .parse()
                .map_err(|_| "--depth must be a non-negative integer".to_string())?;
            Ok(true)
        }
        // Allow captures even when the opposite pit is empty.
        "--capture-empty" => {
            opts.rules.capture_requires_nonempty_opposite = false;
            Ok(true)
        }
        "--tb" => {
            opts.tb_path = Some(take_value(flag, iter)?.to_string());
            Ok(true)
        }
        "--no-tb" => {
            opts.no_tb = true;
            Ok(true)
        }
        "--seeds-cap" => {
            opts.seeds_cap = Some(
                take_value(flag, iter)?
                    .parse()
                    .map_err(|_| "--seeds-cap must be a non-negative integer".to_string())?,
            );
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_turn(s: &str) -> Result<Player, String> {
    match s.to_ascii_lowercase().as_str() {
        "0" | "p0" | "south" | "s" => Ok(Player::P0),
        "1" | "p1" | "north" | "n" => Ok(Player::P1),
        other => Err(format!("invalid --turn '{other}' (use 0/p0/south or 1/p1/north)")),
    }
}

fn cmd_analyze(args: &[String]) -> Result<(), String> {
    let mut opts = CommonOpts::default();
    let mut board_str: Option<String> = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if parse_common(arg, &mut iter, &mut opts)? {
            continue;
        }
        match arg.as_str() {
            "--board" | "-b" => board_str = Some(take_value(arg, &mut iter)?.to_string()),
            other => return Err(format!("unknown flag '{other}' for 'analyze'")),
        }
    }

    let board_str = board_str.ok_or("analyze requires --board \"<notation>\"")?;
    let board = notation::parse(&board_str, opts.turn)?;
    report(&board, &opts)
}

fn cmd_start(args: &[String]) -> Result<(), String> {
    let mut opts = CommonOpts::default();
    let mut pits: usize = 6;
    let mut seeds: u8 = 4;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if parse_common(arg, &mut iter, &mut opts)? {
            continue;
        }
        match arg.as_str() {
            "--pits" => {
                pits = take_value(arg, &mut iter)?
                    .parse()
                    .map_err(|_| "--pits must be a positive integer".to_string())?
            }
            "--seeds" => {
                seeds = take_value(arg, &mut iter)?
                    .parse()
                    .map_err(|_| "--seeds must be 0..=255".to_string())?
            }
            other => return Err(format!("unknown flag '{other}' for 'start'")),
        }
    }

    let board = Board::start(pits, seeds, opts.turn)?;
    report(&board, &opts)
}

fn report(board: &Board, opts: &CommonOpts) -> Result<(), String> {
    println!("{}", render_board(board));
    println!("Notation:    {}", notation::format(board));
    println!("Side to move: {}", board.turn());
    println!("Rules:        capture requires non-empty opposite pit = {}", opts.rules.capture_requires_nonempty_opposite);

    // Resolve the endgame tablebase: explicit `--tb`, the automatic cached one,
    // or none.
    let tb = resolve_tablebase(board, opts)?;
    println!();

    let analysis = analyze_with_tb(board, opts.rules, opts.budget, opts.fallback_depth, tb.as_ref());
    println!("{}", render_analysis(board, &analysis));
    Ok(())
}

/// Decide which endgame tablebase to use: an explicit `--tb`, the automatic
/// cached one (built on first use), or none.
fn resolve_tablebase(board: &Board, opts: &CommonOpts) -> Result<Option<Tablebase>, String> {
    let n = board.pits_per_side();

    // Explicit tablebase file.
    if let Some(path) = &opts.tb_path {
        let tb = Tablebase::load(Path::new(path))
            .map_err(|e| format!("failed to load tablebase '{path}': {e}"))?;
        if tb.pits_per_side() != n {
            return Err(format!(
                "tablebase is for {}-pit boards, but this board has {n} pits per side",
                tb.pits_per_side()
            ));
        }
        println!("Tablebase:    {path} (cap {} seeds in play)", tb.cap());
        return Ok(Some(tb));
    }

    // Automatic tablebase, unless disabled or the board is out of range.
    if opts.no_tb || n > AUTO_TB_MAX_PITS || board.seeds_in_play() < AUTO_TB_MIN_SEEDS {
        return Ok(None);
    }
    let cap = opts.seeds_cap.unwrap_or(AUTO_TB_CAP);
    if Tablebase::entry_count(n, cap) > AUTO_TB_MAX_ENTRIES {
        eprintln!(
            "note: auto-tablebase for {n} pits / cap {cap} would be too large; skipping \
             (use --seeds-cap to lower, or --no-tb to silence)"
        );
        return Ok(None);
    }

    let path = auto_tb_path(n, cap, &opts.rules);
    if path.exists() {
        if let Ok(tb) = Tablebase::load(&path) {
            if tb.pits_per_side() == n && tb.cap() == cap {
                println!("Tablebase:    {} (cached, cap {})", path.display(), tb.cap());
                return Ok(Some(tb));
            }
        }
        // Stale or unreadable cache — fall through and rebuild.
    }

    println!(
        "Tablebase:    building {} (cap {cap}, {} entries; one-time) ...",
        path.display(),
        Tablebase::entry_count(n, cap)
    );
    let tb = Tablebase::build(opts.rules, n, cap);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match tb.save(&path) {
        Ok(()) => println!("              cached to {}", path.display()),
        Err(e) => eprintln!("note: built tablebase but could not cache it: {e}"),
    }
    Ok(Some(tb))
}

/// Directory for cached auto-tablebases (`$MANCALA_TB_DIR`, default `.mancala_tb`).
fn auto_tb_dir() -> PathBuf {
    std::env::var_os("MANCALA_TB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".mancala_tb"))
}

/// Cache filename for a tablebase, keyed by board size, cap, and capture rule.
fn auto_tb_path(n: usize, cap: u32, rules: &Rules) -> PathBuf {
    let capture = if rules.capture_requires_nonempty_opposite {
        "capnonempty"
    } else {
        "capany"
    };
    auto_tb_dir().join(format!("kalah_p{n}_cap{cap}_{capture}.tb"))
}

fn cmd_gen_tb(args: &[String]) -> Result<(), String> {
    let mut pits: usize = 6;
    let mut cap: u32 = 12;
    let mut out: Option<String> = None;
    let mut rules = Rules::default();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--pits" => {
                pits = take_value(arg, &mut iter)?
                    .parse()
                    .map_err(|_| "--pits must be a positive integer".to_string())?
            }
            "--seeds-cap" => {
                cap = take_value(arg, &mut iter)?
                    .parse()
                    .map_err(|_| "--seeds-cap must be a non-negative integer".to_string())?
            }
            "--out" | "-o" => out = Some(take_value(arg, &mut iter)?.to_string()),
            "--capture-empty" => rules.capture_requires_nonempty_opposite = false,
            other => return Err(format!("unknown flag '{other}' for 'gen-tb'")),
        }
    }

    let out = out.ok_or("gen-tb requires --out <file>")?;
    if pits == 0 {
        return Err("--pits must be at least 1".to_string());
    }

    // Guard against accidentally enormous tables.
    let entries = Tablebase::entry_count(pits, cap);
    let bytes = entries * 2;
    const MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB
    if bytes > MAX_BYTES {
        return Err(format!(
            "tablebase for pits={pits} seeds-cap={cap} would be {} entries (~{:.1} GiB) — too large; lower --seeds-cap",
            entries,
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }

    println!(
        "Building tablebase: pits={pits} seeds-cap={cap} ({} entries, ~{:.1} MiB)...",
        entries,
        bytes as f64 / (1024.0 * 1024.0)
    );
    let tb = Tablebase::build(rules, pits, cap);
    tb.save(Path::new(&out))
        .map_err(|e| format!("failed to write '{out}': {e}"))?;
    println!("Wrote {out}");
    Ok(())
}

/// Render the board as a classic two-row Kalah diagram.
///
/// P1 (North) pits run right-to-left across the top; P0 (South) pits run
/// left-to-right across the bottom. P1's store sits on the left, P0's on the
/// right — i.e. each player's store is "ahead" of their own row.
fn render_board(b: &Board) -> String {
    let n = b.pits_per_side();
    let cells = b.cells();
    let w = 3; // per-pit column width

    let p1_top: Vec<String> = (0..n).rev().map(|i| format!("{:>w$}", cells[b.pit_global(Player::P1, i)])).collect();
    let p0_bottom: Vec<String> = (0..n).map(|i| format!("{:>w$}", cells[b.pit_global(Player::P0, i)])).collect();

    let inner_width = n * w + (n - 1); // pits + single-space separators
    let top = format!("       {}", p1_top.join(" "));
    let bottom = format!("       {}", p0_bottom.join(" "));
    let mid = format!(
        "  {:>2} {} {:>2}",
        b.store(Player::P1),
        "-".repeat(inner_width),
        b.store(Player::P0)
    );

    let mut out = String::new();
    out.push_str("        N (P1)\n");
    out.push_str(&top);
    out.push('\n');
    out.push_str(&mid);
    out.push_str("   (P1 store | P0 store)\n");
    out.push_str(&bottom);
    out.push('\n');
    out.push_str("        S (P0)");
    out
}

fn render_analysis(board: &Board, a: &Analysis) -> String {
    let mut out = String::new();

    let mode = if a.exact {
        "EXACT (solved to game end)".to_string()
    } else {
        format!("HEURISTIC (depth {} plies — not provably optimal)", a.depth.unwrap_or(0))
    };
    out.push_str(&format!("Analysis mode: {mode}\n"));
    out.push_str(&format!("Nodes searched: {}\n\n", a.nodes));

    if a.best_move.is_none() {
        let (s0, s1) = board.final_scores();
        out.push_str("Position is TERMINAL — no moves available.\n");
        out.push_str(&format!("Final score: P0 {s0} — P1 {s1}\n"));
        out.push_str(&format!("Result for {}: {}\n", board.turn(), outcome_word(a.outcome())));
        return out;
    }

    // Headline result.
    match a.margin_seeds() {
        Some(m) => {
            let word = outcome_word(a.outcome());
            let by = m.abs();
            out.push_str(&format!(
                "Result for {}: {} (perfect play ends {} {} seed{})\n",
                board.turn(),
                word,
                if m >= 0 { "+" } else { "-" },
                by,
                if by == 1 { "" } else { "s" }
            ));
        }
        None => {
            out.push_str(&format!(
                "Best guess for {}: {} (heuristic score {})\n",
                board.turn(),
                outcome_word(a.outcome()),
                a.value
            ));
        }
    }

    out.push_str(&format!("Best move: pit {}\n\n", a.best_move.unwrap()));

    // Per-move table.
    out.push_str("Move evaluations (best first):\n");
    let unit = if a.exact { "seeds" } else { "score" };
    for (rank, m) in a.move_evals.iter().enumerate() {
        // Inferior moves carry only an upper bound (`≤`); see `MoveEval::bound`.
        let approx = if m.bound { "≤ " } else { "" };
        out.push_str(&format!(
            "  {marker} pit {pit}: {approx}{sign}{val} {unit}{tags}\n",
            marker = if rank == 0 { "*" } else { " " },
            pit = m.pit,
            approx = approx,
            sign = if m.value >= 0 { "+" } else { "-" },
            val = m.value.abs(),
            unit = unit,
            tags = move_tags(m),
        ));
    }
    if a.move_evals.iter().any(|m| m.bound) {
        out.push_str("  (≤ marks moves proven no better than the best; exact value not computed)\n");
    }

    // Principal variation.
    if !a.pv.is_empty() {
        out.push('\n');
        out.push_str("Principal variation:\n  ");
        out.push_str(&render_pv(board, &a.pv));
        out.push('\n');
    }

    out
}

fn move_tags(m: &MoveEval) -> String {
    let mut tags = Vec::new();
    if m.extra_turn {
        tags.push("extra turn");
    }
    if m.captured {
        tags.push("capture");
    }
    if tags.is_empty() {
        String::new()
    } else {
        format!("  [{}]", tags.join(", "))
    }
}

/// Replay the principal variation to annotate each move with the player and
/// any extra-turn / capture flags.
fn render_pv(board: &Board, pv: &[usize]) -> String {
    let rules = Rules::default();
    let mut cur = *board;
    let mut parts = Vec::new();
    for &mv in pv {
        if cur.is_terminal() || mv >= cur.pits_per_side() {
            break;
        }
        let mover = cur.turn();
        let r = cur.apply(&rules, mv);
        let tag = match (r.extra_turn, r.captured) {
            (true, _) => "↻",
            (_, true) => "×",
            _ => "",
        };
        let label = match mover {
            Player::P0 => "P0",
            Player::P1 => "P1",
        };
        parts.push(format!("{label}:{mv}{tag}"));
        cur = r.board;
    }
    parts.join("  →  ")
}

fn outcome_word(o: Outcome) -> &'static str {
    match o {
        Outcome::Win => "WIN",
        Outcome::Loss => "LOSS",
        Outcome::Draw => "DRAW",
    }
}

fn print_help() {
    println!(
        r#"mancala-solver — a Kalah-style Mancala position solver

USAGE:
    mancala-solver <COMMAND> [OPTIONS]

COMMANDS:
    analyze   Analyze a position given in board notation
    start     Analyze the standard opening for a given board size
    gen-tb    Build an offline endgame tablebase and write it to disk
    help      Show this help

ANALYZE:
    mancala-solver analyze --board "<notation>" [OPTIONS]

    Board notation is two comma-separated groups separated by '|':
        p0_pit0,...,p0_pitN-1,p0_store | p1_pit0,...,p1_pitN-1,p1_store
    The number of pits per side is inferred from the notation.

    Example:
        mancala-solver analyze --board "4,4,4,4,4,4,0 | 4,4,4,4,4,4,0"

START:
    mancala-solver start --pits 6 --seeds 4 [OPTIONS]

GEN-TB:
    mancala-solver gen-tb --pits 6 --seeds-cap 14 --out kalah6.tb [--capture-empty]

    Precomputes exact endgame values for every pit layout with up to
    <seeds-cap> seeds in play, writing a tablebase file. Pass it to analyze/start
    with --tb to make those endgame positions O(1) lookups.

OPTIONS (analyze & start):
    --turn <0|1>        Side to move (0 = P0/South, 1 = P1/North). Default: 0
    --budget <N>        Exact-search node budget before falling back to the
                        heuristic search. Default: {budget}
    --depth <D>         Heuristic fallback search depth in plies. Default: {depth}
    --capture-empty     Allow captures even when the opposite pit is empty
                        (default: captures require a non-empty opposite pit)
    --tb <file>         Use a specific precomputed endgame tablebase (see gen-tb)
    --no-tb             Disable the automatic endgame tablebase
    --seeds-cap <N>     Seed cap for the automatic tablebase (default 14)

ENDGAME TABLEBASE (automatic):
    For boards up to 6 pits per side with enough seeds in play, an endgame
    tablebase is built and cached automatically on first use (under
    $MANCALA_TB_DIR, default ./.mancala_tb), then reused on later runs. It makes
    endgame positions O(1) lookups. Use --no-tb to disable, or gen-tb to build
    one explicitly.

OUTPUT:
    A board diagram, the exact result (or heuristic estimate), the best move,
    a per-move evaluation table, and the principal variation.
"#,
        budget = DEFAULT_BUDGET,
        depth = DEFAULT_FALLBACK_DEPTH,
    );
}
