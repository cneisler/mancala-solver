//! WebAssembly API: a tiny C-ABI shim so the engine can run in a browser with no
//! server. Strings cross the boundary through linear memory — JS writes input
//! into a buffer from [`wasm_alloc`], calls a function, and reads a
//! length-prefixed (`u32` little-endian length, then UTF-8 bytes) result, then
//! frees both buffers. Only compiled for `wasm32` (see `lib.rs`).
//!
//! No `wasm-bindgen` / external deps: the whole thing is hand-rolled so the
//! crate stays dependency-free.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;

use crate::board::{Board, Player, Rules};
use crate::notation;
use crate::solver;
use crate::tablebase::Tablebase;

thread_local! {
    /// Optional endgame tablebase supplied by the host (see [`wasm_set_tb`]).
    static TB: RefCell<Option<Tablebase>> = const { RefCell::new(None) };
}

/// Allocate `len` bytes for JS to write into. `len == 0` allocates 1 byte.
#[no_mangle]
pub extern "C" fn wasm_alloc(len: usize) -> *mut u8 {
    let layout = Layout::from_size_align(len.max(1), 1).unwrap();
    unsafe { alloc(layout) }
}

/// Free a buffer previously returned by [`wasm_alloc`] or a result pointer.
#[no_mangle]
pub extern "C" fn wasm_dealloc(ptr: *mut u8, len: usize) {
    let layout = Layout::from_size_align(len.max(1), 1).unwrap();
    unsafe { dealloc(ptr, layout) }
}

/// Pack a string into a freshly-allocated `[len: u32 LE][bytes]` buffer and
/// return its pointer. JS reads the length, then the bytes, then frees `4 + len`.
fn ret(s: String) -> *mut u8 {
    let bytes = s.into_bytes();
    let len = bytes.len();
    let buf = wasm_alloc(4 + len);
    unsafe {
        std::ptr::copy_nonoverlapping((len as u32).to_le_bytes().as_ptr(), buf, 4);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.add(4), len);
    }
    buf
}

unsafe fn read(ptr: *const u8, len: usize) -> String {
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
}

fn player(turn: u32) -> Player {
    if turn == 1 {
        Player::P1
    } else {
        Player::P0
    }
}

fn rules(capture_empty: u32) -> Rules {
    let mut r = Rules::default();
    if capture_empty != 0 {
        r.capture_requires_nonempty_opposite = false;
    }
    r
}

fn json_error(msg: &str) -> String {
    format!("{{\"error\":\"{}\"}}", msg.replace('"', "'"))
}

/// `start(pits, seeds, turn)` → JSON `{notation, turn}` for the opening position.
#[no_mangle]
pub extern "C" fn wasm_start(pits: u32, seeds: u32, turn: u32) -> *mut u8 {
    if seeds > u8::MAX as u32 {
        return ret(json_error("seeds out of range"));
    }
    match Board::start(pits as usize, seeds as u8, player(turn)) {
        Ok(b) => ret(format!(
            "{{\"notation\":\"{}\",\"turn\":{}}}",
            notation::format(&b),
            side(&b)
        )),
        Err(e) => ret(json_error(&e)),
    }
}

/// `apply(board, turn, mv, capture_empty)` → JSON describing the new position
/// after playing pit `mv`, or an `{error}` if the move is illegal.
#[no_mangle]
pub extern "C" fn wasm_apply(
    ptr: *const u8,
    len: usize,
    turn: u32,
    mv: u32,
    capture_empty: u32,
) -> *mut u8 {
    let s = unsafe { read(ptr, len) };
    let board = match notation::parse(&s, player(turn)) {
        Ok(b) => b,
        Err(e) => return ret(json_error(&e)),
    };
    let mv = mv as usize;
    if board.is_terminal() {
        return ret(json_error("game is already over"));
    }
    if mv >= board.pits_per_side() || board.cells()[board.pit_global(board.turn(), mv)] == 0 {
        return ret(json_error("illegal move"));
    }
    let r = board.apply(&rules(capture_empty), mv);
    ret(format!(
        "{{\"notation\":\"{}\",\"turn\":{},\"extra_turn\":{},\"captured\":{},\"terminal\":{}}}",
        notation::format(&r.board),
        side(&r.board),
        r.extra_turn,
        r.captured,
        r.board.is_terminal()
    ))
}

/// Install an endgame tablebase (raw file bytes) for the engine to use. Returns
/// 1 on success, 0 if the bytes are not a valid tablebase. A tablebase only
/// applies to positions of its own board size.
#[no_mangle]
pub extern "C" fn wasm_set_tb(ptr: *const u8, len: usize) -> u32 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    match Tablebase::from_bytes(bytes) {
        Ok(tb) => {
            TB.with(|c| *c.borrow_mut() = Some(tb));
            1
        }
        Err(_) => 0,
    }
}

/// `analyze(board, turn, budget, fallback_depth, capture_empty)` → JSON analysis,
/// consulting the installed tablebase (if any) for the endgame.
#[no_mangle]
pub extern "C" fn wasm_analyze(
    ptr: *const u8,
    len: usize,
    turn: u32,
    budget: u32,
    fallback_depth: u32,
    capture_empty: u32,
) -> *mut u8 {
    let s = unsafe { read(ptr, len) };
    let board = match notation::parse(&s, player(turn)) {
        Ok(b) => b,
        Err(e) => return ret(json_error(&e)),
    };
    let json = TB.with(|c| {
        let tb = c.borrow();
        let a = solver::analyze_with_tb(
            &board,
            rules(capture_empty),
            budget as u64,
            fallback_depth,
            tb.as_ref(),
        );
        analysis_json(&a)
    });
    ret(json)
}

fn side(b: &Board) -> u32 {
    if b.turn() == Player::P1 {
        1
    } else {
        0
    }
}

fn analysis_json(a: &solver::Analysis) -> String {
    use std::fmt::Write;
    let outcome = match a.outcome() {
        solver::Outcome::Win => "WIN",
        solver::Outcome::Loss => "LOSS",
        solver::Outcome::Draw => "DRAW",
    };
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"exact\":{},\"side\":{},\"value\":{},\"outcome\":\"{}\",\"best\":{},\"nodes\":{},\"depth\":{},",
        a.exact,
        if a.side_to_move == Player::P1 { 1 } else { 0 },
        a.value,
        outcome,
        a.best_move.map_or(-1i32, |m| m as i32),
        a.nodes,
        a.depth.map_or(-1i32, |d| d as i32),
    );
    s.push_str("\"pv\":[");
    for (i, mv) in a.pv.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{mv}");
    }
    s.push_str("],\"evals\":[");
    for (i, e) in a.move_evals.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(
            s,
            "{{\"pit\":{},\"value\":{},\"bound\":{},\"extra\":{},\"capture\":{}}}",
            e.pit, e.value, e.bound, e.extra_turn, e.captured
        );
    }
    s.push_str("]}");
    s
}
