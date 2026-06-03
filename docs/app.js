// Mancala Solver — browser front-end. Loads the Rust engine compiled to wasm and
// drives an interactive Kalah board. All computation is client-side.

let wasm = null;

async function loadWasm() {
  // Prefer streaming; fall back to ArrayBuffer if the host serves the wasm with
  // the wrong MIME type (some static hosts do).
  try {
    const { instance } = await WebAssembly.instantiateStreaming(fetch("mancala.wasm"), {});
    wasm = instance.exports;
  } catch {
    const bytes = await (await fetch("mancala.wasm")).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    wasm = instance.exports;
  }
}

// --- wasm string marshalling -------------------------------------------------
const enc = new TextEncoder();
const dec = new TextDecoder();

function readResult(ptr) {
  const len = new DataView(wasm.memory.buffer).getUint32(ptr, true);
  const bytes = new Uint8Array(wasm.memory.buffer, ptr + 4, len).slice();
  wasm.wasm_dealloc(ptr, 4 + len);
  return JSON.parse(dec.decode(bytes));
}

// Call a wasm fn whose first two args are (ptr, len) of a board string.
function callBoard(fn, notation, ...rest) {
  const bytes = enc.encode(notation);
  const ptr = wasm.wasm_alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  const res = fn(ptr, bytes.length, ...rest);
  wasm.wasm_dealloc(ptr, bytes.length);
  return readResult(res);
}

const engine = {
  start: (pits, seeds, turn) => readResult(wasm.wasm_start(pits, seeds, turn)),
  apply: (notation, turn, mv, capEmpty) =>
    callBoard(wasm.wasm_apply, notation, turn, mv, capEmpty),
  analyze: (notation, turn, budget, depth, capEmpty) =>
    callBoard(wasm.wasm_analyze, notation, turn, budget, depth, capEmpty),
};

// --- game state --------------------------------------------------------------
const state = { notation: "", turn: 0, pits: 6, history: [], lastAnalysis: null };

function parseNotation(s) {
  const [p0, p1] = s.split("|").map((g) => g.trim().split(",").map(Number));
  const n = p0.length - 1;
  return {
    n,
    p0pits: p0.slice(0, n),
    p0store: p0[n],
    p1pits: p1.slice(0, n),
    p1store: p1[n],
  };
}

function capEmpty() {
  return document.getElementById("captureEmpty").checked ? 1 : 0;
}

// --- rendering ---------------------------------------------------------------
function render() {
  const b = parseNotation(state.notation);
  const board = document.getElementById("board");
  const best =
    state.lastAnalysis && state.lastAnalysis.matches(state.notation, state.turn)
      ? state.lastAnalysis.data.best
      : -1;

  const pit = (side, idx, seeds) => {
    const playable = side === state.turn && seeds > 0;
    const cls = ["pit"];
    if (playable) cls.push("playable");
    if (side === state.turn && idx === best) cls.push("best");
    const el = document.createElement("div");
    el.className = cls.join(" ");
    el.innerHTML = `<span class="lot">lot ${idx + 1}</span><span class="count">${seeds}</span>`;
    if (playable) el.onclick = () => playMove(idx);
    return el;
  };

  const p1row = document.createElement("div");
  p1row.className = "row";
  for (let i = b.n - 1; i >= 0; i--) p1row.appendChild(pit(1, i, b.p1pits[i]));
  const p0row = document.createElement("div");
  p0row.className = "row";
  for (let i = 0; i < b.n; i++) p0row.appendChild(pit(0, i, b.p0pits[i]));

  const rows = document.createElement("div");
  rows.className = "rows";
  rows.append(p1row, p0row);

  const store = (count, who) => {
    const el = document.createElement("div");
    el.className = "store";
    el.innerHTML = `<span class="count">${count}</span><span class="who">${who}</span>`;
    return el;
  };

  board.replaceChildren(store(b.p1store, "P1 (N)"), rows, store(b.p0store, "P0 (S)"));

  document.getElementById("turnLabel").textContent =
    `${state.turn === 0 ? "P0 (South)" : "P1 (North)"} to move`;
  document.getElementById("undo").disabled = state.history.length === 0;
  document.getElementById("playBest").disabled = false;
}

function playMove(mv) {
  const r = engine.apply(state.notation, state.turn, mv, capEmpty());
  if (r.error) {
    showError(r.error);
    return;
  }
  state.history.push({ notation: state.notation, turn: state.turn });
  state.notation = r.notation;
  state.turn = r.turn;
  state.lastAnalysis = null;
  render();
  let msg = `Played lot ${mv + 1}.`;
  if (r.extra_turn) msg += " Extra turn!";
  if (r.captured) msg += " Capture!";
  if (r.terminal) msg += " — game over.";
  document.getElementById("result").innerHTML = `<p>${msg}</p>`;
}

// --- analysis ----------------------------------------------------------------
function lots(pits) {
  return pits.map((p) => p + 1);
}

// Build a readable principal-variation string by walking the moves.
function formatPv(pv) {
  let cur = state.notation, turn = state.turn;
  const parts = [];
  for (const mv of pv) {
    const r = engine.apply(cur, turn, mv, capEmpty());
    if (r.error) break;
    let tag = "";
    if (r.extra_turn) tag = "↻";
    else if (r.captured) tag = "×";
    parts.push(`${turn === 0 ? "P0" : "P1"}:lot${mv + 1}${tag}`);
    cur = r.notation;
    turn = r.turn;
  }
  return parts.join("  →  ");
}

function analyze() {
  const budget = Number(document.getElementById("budget").value);
  const a = engine.analyze(state.notation, state.turn, budget, 11, capEmpty());
  if (a.error) {
    showError(a.error);
    return null;
  }
  state.lastAnalysis = {
    data: a,
    matches: (n, t) => n === state.notation && t === state.turn,
  };
  renderAnalysis(a);
  render(); // re-render to highlight the best pit
  return a;
}

function renderAnalysis(a) {
  const who = a.side === 0 ? "P0 (South)" : "P1 (North)";
  const res = document.getElementById("result");
  const unit = a.exact ? "seeds" : "(heuristic score)";
  let verdict;
  if (a.best < 0) {
    verdict = `${who}: terminal — ${a.outcome} by ${Math.abs(a.value)} seeds`;
  } else if (a.exact) {
    verdict = `${who}: ${a.outcome} — perfect play ends ${a.value >= 0 ? "+" : ""}${a.value} seeds`;
  } else {
    verdict = `${who}: best guess ${a.outcome} (heuristic, depth ${a.depth}; not proven)`;
  }

  let html = `<p class="verdict">${verdict}</p>`;
  if (a.best >= 0) {
    html += `<p>Best move: <strong>lot ${a.best + 1}</strong> · ${a.exact ? "exact" : "heuristic"} · ${a.nodes.toLocaleString()} nodes</p>`;
    html += `<div class="evals">`;
    for (const e of a.evals) {
      const isBest = e.pit === a.best;
      const bound = e.bound ? "≤ " : "";
      const sign = e.value >= 0 ? "+" : "";
      const tags = [e.extra ? "extra turn" : "", e.capture ? "capture" : ""].filter(Boolean).join(", ");
      html += `<span class="${isBest ? "b" : ""}">${isBest ? "★" : ""} lot ${e.pit + 1}</span>`;
      html += `<span class="${isBest ? "b" : ""}">${bound}${sign}${e.value} ${unit}</span>`;
      html += `<span>${tags}</span>`;
    }
    html += `</div>`;
    if (a.pv.length) html += `<p class="pv">PV: ${formatPv(a.pv)}</p>`;
  }
  res.className = "result " + a.outcome.toLowerCase();
  res.innerHTML = html;
}

function playBest() {
  const a = state.lastAnalysis && state.lastAnalysis.matches(state.notation, state.turn)
    ? state.lastAnalysis.data
    : analyze();
  if (a && a.best >= 0) playMove(a.best);
}

function showError(msg) {
  document.getElementById("result").innerHTML = `<p class="err">Error: ${msg}</p>`;
}

// --- new game / undo ---------------------------------------------------------
function newGame() {
  const pits = Math.max(1, Math.min(9, Number(document.getElementById("pits").value)));
  const seeds = Math.max(0, Math.min(20, Number(document.getElementById("seeds").value)));
  const r = engine.start(pits, seeds, 0);
  if (r.error) {
    showError(r.error);
    return;
  }
  state.notation = r.notation;
  state.turn = r.turn;
  state.pits = pits;
  state.history = [];
  state.lastAnalysis = null;
  document.getElementById("result").innerHTML = "";
  render();
}

function undo() {
  const prev = state.history.pop();
  if (!prev) return;
  state.notation = prev.notation;
  state.turn = prev.turn;
  state.lastAnalysis = null;
  document.getElementById("result").innerHTML = "";
  render();
}

// --- boot --------------------------------------------------------------------
(async function () {
  try {
    await loadWasm();
    document.getElementById("status").textContent = "engine ready";
    document.getElementById("newGame").onclick = newGame;
    document.getElementById("undo").onclick = undo;
    document.getElementById("analyze").onclick = analyze;
    document.getElementById("playBest").onclick = playBest;
    newGame();
  } catch (e) {
    document.getElementById("status").innerHTML = `<span class="err">failed to load engine: ${e}</span>`;
  }
})();
