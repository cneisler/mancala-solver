// Analysis worker. Runs the wasm engine off the main thread so a long exact
// search never freezes the page. Owns its own wasm instance and the endgame
// tablebase; the main thread keeps a separate instance for instant moves.

let wasm = null;
const enc = new TextEncoder();
const dec = new TextDecoder();

async function loadWasm() {
  try {
    const { instance } = await WebAssembly.instantiateStreaming(fetch("mancala.wasm"), {});
    wasm = instance.exports;
  } catch {
    const bytes = await (await fetch("mancala.wasm")).arrayBuffer();
    wasm = (await WebAssembly.instantiate(bytes, {})).instance.exports;
  }
}

async function loadTb() {
  try {
    const bytes = new Uint8Array(await (await fetch("kalah6.bin")).arrayBuffer());
    const ptr = wasm.wasm_alloc(bytes.length);
    new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
    const ok = wasm.wasm_set_tb(ptr, bytes.length);
    wasm.wasm_dealloc(ptr, bytes.length);
    return ok === 1;
  } catch {
    return false;
  }
}

function readResult(ptr) {
  const len = new DataView(wasm.memory.buffer).getUint32(ptr, true);
  const bytes = new Uint8Array(wasm.memory.buffer, ptr + 4, len).slice();
  wasm.wasm_dealloc(ptr, 4 + len);
  return JSON.parse(dec.decode(bytes));
}

function analyze(notation, turn, budget, depth, capEmpty) {
  const bytes = enc.encode(notation);
  const ptr = wasm.wasm_alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  const res = wasm.wasm_analyze(ptr, bytes.length, turn, budget, depth, capEmpty);
  wasm.wasm_dealloc(ptr, bytes.length);
  return readResult(res);
}

onmessage = (e) => {
  const m = e.data;
  if (m.type !== "analyze") return;
  try {
    const data = analyze(m.notation, m.turn, m.budget, m.depth, m.capEmpty);
    postMessage({ type: "result", id: m.id, data });
  } catch (err) {
    postMessage({ type: "result", id: m.id, error: String(err && err.message ? err.message : err) });
  }
};

(async function () {
  await loadWasm();
  const tb = await loadTb();
  postMessage({ type: "ready", tb });
})();
