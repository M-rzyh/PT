#!/usr/bin/env python3
"""Web preference labeller for the PT LunarLander pipeline.

A slimmer variant of `BPref3/label_web.py` (same HTML/JS, no PEBBLE-specific
fields). Reads the batch_NNN/pair_NNN/{seg_A,seg_B}.mp4 layout produced by
`render_lunarlander_segments.py`, plays paired videos in the browser, records
labels in {-1, 0, 1}, and saves them to a pickle.

Output (pickle, written after every label):

    {
        'labels':        [int|None, ...]   # length == #pairs labeled-or-skipped
        'pair_starts_a': [int, ...]        # start index of seg_A in source HDF5
        'pair_starts_b': [int, ...]        # start index of seg_B in source HDF5
        'time_sec':      [float, ...]      # decision time per labeled pair
        'metadata':      {hdf5, query_len, ...}  # from the renderer
    }

Use `labels_web_to_pt_format.py` to convert this to the PT
`human_label/<env>/seed_<N>/{indices_num<M>_q<L>, indices_2_num<M>_q<L>,
label_human}` triple.

Usage
-----
    python -m scripts.preference.label_web \
        --query_dir $SCRATCH/PT/lunarlander/labels/lunarlander-medium-replay-v2/seed_0/videos \
        --output    $SCRATCH/PT/lunarlander/labels/lunarlander-medium-replay-v2/seed_0/labels.pkl \
        --port 8080
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_file
from flask import request as flask_request


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all_pairs(query_dir: Path) -> tuple[list, dict]:
    """Walk the query_dir and return (pairs, top-level-metadata).

    pairs: list of dicts with keys batch_idx, pair_idx, seg_a, seg_b, idx_a, idx_b.
    """
    index_path = query_dir / "index.pkl"
    if index_path.exists():
        with open(index_path, "rb") as f:
            batch_dirs = [Path(p) for p in pickle.load(f)]
    else:
        batch_dirs = sorted(query_dir.glob("batch_*"))

    pairs: list[dict] = []
    top_meta: dict = {}
    for batch_dir in batch_dirs:
        meta_path = batch_dir / "metadata.pkl"
        if not meta_path.exists():
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if not top_meta:
            top_meta = {k: meta[k] for k in ("hdf5", "query_len", "fps") if k in meta}
        idx_1 = meta.get("indices_1", [])
        idx_2 = meta.get("indices_2", [])
        for i in range(meta["n_pairs"]):
            pair_dir = batch_dir / f"pair_{i:03d}"
            seg_a, seg_b = pair_dir / "seg_A.mp4", pair_dir / "seg_B.mp4"
            if seg_a.exists() and seg_b.exists():
                pairs.append({
                    "batch_idx": meta["batch_idx"],
                    "pair_idx": i,
                    "seg_a": str(seg_a),
                    "seg_b": str(seg_b),
                    "idx_a": int(idx_1[i]) if i < len(idx_1) else -1,
                    "idx_b": int(idx_2[i]) if i < len(idx_2) else -1,
                })
    return pairs, top_meta


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
ALL_PAIRS: list[dict] = []
TOP_META: dict = {}
QUERY_DIR: Path = Path(".")
LABELS: list[dict] = []     # all clicks (including Skip/None)
LABELED_KEYS: set = set()   # (batch_idx, pair_idx) — survives refresh
OUTPUT_PATH: Path = Path("labels.pkl")


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PT Preference Labeling - LunarLander</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f172a; color: #e2e8f0; min-height: 100vh; }
.header { text-align: center; padding: 20px; background: #1e293b; }
.header h1 { font-size: 22px; color: #f8fafc; }
.progress { font-size: 14px; color: #94a3b8; margin-top: 6px; }
.main { max-width: 1100px; margin: 20px auto; padding: 0 20px; }
.videos { display: flex; gap: 24px; justify-content: center; }
.video-panel { flex: 1; max-width: 520px; }
.video-panel h2 { text-align: center; font-size: 18px; margin-bottom: 10px;
                   padding: 8px; border-radius: 8px; }
.panel-a h2 { background: #1e3a5f; color: #60a5fa; }
.panel-b h2 { background: #3b1f3b; color: #c084fc; }
video { width: 100%; border-radius: 8px; background: #000; }
.controls { display: flex; gap: 12px; justify-content: center; margin: 24px 0; flex-wrap: wrap; }
.btn { padding: 14px 32px; border: 2px solid transparent; border-radius: 10px;
       font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.btn:hover { transform: translateY(-2px); }
.btn-a { background: #1e3a5f; color: #60a5fa; border-color: #2563eb; }
.btn-a:hover { background: #2563eb; color: #fff; }
.btn-b { background: #3b1f3b; color: #c084fc; border-color: #7c3aed; }
.btn-b:hover { background: #7c3aed; color: #fff; }
.btn-eq { background: #1e293b; color: #94a3b8; border-color: #475569; }
.btn-eq:hover { background: #475569; color: #fff; }
.btn-skip { background: #1e293b; color: #64748b; border-color: #334155; }
.btn-skip:hover { background: #334155; color: #fff; }
.btn-replay { background: #0f172a; color: #38bdf8; border-color: #0284c7; padding: 10px 20px; font-size: 14px; }
.btn-replay:hover { background: #0284c7; color: #fff; }
.btn-pause { background: #1e293b; color: #fbbf24; border-color: #d97706; padding: 10px 20px; font-size: 14px; }
.btn-pause:hover { background: #d97706; color: #fff; }
.btn-pause.paused { background: #d97706; color: #fff; }
.replay-row { display: flex; gap: 12px; justify-content: center; margin: 12px 0; }
.toolbar { display: flex; gap: 12px; justify-content: center; margin: 16px 0; }
.feedback { text-align: center; font-size: 16px; margin: 10px 0; min-height: 24px; color: #22c55e; }
.timer { text-align: center; font-size: 14px; color: #64748b; margin: 6px 0; }
.paused-banner { text-align: center; font-size: 18px; color: #fbbf24; margin: 10px 0;
                  padding: 10px; background: #422006; border-radius: 8px; display: none; }
.stats { text-align: center; font-size: 13px; color: #475569; margin-top: 20px; padding-top: 16px; border-top: 1px solid #1e293b; }
.done-screen { text-align: center; padding: 60px 20px; }
.done-screen h2 { font-size: 28px; color: #22c55e; margin-bottom: 12px; }
.keyboard-hint { text-align: center; font-size: 13px; color: #475569; margin-top: 8px; }
</style>
</head>
<body>
<div class="header">
  <h1>PT Preference Labeling - LunarLander</h1>
  <div class="progress" id="progress">Loading...</div>
</div>
<div class="main">
  <div id="labeling-ui">
    <div class="videos">
      <div class="video-panel panel-a">
        <h2>Segment A</h2>
        <video id="vid-a" muted></video>
      </div>
      <div class="video-panel panel-b">
        <h2>Segment B</h2>
        <video id="vid-b" muted></video>
      </div>
    </div>
    <div class="replay-row">
      <button class="btn btn-replay" onclick="replay('a')">Replay A</button>
      <button class="btn btn-replay" onclick="replay('b')">Replay B</button>
      <button class="btn btn-replay" onclick="replay('both')">Replay Both</button>
    </div>
    <div class="toolbar">
      <button class="btn btn-pause" id="pause-btn" onclick="togglePause()">Pause (p)</button>
    </div>
    <div class="paused-banner" id="paused-banner">PAUSED - timer stopped. Press P or click Pause to resume.</div>
    <div class="timer" id="timer"></div>
    <div class="controls" id="choice-controls">
      <button class="btn btn-a" onclick="choose(-1)">A is better (1)</button>
      <button class="btn btn-eq" onclick="choose(0)">Equal (0)</button>
      <button class="btn btn-b" onclick="choose(1)">B is better (2)</button>
      <button class="btn btn-skip" onclick="choose(null)">Skip (s)</button>
    </div>
    <div class="keyboard-hint">Keyboard: 1=A, 2=B, 0=Equal, s=Skip, r=Replay, p=Pause</div>
    <div class="feedback" id="feedback"></div>
  </div>
  <div id="done-screen" style="display:none" class="done-screen">
    <h2>Caught up!</h2>
    <p id="done-msg"></p>
  </div>
  <div class="stats" id="stats"></div>
</div>
<script>
let idx = 0, total = 0, labeled = 0, cumTime = 0;
let startTime = null, pausedAt = null, pausedElapsed = 0, paused = false;

async function init() {
  const r = await fetch('/api/status');
  const d = await r.json();
  total = d.total;
  labeled = d.labeled;
  idx = d.current;
  cumTime = d.cumulative_time;
  paused = false;
  pausedAt = null;
  pausedElapsed = 0;
  document.getElementById('paused-banner').style.display = 'none';
  document.getElementById('pause-btn').classList.remove('paused');
  document.getElementById('pause-btn').textContent = 'Pause (p)';
  updateStats();
  if (idx < total) {
    document.getElementById('labeling-ui').style.display = '';
    document.getElementById('done-screen').style.display = 'none';
    loadPair(idx);
  } else {
    showDone();
  }
}

function updateStats() {
  document.getElementById('progress').textContent =
    `Query ${Math.min(idx+1, total)} / ${total}  |  Labeled: ${labeled}`;
  document.getElementById('stats').textContent =
    `Labeled: ${labeled}  |  Total time: ${cumTime.toFixed(1)}s`;
}

function loadPair(i) {
  const va = document.getElementById('vid-a');
  const vb = document.getElementById('vid-b');
  va.src = '/video/' + i + '/a?' + Date.now();
  vb.src = '/video/' + i + '/b?' + Date.now();
  va.load(); vb.load();
  let loadedCount = 0;
  function onLoaded() {
    loadedCount++;
    if (loadedCount === 2 && !paused) { va.play(); vb.play(); }
  }
  va.onloadeddata = onLoaded;
  vb.onloadeddata = onLoaded;
  startTime = performance.now();
  pausedElapsed = 0;
  document.getElementById('feedback').textContent = '';
  updateStats();
  updateTimer();
}

function getElapsed() {
  if (!startTime) return 0;
  if (paused && pausedAt) return pausedElapsed + (pausedAt - startTime) / 1000;
  return pausedElapsed + (performance.now() - startTime) / 1000;
}

function updateTimer() {
  if (idx < total) {
    const el = getElapsed();
    const label = paused ? 'PAUSED' : 'Decision time';
    document.getElementById('timer').textContent = `${label}: ${el.toFixed(1)}s`;
    if (!paused) requestAnimationFrame(updateTimer);
  }
}

function togglePause() {
  const va = document.getElementById('vid-a');
  const vb = document.getElementById('vid-b');
  if (!paused) {
    paused = true;
    pausedAt = performance.now();
    va.pause(); vb.pause();
    document.getElementById('paused-banner').style.display = '';
    document.getElementById('pause-btn').classList.add('paused');
    document.getElementById('pause-btn').textContent = 'Resume (p)';
    updateTimer();
  } else {
    paused = false;
    pausedElapsed += (performance.now() - pausedAt) / 1000;
    startTime += (performance.now() - pausedAt);
    pausedAt = null;
    va.play(); vb.play();
    document.getElementById('paused-banner').style.display = 'none';
    document.getElementById('pause-btn').classList.remove('paused');
    document.getElementById('pause-btn').textContent = 'Pause (p)';
    updateTimer();
  }
}

function replay(which) {
  if (paused) return;
  const va = document.getElementById('vid-a');
  const vb = document.getElementById('vid-b');
  if (which === 'a' || which === 'both') { va.currentTime = 0; va.play(); }
  if (which === 'b' || which === 'both') { vb.currentTime = 0; vb.play(); }
}

async function choose(label) {
  if (idx >= total || paused) return;
  const dt = getElapsed() - pausedElapsed;
  if (label !== null) { cumTime += dt; labeled++; }
  const r = await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pair_index: idx, label: label, time_sec: dt})
  });
  const d = await r.json();
  if (label !== null) {
    const names = {'-1': 'A preferred', 1: 'B preferred', 0: 'Equal'};
    document.getElementById('feedback').textContent =
      (names[String(label)] || 'Labeled') + ` (${dt.toFixed(1)}s)`;
  } else {
    document.getElementById('feedback').textContent = 'Skipped';
  }
  idx = d.current;
  updateStats();
  if (idx < total) {
    setTimeout(() => loadPair(idx), 400);
  } else {
    showDone();
  }
}

function showDone() {
  document.getElementById('labeling-ui').style.display = 'none';
  document.getElementById('done-screen').style.display = 'block';
  document.getElementById('done-msg').textContent =
    `Labeled ${labeled} / ${total} pairs in ${cumTime.toFixed(1)}s. Labels saved to ${'__OUTPUT__'}.`;
}

document.addEventListener('keydown', e => {
  if (e.key === '1') choose(-1);
  else if (e.key === '2') choose(1);
  else if (e.key === '0') choose(0);
  else if (e.key === 's' || e.key === 'S') choose(null);
  else if (e.key === 'r' || e.key === 'R') replay('both');
  else if (e.key === 'p' || e.key === 'P') togglePause();
});

init();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML_PAGE.replace("__OUTPUT__", str(OUTPUT_PATH))


@app.route("/video/<int:pair_idx>/<segment>")
def serve_video(pair_idx: int, segment: str):
    if pair_idx < 0 or pair_idx >= len(ALL_PAIRS):
        return "Not found", 404
    p = ALL_PAIRS[pair_idx]
    path = p["seg_a"] if segment == "a" else p["seg_b"]
    return send_file(path, mimetype="video/mp4")


@app.route("/api/status")
def api_status():
    labeled_count = sum(1 for l in LABELS if l["label"] is not None)
    cum_time = sum(l["time_sec"] for l in LABELS if l["label"] is not None)
    return jsonify({
        "total": len(ALL_PAIRS),
        "labeled": labeled_count,
        "current": len(LABELS),
        "cumulative_time": round(cum_time, 3),
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    data = flask_request.get_json()
    pair_idx = data["pair_index"]
    label = data["label"]
    time_sec = float(data.get("time_sec", 0))
    pair = ALL_PAIRS[pair_idx]
    LABELS.append({
        "pair_index": pair_idx,
        "batch_idx": pair["batch_idx"],
        "pair_idx_in_batch": pair["pair_idx"],
        "label": label,
        "time_sec": round(time_sec, 3),
    })
    _save_labels()
    labeled_count = sum(1 for l in LABELS if l["label"] is not None)
    cum_time = sum(l["time_sec"] for l in LABELS if l["label"] is not None)
    return jsonify({
        "ok": True, "current": len(LABELS),
        "labeled": labeled_count,
        "cumulative_time": round(cum_time, 3),
    })


def _save_labels() -> None:
    """Write the in-memory LABELS to OUTPUT_PATH after every click."""
    out = {
        "labels":        [l["label"] for l in LABELS],
        "pair_starts_a": [ALL_PAIRS[l["pair_index"]]["idx_a"] for l in LABELS],
        "pair_starts_b": [ALL_PAIRS[l["pair_index"]]["idx_b"] for l in LABELS],
        "time_sec":      [l["time_sec"] for l in LABELS],
        "metadata":      TOP_META,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(out, f)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query_dir", required=True, type=Path,
                        help="Dir from render_lunarlander_segments.py (contains batch_NNN/ + index.pkl).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Where to save the labels pickle.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global ALL_PAIRS, TOP_META, QUERY_DIR, OUTPUT_PATH
    QUERY_DIR = args.query_dir
    OUTPUT_PATH = args.output
    ALL_PAIRS, TOP_META = load_all_pairs(QUERY_DIR)
    print(f"[label_web] loaded {len(ALL_PAIRS)} pairs from {QUERY_DIR}")
    print(f"[label_web] meta: {TOP_META}")
    print(f"[label_web] writing labels to {OUTPUT_PATH} (auto-saved after each click)")
    print(f"[label_web] starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
