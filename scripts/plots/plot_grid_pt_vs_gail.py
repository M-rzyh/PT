"""Combined PT-vs-GAIL grid plots: count axis (x = human time) and noise
axis. Style follows IRL3's plot_gail_grid.py annotated charts.

Count axis GAIL: new 30-seed shuffle runs (gail_grid_count_2026-06-18.csv),
                 eval_reward_mean from agent_rollouts.npz.
Noise axis GAIL: old 10-seed nonFS runs (gail_grid_2026-05-12.csv),
                 eval_reward_mean column.
PT data:         per-task eval_summary.json under grid_mixture_ms.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.ipc


SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
GRID_ROOT = SCRATCH / "PT/lunarlander/grid_mixture"
GRID_ROOT_MS = SCRATCH / "PT/lunarlander/grid_mixture_ms"
GAIL_CSV = "/home/marzii/IRL3/experiments/gail_grid_2026-05-12.csv"
GAIL_COUNT_CSV_NEW = "/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv"
GAIL_NOISE_CSV_NEW = "/home/marzii/IRL3/experiments/gail_grid_noise_2026-06-18.csv"
GAIL_BASE = Path("/scratch/marzii/imitation_runs/gail/lunarlander")
FIG_DIR = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PT_SEC_PER_LABEL = 5.81     # PT human100 measurement
GAIL_SEC_PER_DEMO = 43.2    # IRL3 sess2 measurement


# ── data loaders ─────────────────────────────────────────────────────────────
def load_gail():
    """Load old May-12 GAIL grid (10 seeds, no shuffle) — used for noise axis."""
    rows = defaultdict(list)
    with open(GAIL_CSV) as f:
        for r in csv.DictReader(f):
            v = r.get("eval_reward_mean", "")
            if not v: continue
            try: rows[r["condition_id"]].append(float(v))
            except (ValueError, TypeError): pass
    return rows


def _load_from_csv(csv_path):
    """Generic loader: reads agent_rollouts.npz for each job in a CSV.
    Returns dict of condition_id -> list of mean eval rewards."""
    rows = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            job_id = r["slurm_job_id"]
            cond = r["condition_id"]
            npz = GAIL_BASE / job_id / "eval_data" / "agent_rollouts.npz"
            if not npz.exists():
                continue
            try:
                d = np.load(npz, allow_pickle=True)
                ep_rewards = [float(ep.sum()) for ep in d["rews"]]
                rows[cond].append(float(np.mean(ep_rewards)))
            except Exception:
                pass
    return rows


def load_gail_count_new():
    """Load new 30-seed GAIL count runs (shuffle=1, expert 4615187)."""
    return _load_from_csv(GAIL_COUNT_CSV_NEW)


def load_gail_noise_new():
    """Load new 30-seed GAIL noise runs (all seeds, online noise, expert 4615187) from CSV column."""
    rows = defaultdict(list)
    with open(GAIL_NOISE_CSV_NEW) as f:
        for r in csv.DictReader(f):
            v = r.get("eval_reward_mean", "")
            if not v: continue
            try: rows[r["condition_id"]].append(float(v))
            except (ValueError, TypeError): pass
    return rows


def load_pt():
    rows = defaultdict(list)
    for env_dir in GRID_ROOT.glob("lunarlander-grid-*"):
        cond = env_dir.name
        for f in env_dir.glob("seed_*/eval_summary.json"):
            with open(f) as g: d = json.load(g)
            rows[cond].append(float(d["last10_eval_reward"]))
    return rows


def load_pt_ms():
    """Load multi-seed results from grid_mixture_ms (per-seed noise/count draws)."""
    rows = defaultdict(list)
    for env_dir in GRID_ROOT_MS.glob("lunarlander-grid-ms-*"):
        cond = env_dir.name
        for f in env_dir.glob("seed_*/eval_summary.json"):
            with open(f) as g: d = json.load(g)
            rows[cond].append(float(d["last10_eval_reward"]))
    return rows


def agg(rows: list[float]):
    if not rows: return None, None, 0
    a = np.asarray(rows)
    return float(a.mean()), float(a.std()), len(a)


def load_gail_demo_returns(noise_pcts):
    """Mean cumulative return of the demonstration trajectories at each noise level.
    0% uses the shared clean pool; other levels average across all 30 per-seed pools."""
    DEMO_CLEAN = Path("/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean")
    DEMO_NOISY = Path("/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187")
    result = {}
    for pct in [0] + list(noise_pcts):
        dirs = [DEMO_CLEAN] if pct == 0 else [
            DEMO_NOISY / f"n100_p{pct}_s{seed}" for seed in range(30)
        ]
        ep_returns = []
        for d in dirs:
            arrow = d / "data-00000-of-00001.arrow"
            if not arrow.exists():
                continue
            try:
                with pyarrow.ipc.open_stream(str(arrow)) as r:
                    tbl = r.read_all()
                for ep_rews in tbl.column("rews").to_pylist():
                    ep_returns.append(float(sum(ep_rews)))
            except Exception:
                pass
        if ep_returns:
            result[pct] = float(np.mean(ep_returns))
            print(f"  demo return  noise={pct:>3d}%  n_eps={len(ep_returns)}  mean={result[pct]:+.1f}")
    return result


# ── count axis: PT + GAIL on shared "human time" X ───────────────────────────
def plot_count_combined(pt, pt_ms, gail_count_new, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))

    # GAIL: new 30-seed shuffle runs, condition_id is count_N{N}
    print("GAIL (new 30-seed, shuffle, expert 4615187):")
    gail_Ns = [1, 5, 10, 50, 100]
    series_g = []
    for N in gail_Ns:
        m, s, n = agg(gail_count_new.get(f"count_N{N}", []))
        if m is None: continue
        series_g.append({"N": N, "x": N * GAIL_SEC_PER_DEMO / 60.0,
                          "mean": m, "std": s, "n": n})
        print(f"  N={N:>4d}  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}  "
              f"human={N*GAIL_SEC_PER_DEMO/60:.2f} min")
    xs = [d["x"] for d in series_g]; ys = [d["mean"] for d in series_g]; es = [d["std"] for d in series_g]
    ax.errorbar(xs, ys, yerr=es, fmt="-s", color="#2ca02c", markersize=7,
                capsize=4, linewidth=1.6,
                label=f"GAIL count axis (30 seeds, {GAIL_SEC_PER_DEMO} sec/demo, nonFS expert 4615187)")
    for d in series_g:
        ax.annotate(f"N={d['N']}", (d["x"], d["mean"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    color="#2ca02c")

    # PT: per-seed count ms runs (each seed samples different pairs)
    print("\nPT (30 seeds, per-seed pair pool):")
    pt_Ns = [50, 100, 250, 500, 750, 1000]
    series_p = []
    for N in pt_Ns:
        m, s, n = agg(pt_ms.get(f"lunarlander-grid-ms-count-N{N}", []))
        if m is None: continue
        series_p.append({"N": N, "x": N * PT_SEC_PER_LABEL / 60.0,
                          "mean": m, "std": s, "n": n})
        print(f"  N={N:>4d}  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}  "
              f"human={N*PT_SEC_PER_LABEL/60:.2f} min")
    xs = [d["x"] for d in series_p]; ys = [d["mean"] for d in series_p]; es = [d["std"] for d in series_p]
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color="#1f77b4", markersize=8,
                capsize=4, linewidth=1.8,
                label=f"PT count axis (30 seeds, {PT_SEC_PER_LABEL} sec/label)")
    for d in series_p:
        ax.annotate(f"N={d['N']}", (d["x"], d["mean"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8,
                    color="#1f77b4", fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Human Time (min)")
    ax.set_ylabel("Mean Eval Reward (mean over seeds)")
    ax.set_title("Count axis — PT vs GAIL (30 seeds each, nonFS expert 4615187)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── noise axis: PT + GAIL on shared "noise %" X ──────────────────────────────
def plot_noise_combined(pt, pt_ms, gail_noise_new, gail_count_new, out_path: Path):
    """
    GAIL noise: all 30 seeds from gail_noise_new (noise_p{N}, online noise, expert 4615187).
    0% baseline: count_N100 from gail_count_new (30 seeds, shuffle).
    PT noise:    30-seed ms runs from grid_mixture_ms.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    gail_pcts = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    print("Loading GAIL demo returns from Arrow files...")
    demo_returns = load_gail_demo_returns([p for p in gail_pcts if p > 0])

    data_g = []
    print("GAIL noise (30 seeds, online noise, expert 4615187):")
    for pct in gail_pcts:
        if pct == 0:
            vals = gail_count_new.get("count_N100", [])
        else:
            vals = gail_noise_new.get(f"noise_p{pct}", [])
        m, s, n = agg(vals)
        if m is None: continue
        data_g.append({"pct": pct, "mean": m, "std": s, "n": n})
        print(f"  noise={pct:>3d}%  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}")
    xs = [d["pct"] for d in data_g]; ys = [d["mean"] for d in data_g]; es = [d["std"] for d in data_g]
    ax.errorbar(xs, ys, yerr=es, fmt="-^", color="#9467bd", markersize=7,
                capsize=3, linewidth=1.5,
                label="GAIL noise axis (30 seeds, N=100, nonFS expert 4615187)")
    for d in data_g:
        dr = demo_returns.get(d["pct"])
        if dr is not None:
            ax.annotate(f"demo:{dr:+.0f}", (d["pct"], d["mean"]),
                        textcoords="offset points", xytext=(4, -14),
                        fontsize=7, color="#7b4f9e", style="italic")

    # PT noise axis (30-seed ms runs)
    ms_pcts = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    data_ms = []
    print("\nPT noise (30 seeds):")
    for pct in ms_pcts:
        cond = ("lunarlander-grid-ms-N1000-clean" if pct == 0
                else f"lunarlander-grid-ms-N1000-noise{pct}")
        m, s, n = agg(pt_ms.get(cond, []))
        if m is None: continue
        data_ms.append({"pct": pct, "mean": m, "std": s, "n": n})
        print(f"  noise={pct:>3d}%  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}")
    xs = [d["pct"] for d in data_ms]; ys = [d["mean"] for d in data_ms]; es = [d["std"] for d in data_ms]
    ax.errorbar(xs, ys, yerr=es, fmt="-s", color="#1f77b4", markersize=8,
                capsize=4, linewidth=1.8,
                label="PT label-flip (30 seeds, N=1000, per-seed flipped pairs)")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("Mean Eval Reward (mean over seeds)")
    ax.set_title("Noise axis — PT vs GAIL (30 seeds each, nonFS expert 4615187)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    gail            = load_gail()
    gail_count_new  = load_gail_count_new()
    gail_noise_new  = load_gail_noise_new()
    pt              = load_pt()
    pt_ms           = load_pt_ms()
    print(f"loaded GAIL (old): {len(gail)}, GAIL count (new): {len(gail_count_new)}, "
          f"GAIL noise (new): {len(gail_noise_new)}, PT-ms: {len(pt_ms)} conditions")
    print("\n=== COUNT AXIS — PT vs GAIL (30 seeds each) ===")
    plot_count_combined(pt, pt_ms, gail_count_new, FIG_DIR / "grid_count_pt_vs_gail.png")
    print("\n=== NOISE AXIS — PT vs GAIL (30 seeds each) ===")
    plot_noise_combined(pt, pt_ms, gail_noise_new, gail_count_new,
                        FIG_DIR / "grid_noise_pt_vs_gail.png")
