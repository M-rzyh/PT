"""Low-budget / matched-human-time noise axis: GAIL N=5 vs PT N=50.

Human-time at this operating point is ~comparable:
  GAIL 5 demos  * 43.2 s/demo  = 3.6 min
  PT   50 labels * 5.81 s/label = 4.8 min

Three series (x = noise %):
  - GAIL N=5            (nonFS expert 4615187; 0% reuses count_N5)
  - PT N=50 random_replace (uniform {-1,0,1};   0% reuses count-N50)
  - PT N=50 deterministic_flip (invert 0<->1;   0% reuses count-N50)

GAIL eval is read from agent_rollouts.npz per job (like plot_grid_pt_vs_gail.py).
PT eval is read from grid_mixture_ms eval_summary.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
GRID_ROOT_MS = SCRATCH / "PT/lunarlander/grid_mixture_ms"
GAIL_BASE = SCRATCH / "imitation_runs/gail/lunarlander"
GAIL_COUNT_CSV = "/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv"
FIG_DIR = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def agg(vals):
    if not vals:
        return None, None, 0
    a = np.asarray(vals)
    return float(a.mean()), float(a.std()), len(a)


# ── PT loaders ───────────────────────────────────────────────────────────────
def load_pt_ms():
    rows = defaultdict(list)
    for env_dir in GRID_ROOT_MS.glob("lunarlander-grid-ms-*"):
        for f in env_dir.glob("seed_*/eval_summary.json"):
            with open(f) as g:
                rows[env_dir.name].append(float(json.load(g)["last10_eval_reward"]))
    return rows


def pt_series(pt_ms, noise_word):
    """noise_word in {'noise','flipnoise'}; 0% reuses count-N50."""
    out = []
    for pct in LEVELS:
        cond = ("lunarlander-grid-ms-count-N50" if pct == 0
                else f"lunarlander-grid-ms-N50-{noise_word}{pct}")
        m, s, n = agg(pt_ms.get(cond, []))
        if m is None:
            continue
        out.append({"pct": pct, "mean": m, "std": s, "n": n})
    return out


# ── GAIL loaders ─────────────────────────────────────────────────────────────
def _npz_mean(job_id):
    npz = GAIL_BASE / job_id / "eval_data" / "agent_rollouts.npz"
    if not npz.exists():
        return None
    try:
        d = np.load(npz, allow_pickle=True)
        return float(np.mean([float(ep.sum()) for ep in d["rews"]]))
    except Exception:
        return None


def _load_gail_csv(csv_path, cond_filter=None):
    """cond -> list of mean eval rewards (from npz; fallback eval_reward_mean)."""
    rows = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            cond = r["condition_id"]
            if cond_filter and cond not in cond_filter:
                continue
            v = _npz_mean(r.get("slurm_job_id", ""))
            if v is None:
                try:
                    v = float(r.get("eval_reward_mean", ""))
                except (ValueError, TypeError):
                    continue
            rows[cond].append(v)
    return rows


def gail_series():
    out = []
    # 0% from count_N5
    count = _load_gail_csv(GAIL_COUNT_CSV, cond_filter={"count_N5"})
    m, s, n = agg(count.get("count_N5", []))
    if m is not None:
        out.append({"pct": 0, "mean": m, "std": s, "n": n})
    # 10..100 from the newest N5 noise index CSV
    n5_csvs = sorted(glob.glob("/home/marzii/IRL3/experiments/gail_grid_noise_N5_*.csv"))
    if n5_csvs:
        noise = _load_gail_csv(n5_csvs[-1])
        for pct in LEVELS:
            if pct == 0:
                continue
            m, s, n = agg(noise.get(f"noise_N5_p{pct}", []))
            if m is None:
                continue
            out.append({"pct": pct, "mean": m, "std": s, "n": n})
    else:
        print("  [GAIL] no gail_grid_noise_N5_*.csv yet — only 0% point shown")
    return sorted(out, key=lambda d: d["pct"])


def draw(ax, data, color, fmt, label):
    if not data:
        return
    xs = [d["pct"] for d in data]
    ys = [d["mean"] for d in data]
    es = [d["std"] for d in data]
    ax.errorbar(xs, ys, yerr=es, fmt=fmt, color=color, markersize=8,
                capsize=4, linewidth=1.8, label=label)


def main():
    pt_ms = load_pt_ms()
    gail = gail_series()
    rr = pt_series(pt_ms, "noise")
    fl = pt_series(pt_ms, "flipnoise")

    for name, d in [("GAIL N=5", gail), ("PT N=50 random", rr), ("PT N=50 flip", fl)]:
        print(f"{name}: " + ", ".join(f"{x['pct']}%={x['mean']:+.0f}(n{x['n']})" for x in d))

    fig, ax = plt.subplots(figsize=(11, 6))
    draw(ax, gail, "#9467bd", "-^", "GAIL N=5 (nonFS expert 4615187)")
    draw(ax, rr, "#1f77b4", "-s", "PT N=50 random-replace (uniform {-1,0,1})")
    draw(ax, fl, "#d62728", "-o", "PT N=50 deterministic-flip (invert 0<->1)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("Mean Eval Reward (mean over seeds)")
    ax.set_title("Low-budget noise axis — GAIL N=5 vs PT N=50 "
                 "(~4 min human time, 30 seeds)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "grid_noise_lowbudget_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
