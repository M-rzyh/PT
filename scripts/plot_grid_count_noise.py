"""PT count + noise grid plots, styled after IRL3's
plot_gail_grid.py {count,noise}_tb_annotated charts.

Reads eval_summary.json files written by each grid task and aggregates
mean ± std over the 10 seeds per condition.

Output:
    figures/pt_grid_count_tb_annotated.png
    figures/pt_grid_noise_tb_annotated.png
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
GRID_ROOT = SCRATCH / "PT/lunarlander/grid_mixture"
FIG_DIR = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Human-time conversion (from PT human100 medium-v2 measurement: 5.81 sec/label).
SEC_PER_LABEL = 5.81


def human_time_min(n_labels: int) -> float:
    return n_labels * SEC_PER_LABEL / 60.0


def gather(condition_id: str, y_key: str = "last10_eval_reward"):
    files = sorted((GRID_ROOT / condition_id).glob("seed_*/eval_summary.json"))
    vals = []
    for f in files:
        with open(f) as g:
            d = json.load(g)
        vals.append(float(d[y_key]))
    if not vals:
        return None, None, 0
    a = np.asarray(vals)
    return float(a.mean()), float(a.std()), len(a)


# ── count axis ──────────────────────────────────────────────────────────────
def plot_count(out_path: Path, y_key: str = "last10_eval_reward"):
    Ns = [50, 100, 500, 1000]
    rows = []
    for N in Ns:
        cond = f"lunarlander-grid-N{N}-clean"
        m, s, n = gather(cond, y_key)
        if m is None:
            print(f"  {cond}: NO DATA")
            continue
        rows.append({"N": N, "mean": m, "std": s, "n": n,
                     "x": human_time_min(N)})
        print(f"  N={N:>4d}  n_seeds={n}  mean={m:+7.2f}  std={s:+6.2f}  "
              f"human={human_time_min(N):.2f} min")

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [d["x"]    for d in rows]
    ys = [d["mean"] for d in rows]
    es = [d["std"]  for d in rows]
    ax.errorbar(xs, ys, yerr=es, fmt="-s", color="#1f77b4", markersize=8,
                capsize=4, linewidth=1.6, label="PT count axis (noise=0%)")
    for d in rows:
        ax.annotate(f"N={d['N']}", (d["x"], d["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel(f"Human Time (min) — assuming {SEC_PER_LABEL} sec/label (PT human100 measurement)")
    ax.set_ylabel("TB last-10% Eval Reward (mean over seeds)")
    ax.set_title("PT count axis (noise=0%) — 10 seeds per point, mean ± std")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── noise axis ──────────────────────────────────────────────────────────────
def plot_noise(out_path: Path, y_key: str = "last10_eval_reward"):
    # 0% comes from the count_N1000-clean condition (same N, just no noise).
    pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rows = []
    for pct in pcts:
        cond = ("lunarlander-grid-N1000-clean" if pct == 0
                else f"lunarlander-grid-N1000-noise{pct}")
        m, s, n = gather(cond, y_key)
        if m is None:
            print(f"  {cond}: NO DATA")
            continue
        rows.append({"pct": pct, "mean": m, "std": s, "n": n})
        print(f"  noise={pct:>3d}%  n_seeds={n}  mean={m:+7.2f}  std={s:+6.2f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [d["pct"]  for d in rows]
    ys = [d["mean"] for d in rows]
    es = [d["std"]  for d in rows]
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color="#d62728", markersize=8,
                capsize=4, linewidth=1.6, label="PT noise axis (N=1000)")
    for d in rows:
        ax.annotate(f"{d['mean']:.0f}", (d["pct"], d["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Label-flip noise (%) — uniform re-sample over {-1, 0, 1}")
    ax.set_ylabel("TB last-10% Eval Reward (mean over seeds)")
    ax.set_title("PT noise axis (N=1000) — 10 seeds per point, mean ± std")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("=== count axis ===")
    plot_count(FIG_DIR / "pt_grid_count_tb_annotated.png")
    print()
    print("=== noise axis ===")
    plot_noise(FIG_DIR / "pt_grid_noise_tb_annotated.png")
