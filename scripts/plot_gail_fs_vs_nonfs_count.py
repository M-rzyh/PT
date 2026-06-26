"""GAIL FS vs NonFS count axis — 10 seeds each, from gail_grid_2026-05-12.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GAIL_CSV = "/home/marzii/IRL3/experiments/gail_grid_2026-05-12.csv"
FIG_DIR = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

GAIL_SEC_PER_DEMO = 43.2


def load_gail():
    rows = defaultdict(list)
    with open(GAIL_CSV) as f:
        for r in csv.DictReader(f):
            v = r.get("eval_reward_mean", "")
            if not v: continue
            try: rows[r["condition_id"]].append(float(v))
            except (ValueError, TypeError): pass
    return rows


def agg(vals):
    if not vals: return None, None, 0
    a = np.asarray(vals)
    return float(a.mean()), float(a.std()), len(a)


def main():
    gail = load_gail()
    Ns = [1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 500]

    fig, ax = plt.subplots(figsize=(11, 6))

    for prefix, color, marker, label in [
        ("count_N",      "#2ca02c", "s", "GAIL FS   (expert 4720242, frame-skip, 10 seeds)"),
        ("count_nonFS_N", "#9467bd", "^", "GAIL NonFS (expert 4615187, per-frame,  10 seeds)"),
    ]:
        series = []
        print(f"\n[{label}]")
        for N in Ns:
            m, s, n = agg(gail.get(f"{prefix}{N}", []))
            if m is None: continue
            series.append({"N": N, "x": N * GAIL_SEC_PER_DEMO / 60.0,
                           "mean": m, "std": s, "n": n})
            print(f"  N={N:>4d}  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}  "
                  f"human={N*GAIL_SEC_PER_DEMO/60:.1f} min")
        xs = [d["x"] for d in series]
        ys = [d["mean"] for d in series]
        es = [d["std"] for d in series]
        ax.errorbar(xs, ys, yerr=es, fmt=f"-{marker}", color=color,
                    markersize=7, capsize=4, linewidth=1.6, label=label)
        for d in series:
            ax.annotate(f"N={d['N']}", (d["x"], d["mean"]),
                        textcoords="offset points", xytext=(4, 5),
                        fontsize=7, color=color)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Human Time (min)  —  43.2 sec/demo")
    ax.set_ylabel("Mean Eval Reward (mean over 10 seeds)")
    ax.set_title("GAIL count axis — FS vs NonFS expert (10 seeds, no shuffle)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "gail_fs_vs_nonfs_count.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
