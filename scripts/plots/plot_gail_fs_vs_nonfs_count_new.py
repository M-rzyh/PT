"""GAIL FS vs NonFS count axis.
FS old:  eval_run_list.txt (expert 4720242, 10 seeds, no shuffle)
FS new:  gail_grid_count_FS_2026-06-18.csv (expert 4720242, 30 seeds, shuffle)
NonFS:   gail_grid_count_2026-06-18.csv (expert 4615187, 30 seeds, shuffle)
X-axis: human time (43.2 sec/demo), Y-axis: mean eval reward.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_RUN_LIST      = "/home/marzii/IRL3/experiments/eval_run_list.txt"
GAIL_CSV_COUNT     = "/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv"
GAIL_CSV_COUNT_FS  = "/home/marzii/IRL3/experiments/gail_grid_count_FS_2026-06-18.csv"
GAIL_BASE          = Path("/scratch/marzii/imitation_runs/gail/lunarlander")
FIG_DIR            = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEC_PER_DEMO = 43.2

FS_OLD_Ns = [1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150]
FS_NEW_Ns = [1, 5, 10, 50, 100]
NONFS_Ns  = [1, 5, 10, 50, 100]


def agg(vals):
    if not vals: return None, None, 0
    a = np.asarray(vals)
    return float(a.mean()), float(a.std()), len(a)


def load_fs_count():
    """Scrape FS count runs from eval_run_list.txt (expert_4720242, clean demos, no shuffle)."""
    rows = defaultdict(list)
    with open(EVAL_RUN_LIST) as f:
        job_dirs = [Path(l.strip()) for l in f if l.strip()]

    for jdir in job_dirs:
        meta_path = jdir / "eval_data" / "meta.json"
        npz_path  = jdir / "eval_data" / "agent_rollouts.npz"
        if not meta_path.exists() or not npz_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "expert_4720242" not in meta.get("demo_path", ""):
            continue
        if "p0_clean" not in meta.get("demo_path", ""):
            continue
        N = meta.get("n_demos")
        if N not in FS_OLD_Ns:
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
            mean_r = float(np.mean([ep.sum() for ep in d["rews"]]))
            rows[f"count_N{N}"].append(mean_r)
        except Exception:
            pass

    return rows


def load_from_csv(csv_path, cond_key=None):
    """Load eval_reward_mean from scraped CSV. cond_key overrides condition_id if set."""
    rows = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            v = r.get("eval_reward_mean", "")
            if not v: continue
            try:
                key = cond_key(r) if cond_key else r["condition_id"]
                rows[key].append(float(v))
            except (ValueError, TypeError):
                pass
    return rows


def load_nonfs_count():
    """Load NonFS 30-seed count runs (shuffle=1, expert 4615187)."""
    return load_from_csv(GAIL_CSV_COUNT)


def load_fs_new_count():
    """Load new FS 30-seed count runs (shuffle=1, expert 4720242)."""
    # condition_id is count_FS_N{N} — remap to count_N{N} for uniform lookup
    rows = defaultdict(list)
    with open(GAIL_CSV_COUNT_FS) as f:
        for r in csv.DictReader(f):
            v = r.get("eval_reward_mean", "")
            if not v: continue
            try:
                # strip the _FS_ prefix for uniform key
                key = r["condition_id"].replace("count_FS_", "count_")
                rows[key].append(float(v))
            except (ValueError, TypeError):
                pass
    return rows


def main():
    fs_old  = load_fs_count()
    fs_new  = load_fs_new_count()
    nonfs   = load_nonfs_count()

    fig, ax = plt.subplots(figsize=(13, 6))

    configs = [
        (fs_new, FS_NEW_Ns, "#2ca02c", "D", "GAIL FS  (expert 4720242, 30 seeds, shuffle)"),
        (nonfs,  NONFS_Ns,  "#9467bd", "^", "GAIL NonFS (expert 4615187, 30 seeds, shuffle)"),
    ]

    for data, Ns, color, marker, label_str in configs:
        series = []
        print(f"\n[{label_str}]")
        for N in Ns:
            m, s, n = agg(data.get(f"count_N{N}", []))
            if m is None:
                continue
            x = N * SEC_PER_DEMO / 60.0
            series.append({"N": N, "x": x, "mean": m, "std": s, "n": n})
            print(f"  N={N:>4d}  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}  "
                  f"human={x:.1f} min")
        xs = [d["x"] for d in series]
        ys = [d["mean"] for d in series]
        es = [d["std"] for d in series]
        ax.errorbar(xs, ys, yerr=es, fmt=f"-{marker}", color=color,
                    markersize=7, capsize=4, linewidth=1.6, label=label_str)
        for d in series:
            ax.annotate(f"N={d['N']}", (d["x"], d["mean"]),
                        textcoords="offset points", xytext=(4, 5),
                        fontsize=8, color=color)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Human Time (min)  —  43.2 sec/demo")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("GAIL count axis — FS vs NonFS expert")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "gail_fs_vs_nonfs_count_new.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
