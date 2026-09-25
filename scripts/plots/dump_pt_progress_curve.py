"""Dump the PT evaluation-progress curve at ONE preference budget as JSON.

New script; plot_pt_learning_curves.py is imported, not modified. Curves are the
per-seed progress.txt rows written by train_offline.py: x = IQL gradient updates
(every 5,000), y = true-environment return over 10 evaluation episodes. PT is
offline, so there are no environment interaction steps.

Per eval point we report the mean over seeds and the
half-width of a 95% CI (t(0.975, n-1) * SD/sqrt(n)), matching the count-axis
figures.

Usage (repo root, pt env):
    python -m scripts.plots.dump_pt_progress_curve --n 250 --out /path/pt_curve.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.plots.plot_pt_learning_curves import RUNS_ROOT, load_curves

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 19: 2.093, 29: 2.045}


def roll(v, k):
    """Centred rolling mean over k points, truncating at the edges (no padding).

    Applied per seed BEFORE averaging, so the band reflects smoothed seeds.
    """
    if k <= 1:
        return v
    h, n = k // 2, v.shape[-1]
    out = np.empty_like(v, dtype=float)
    for i in range(n):
        out[..., i] = np.nanmean(v[..., max(0, i - h):min(n, i + h + 1)], axis=-1)
    return out


def ci95(v):
    """(mean, half-width) per column of v[n_seeds, n_points]."""
    n = v.shape[0]
    sd = v.std(axis=0, ddof=1) if n > 1 else np.zeros(v.shape[1])
    return v.mean(axis=0), sd / np.sqrt(n) * T95.get(n - 1, 1.96)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=250, help="preference budget")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    p.add_argument("--human_pattern", default="lunarlander-human-count-v2-N{n}-policy")
    p.add_argument("--synth_pattern", default="lunarlander-human-count-v2-N{n}-scriptedlabels")
    p.add_argument("--baseline", default="lunarlander-truereward-iql")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--smooth", type=int, default=5,
                   help="centred rolling mean over this many points, per seed "
                        "(1 = none)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    out = {"panel": f"PT (PbRL), N = {a.n}", "xlabel": "IQL gradient updates",
           "ylabel": "return", "series": [], "baseline": None}

    for role, pattern in (("synthetic", a.synth_pattern), ("human", a.human_pattern)):
        steps, vals = load_curves(a.runs_root / pattern.format(n=a.n), a.seeds)
        if steps is None:
            print(f"  (no runs) {role}")
            continue
        m, e = ci95(roll(vals, a.smooth))
        out["series"].append(dict(label=role, x=steps.tolist(), mean=m.tolist(),
                                  err=e.tolist(), n_seeds=int(vals.shape[0])))
        print(f"  {role:<10} seeds={vals.shape[0]} points={len(steps)} final={m[-1]:.1f}")

    if not a.no_baseline:
        steps, vals = load_curves(a.runs_root / a.baseline, a.seeds)
        if steps is not None:
            m, e = ci95(roll(vals, a.smooth))
            out["baseline"] = dict(label="IQL, true reward", x=steps.tolist(),
                                   mean=m.tolist(), err=e.tolist(),
                                   n_seeds=int(vals.shape[0]))
            print(f"  baseline   seeds={vals.shape[0]} final={m[-1]:.1f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
