"""PT noise axis: deterministic-flip vs random-replace corruption.

Both series are PT (N=1000, per-seed corrupted pairs, IQL on the mixture).
The only difference is how a corrupted preference label is generated:
  random_replace   : label -> uniform draw from {-1,0,1}  (alignment 1-(2/3)p)
  deterministic_flip: label -> inverted 0<->1             (alignment 1-p)

Reads per-seed eval_summary.json under grid_mixture_ms/<COND_ID>/seed_*/.
"""
from __future__ import annotations

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
FIG_DIR = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PCTS = [0, 20, 40, 60, 80, 100]


def load_pt_ms():
    rows = defaultdict(list)
    for env_dir in GRID_ROOT_MS.glob("lunarlander-grid-ms-*"):
        for f in env_dir.glob("seed_*/eval_summary.json"):
            with open(f) as g:
                d = json.load(g)
            rows[env_dir.name].append(float(d["last10_eval_reward"]))
    return rows


def agg(vals):
    if not vals:
        return None, None, 0
    a = np.asarray(vals)
    return float(a.mean()), float(a.std()), len(a)


def series(pt_ms, noise_word, clean_word):
    out = []
    for pct in PCTS:
        cond = (f"lunarlander-grid-ms-N1000-{clean_word}" if pct == 0
                else f"lunarlander-grid-ms-N1000-{noise_word}{pct}")
        m, s, n = agg(pt_ms.get(cond, []))
        if m is None:
            print(f"  [missing] {cond}")
            continue
        out.append({"pct": pct, "mean": m, "std": s, "n": n})
        print(f"  noise={pct:>3d}%  n_seeds={n}  mean={m:+7.2f}  ±{s:5.2f}  ({cond})")
    return out


def main():
    pt_ms = load_pt_ms()
    fig, ax = plt.subplots(figsize=(11, 6))

    print("PT random_replace:")
    rr = series(pt_ms, "noise", "clean")
    print("PT deterministic_flip:")
    fl = series(pt_ms, "flipnoise", "flipclean")

    for data, color, fmt, lbl in [
        (rr, "#1f77b4", "-s", "PT random-replace (uniform {-1,0,1})"),
        (fl, "#d62728", "-o", "PT deterministic-flip (invert 0<->1)"),
    ]:
        if not data:
            continue
        xs = [d["pct"] for d in data]
        ys = [d["mean"] for d in data]
        es = [d["std"] for d in data]
        ax.errorbar(xs, ys, yerr=es, fmt=fmt, color=color, markersize=8,
                    capsize=4, linewidth=1.8, label=lbl)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("Mean Eval Reward (mean over seeds)")
    n_rr = rr[0]["n"] if rr else 0
    n_fl = fl[0]["n"] if fl else 0
    ax.set_title(f"PT noise axis — random-replace vs deterministic-flip "
                 f"(N=1000; rr n≈{n_rr}, flip n≈{n_fl})")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "grid_noise_pt_flip_vs_random.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
