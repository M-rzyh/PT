"""PT learning curves: true-environment evaluation return vs IQL TRAINING steps.

Two panels, one row: left = human labels, right = scripted/synthetic labels.
One curve per preference budget N, mean over seeds with a +/- band.

WHAT THE AXES ARE (verified in code, not assumed):
  x  IQL GRADIENT/TRAINING steps. train_offline.py:220 appends (i, return) where
     i is the training-loop counter, and eval_interval=5000 over max_steps=1e6
     gives 200 rows per run. PT is OFFLINE RL: the policy never interacts with
     the environment during training, so there are no environment interaction
     steps to plot.
  y  TRUE environment return. evaluation.py:9-25 rolls the policy in the real
     gym env (temperature=0, deterministic) and reads info['episode']['return']
     from the EpisodeMonitor wrapper -- the environment's own reward, NOT the
     PT-predicted reward that IQL trains on. 10 episodes per point.

Run from the repo root with the pt env:
    python -m scripts.plots.plot_pt_learning_curves
    python -m scripts.plots.plot_pt_learning_curves --smooth 5 --err sd
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_ROOT = Path("/scratch/marzii/PT/lunarlander/grid_mixture_ms")
OUT_DEFAULT = Path("/scratch/marzii/PT/figures/pt_learning_curves_human_vs_synthetic.png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 50, 100, 250, 750, 1000])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    p.add_argument("--human_pattern", default="lunarlander-human-count-v2-N{n}-policy")
    p.add_argument("--synth_pattern", default="lunarlander-human-count-v2-N{n}-scriptedlabels")
    p.add_argument("--baseline", default="lunarlander-truereward-iql",
                   help="ground-truth-reward IQL runs; drawn in both panels")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--err", choices=["se", "sd"], default="se")
    p.add_argument("--smooth", type=int, default=3,
                   help="centred rolling mean over this many eval points "
                        "(1 = none; each point is 5k updates apart)")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    return p.parse_args()


def load_curves(cond_dir: Path, seeds: int):
    """(steps, values[n_seeds, n_points]) for one condition, or (None, None)."""
    curves, steps_ref = [], None
    for s in range(seeds):
        seed_dir = cond_dir / f"seed_{s}"
        hits = sorted(seed_dir.rglob("progress.txt"))
        if not hits:
            continue
        data = np.loadtxt(hits[-1])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        steps, vals = data[:, 0], data[:, 1]
        if steps_ref is None:
            steps_ref = steps
        m = min(len(steps_ref), len(vals))
        steps_ref, vals = steps_ref[:m], vals[:m]
        curves = [c[:m] for c in curves]
        curves.append(vals)
    if not curves:
        return None, None
    return steps_ref, np.vstack(curves)


def smooth(y: np.ndarray, k: int):
    """Centred rolling mean that TRUNCATES at the edges.

    np.convolve(..., mode='same') zero-pads instead, which pulls the first and
    last k//2 points toward 0 and corrupts exactly the end-of-training values
    we care about. This averages over whatever window is available.
    """
    if k <= 1:
        return y
    h = k // 2
    out = np.empty_like(y, dtype=float)
    n = y.shape[-1]
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[..., i] = y[..., lo:hi].mean(axis=-1)
    return out


def main():
    a = parse_args()
    # standard categorical palette (matplotlib tab10), not a continuous colormap
    palette = plt.get_cmap("tab10").colors
    colors = {n: palette[i % len(palette)] for i, n in enumerate(a.budgets)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True, sharex=True)
    panels = [("human preferences", a.human_pattern, axes[0]),
              ("synthetic preferences", a.synth_pattern, axes[1])]

    base_steps = base_vals = None
    if not a.no_baseline:
        base_steps, base_vals = load_curves(a.runs_root / a.baseline, a.seeds)

    for title, pattern, ax in panels:
        print(f"--- {title}")
        for n in a.budgets:
            cond = a.runs_root / pattern.format(n=n)
            steps, vals = load_curves(cond, a.seeds)
            if steps is None:
                print(f"  (no runs) {cond.name}")
                continue
            vals = smooth(vals, a.smooth)
            m = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if vals.shape[0] > 1 else np.zeros_like(m)
            e = sd / np.sqrt(vals.shape[0]) if a.err == "se" else sd
            ax.plot(steps, m, color=colors[n], lw=1.8, label=f"N={n}")
            ax.fill_between(steps, m - e, m + e, color=colors[n], alpha=0.18, lw=0)
            print(f"  N={n:<5} seeds={vals.shape[0]:<3} final={m[-1]:7.1f}")

        if base_vals is not None:
            bm = smooth(base_vals, a.smooth).mean(axis=0)
            ax.plot(base_steps, bm, color="0.35", ls="--", lw=1.6,
                    label=f"ground-truth reward (n={base_vals.shape[0]})")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("IQL updates")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)

    axes[0].set_ylabel("true environment return")
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
