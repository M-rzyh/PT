"""PT human count-axis learning curves (fixed labels): N = 10, 50, 100, 750, 1000.

x = IQL GRADIENT steps (PT is offline -- it never interacts with the env during training,
    unlike the GAIL/PPO curves whose x is env steps). 200 evals, every 5000 steps.
y = eval return, mean over 10 seeds; band = +/- 1 std across seeds.

N is an ordered quantity, so the arms use ONE hue light->dark (a sequential ramp) rather
than unordered categorical colours -- the reading order is the point.

    python scripts/plots/plot_pt_human_count_curves.py
"""
import argparse, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
NS_DEFAULT = [10, 50, 100, 750, 1000]
# ColorBrewer Blues, 5 steps, monotone lightness: light = few preferences, dark = many
COLORS = ["#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]


def curves(cond, N, seeds=range(10)):
    out = []
    for s in seeds:
        p = glob.glob(f"{B}/{cond.format(N=N)}/seed_{s}/tb/**/progress.txt", recursive=True)
        if not p:
            continue
        d = np.loadtxt(p[0])
        if d.ndim == 2 and len(d) >= 50:
            out.append(d)
    return out


def smooth(y, w):
    return (y, 0) if w <= 1 or len(y) < w else (np.convolve(y, np.ones(w) / w, mode="valid"), (w - 1) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marzii/IRL3/figures/pt_human_count_curves.png")
    ap.add_argument("--window", type=int, default=5, help="rolling mean in eval points (1 = raw)")
    ap.add_argument("--cond", default="lunarlander-human-count-N{N}-fixedlabels")
    ap.add_argument("--ns", default="10,50,100,750,1000")
    ap.add_argument("--label", default="preferences")
    ap.add_argument("--title", default="PT on human preferences — count axis with corrected labels")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12, 6.5))
    print(f"{'N':>6} {'seeds':>6} {'peak':>8} {'final(last10)':>14} {'collapsed':>10}")
    NS = [int(v) for v in a.ns.split(",")]
    cols = COLORS if len(NS) >= 5 else COLORS[1:1+len(NS)]
    for N, c in zip(NS, cols):
        cs = curves(a.cond, N)
        if not cs:
            print(f"{N:>6}  no runs"); continue
        n = min(len(d) for d in cs)
        x = cs[0][:n, 0]
        Y = np.vstack([smooth(d[:n, 1], a.window)[0] for d in cs])
        off = smooth(cs[0][:n, 1], a.window)[1]
        xs = x[off:off + Y.shape[1]]
        m, sd = Y.mean(0), Y.std(0)
        ax.fill_between(xs, m - sd, m + sd, color=c, alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, m, color=c, lw=2.2, zorder=3, label=f"N = {N} {a.label}  ({len(cs)} seeds)")
        ax.scatter([xs[-1]], [m[-1]], color=c, s=30, zorder=4, edgecolor="white", linewidth=0.8)
        raw = np.vstack([d[:n, 1] for d in cs])
        coll = sum(1 for r in raw if r.max() - r[-10:].mean() > 100)
        print(f"{N:>6} {len(cs):>6} {raw.max(1).mean():8.1f} {raw[:, -10:].mean():14.1f} {str(coll)+'/'+str(len(cs)):>10}")

    ax.axhline(0, color="gray", ls=":", alpha=0.5, zorder=1)
    ax.axhline(200, color="gray", ls=":", alpha=0.3, zorder=1)
    ax.set_xlabel("IQL gradient steps  (offline — no environment interaction during training)")
    ax.set_ylabel(f"Eval return, true env  (rolling {a.window} evals = {a.window*5}k steps)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    ax.set_title(a.title + "\nmean over 10 seeds; band = ±1 std", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
