"""Noise at the FULL budgets: PT (N=750 preferences) vs GAIL (N=100 demos),
random-replace COIN, 0-100% (11 levels), 30 seeds. Single panel, same style as
plot_noise_N350_N50.py (bold marked line + band, GAIL demo:… annotations, 100%-noise
dashed lines). Run with the imitation-gail env (pyarrow for demo returns).

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_noise_N750_N100.py
"""
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.ipc

GRID = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C, GA_C = "#1f77b4", "#9467bd"
LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_NONFS = "/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187"
_CLEAN = "/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"


def agg(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def pt_series():
    out = {}
    for p in LEVELS:
        cond = (f"{GRID}/lunarlander-grid-ms-count-N750" if p == 0   # 0% = count-N750 (no N750-clean)
                else f"{GRID}/lunarlander-grid-ms-N750-noise{p}")
        v = [json.load(open(f))["last10_eval_reward"] for f in glob.glob(f"{cond}/seed_*/eval_summary.json")]
        if v:
            out[p] = agg(v)
    return out


def gail_series():
    """0% from count_N100; 10-100% from the N=100 noise csv's eval_reward_mean."""
    per = {}
    for r in csv.DictReader(open(f"{EXP}/gail_grid_count_2026-06-18.csv")):
        if r["condition_id"] == "count_N100" and r.get("eval_reward_mean"):
            try: per.setdefault(0, []).append(float(r["eval_reward_mean"]))
            except ValueError: pass
    for r in csv.DictReader(open(f"{EXP}/gail_grid_noise_2026-06-18.csv")):
        if r.get("N") != "100" or not r.get("eval_reward_mean"):
            continue
        L = int(r["noise_pct"])
        if L not in LEVELS:            # skip the extra 25/75 levels not on this axis
            continue
        try: per.setdefault(L, []).append(float(r["eval_reward_mean"]))
        except ValueError: pass
    return {L: agg(v) for L, v in per.items() if v}


def demo_returns():
    out = {}
    for L in LEVELS:
        dirs = [_CLEAN] if L == 0 else [f"{_NONFS}/n100_p{L}_s{s}" for s in range(30)]
        r = []
        for d in dirs:
            for a in glob.glob(d + "/*.arrow"):
                try:
                    with pyarrow.ipc.open_stream(a) as rd:
                        t = rd.read_all()
                    for ep in t.column("rews").to_pylist(): r.append(float(sum(ep)))
                except Exception: pass
        if r: out[L] = float(np.mean(r))
    return out


def curve(ax, d, color, marker, label):
    x = np.array(sorted(d)); y = np.array([d[k][0] for k in x]); e = np.array([d[k][1] for k in x])
    ax.plot(x, y, marker + "-", color=color, lw=2.2, ms=7, label=label, zorder=4)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.13, lw=0, zorder=1)
    return x, y


def main():
    pt = pt_series(); ga = gail_series(); dr = demo_returns()
    print("PT N=750:", {k: round(v[0]) for k, v in pt.items()})
    print("GAIL N=100:", {k: round(v[0]) for k, v in ga.items()})
    fig, ax = plt.subplots(figsize=(11, 6))
    gx, gy = curve(ax, ga, GA_C, "^", "GAIL noise (N=100, 30 seeds, nonFS expert 4615187)")
    curve(ax, pt, PT_C, "s", "PT noise (N=750, 30 seeds, per-seed corrupted pairs)")
    for xi, yi in zip(gx, gy):
        v = dr.get(int(xi))
        if v is not None:
            ax.annotate(f"{v:+.0f}", (xi, yi), textcoords="offset points", xytext=(5, -15),
                        fontsize=9, color=GA_C, fontweight="bold", zorder=8)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Noise (%)"); ax.set_ylabel("Mean eval reward (30 seeds)")
    ax.grid(True, alpha=0.3); ax.set_xlim(-3, 103); ax.legend(loc="lower left", fontsize=9)
    for d, c in [(pt, PT_C), (ga, GA_C)]:
        if 100 in d:
            yv = d[100][0]; ax.axhline(yv, color=c, ls="--", lw=1.1, alpha=0.6)
            ax.annotate(f"{yv:+.0f}", (ax.get_xlim()[0], yv), xytext=(3, 2), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=c, va="bottom")
    ax.set_title("Noise axis — PT (N=750) vs GAIL (N=100), coin random-replace (30 seeds, nonFS expert 4615187)")
    fig.tight_layout()
    out = f"{FIG}/noise_N750_N100_pt_vs_gail.png"
    fig.savefig(out, dpi=150); print("Saved:", out)


if __name__ == "__main__":
    main()
