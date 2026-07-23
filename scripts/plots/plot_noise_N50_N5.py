"""Small-budget noise plot: PT (N=50 preferences) vs GAIL (N=5 demos), COIN only.
Two panels — random-replace | flip. Same look as the big-budget plot (coin = bold
marked line + band, GAIL demo:… annotations) but no exact (we only ran coin here).
At small N the coin corruption count wobbles, so curves are noisier than N=1000/N=100.

PT reward   = last10_eval_reward (grid_mixture_ms eval_summary, N=50, coin tags).
GAIL reward = 50-ep eval_reward_mean from agent_rollouts.npz (N=5 index CSVs).
Run with the imitation-gail env (needs pyarrow for the demo-return annotations).
"""
import csv
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.ipc

GRID = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C, GA_C = "#1f77b4", "#9467bd"
LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def agg(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def pt_series(noise_word, clean_word):
    out = {}
    for p in LEVELS:
        cond = (f"{GRID}/lunarlander-grid-ms-N50-{clean_word}" if p == 0
                else f"{GRID}/lunarlander-grid-ms-N50-{noise_word}{p}")
        v = [json.load(open(f))["last10_eval_reward"] for f in glob.glob(f"{cond}/seed_*/eval_summary.json")]
        if v:
            out[p] = agg(v)
    return out


def gail_zero_N5():
    vals = []
    for r in csv.DictReader(open("/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv")):
        if r["condition_id"] == "count_N5" and r.get("eval_reward_mean"):
            try:
                vals.append(float(r["eval_reward_mean"]))
            except ValueError:
                pass
    return vals


def gail_from_csv(csv_glob, zero_vals):
    files = sorted(glob.glob(csv_glob))
    if not files:
        return {}
    per = {}
    for r in csv.DictReader(open(files[-1])):
        L = int(r["noise_pct"])
        npz = f"{GAIL_BASE}/{r['slurm_job_id']}/eval_data/agent_rollouts.npz"
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        per.setdefault(L, []).append(float(np.mean([float(e.sum()) for e in d["rews"]])))
    out = {L: agg(v) for L, v in per.items()}
    if zero_vals:
        out[0] = agg(zero_vals)
    return out


# GAIL demonstration returns (pool means) — same pools as the big-budget plot
_NONFS = "/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187"
_CLEAN = "/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"
_CEXCL = "/scratch/marzii/imitation_runs/noise_variants/demos/coin_exclude"


def _pool_return(d):
    r = []
    for a in glob.glob(str(d) + "/*.arrow"):
        try:
            with pyarrow.ipc.open_stream(a) as rd:
                t = rd.read_all()
            for ep in t.column("rews").to_pylist():
                r.append(float(sum(ep)))
        except Exception:
            pass
    return r


def demo_returns(pool_fn):
    out = {}
    for L in LEVELS:
        dirs = [_CLEAN] if L == 0 else pool_fn(L)
        r = []
        for d in dirs:
            r += _pool_return(d)
        if r:
            out[L] = float(np.mean(r))
    return out


def bold_line(ax, d, color, marker, label):
    """coin = bold marked line + band (noise_variants style). Returns (x,y)."""
    if not d:
        return None
    x = np.array(sorted(d))
    y = np.array([d[k][0] for k in x]); e = np.array([d[k][1] for k in x])
    ax.plot(x, y, marker + "-", color=color, lw=2.2, ms=7, label=label, zorder=4)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.13, lw=0, zorder=1)
    return x, y


def annotate(ax, xy, dret):
    if xy is None:
        return
    x, y = xy
    for xi, yi in zip(x, y):
        dr = dret.get(int(xi))
        if dr is not None:
            ax.annotate(f"{dr:+.0f}", (xi, yi), textcoords="offset points",
                        xytext=(5, -15), fontsize=9, color=GA_C, fontweight="bold",
                        ha="left", zorder=8)


def mark_100(ax, d, color):
    """Dashed line to the y-axis at the 100%-noise value, labelled — for cross-N comparison."""
    if 100 not in d:
        return
    yv = d[100][0]
    ax.axhline(yv, color=color, ls="--", lw=1.1, alpha=0.6)
    ax.annotate(f"{yv:+.0f}", (ax.get_xlim()[0], yv), xytext=(3, 2), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color, va="bottom")


def main():
    z = gail_zero_N5()
    pt_rand = pt_series("noise", "clean")
    pt_flip = pt_series("flipnoise", "flipclean")
    g_rand = gail_from_csv("/home/marzii/IRL3/experiments/gail_grid_noise_N5_*.csv", z)
    g_flip = gail_from_csv("/home/marzii/IRL3/experiments/gail_noise_variants_N5_*.csv", z)
    dr_rand = demo_returns(lambda L: [f"{_NONFS}/n100_p{L}_s{s}" for s in range(30)])
    dr_flip = demo_returns(lambda L: [f"{_CEXCL}/n100_p{L}_s{s}" for s in range(30)])
    print("PT N=50 random:", {k: round(v[0]) for k, v in pt_rand.items()})
    print("GAIL N=5 random:", {k: round(v[0]) for k, v in g_rand.items()})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    xy = bold_line(axL, g_rand, GA_C, "^", "GAIL coin (N=5)"); annotate(axL, xy, dr_rand)
    bold_line(axL, pt_rand, PT_C, "s", "PT coin (N=50)")
    axL.set_title("Random-replace noise")
    xy = bold_line(axR, g_flip, GA_C, "^", "GAIL coin (N=5)"); annotate(axR, xy, dr_flip)
    bold_line(axR, pt_flip, PT_C, "s", "PT coin (N=50)")
    axR.set_title("Flip noise")
    for ax in (axL, axR):
        ax.axhline(0, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("Noise (%)"); ax.grid(True, alpha=0.3); ax.set_xlim(-3, 103)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    axL.set_ylabel("Mean eval reward (10 seeds)")
    axL.set_ylim(-700, 320)
    mark_100(axL, pt_rand, PT_C); mark_100(axL, g_rand, GA_C)
    mark_100(axR, pt_flip, PT_C); mark_100(axR, g_flip, GA_C)
    fig.suptitle("Noise injection: PT (N=50) vs GAIL (N=5) — bold = coin (■ PT, ▲ GAIL)  "
                 "(10 seeds, nonFS expert 4615187; demo:… = GAIL demo return)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{FIG}/noise_N50_N5_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
