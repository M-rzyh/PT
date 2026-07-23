"""Noise at the frame-blanking budgets: PT (N=350 preferences) vs GAIL (N=50 demos),
random-replace COIN, 0-100% (11 levels), 30 seeds. Single panel, grid_noise style
(bold marked line + band, GAIL demo:… annotations, 100%-noise dashed lines).
Run with the imitation-gail env (pyarrow for demo returns)."""
import csv, glob, json, os
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
_NONFS = "/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187"
_CLEAN = "/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"


def agg(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def pt_series():
    out = {}
    for p in LEVELS:
        cond = (f"{GRID}/lunarlander-grid-ms-N350-clean" if p == 0
                else f"{GRID}/lunarlander-grid-ms-N350-noise{p}")
        v = [json.load(open(f))["last10_eval_reward"] for f in glob.glob(f"{cond}/seed_*/eval_summary.json")]
        if v:
            out[p] = agg(v)
    return out


def gail_series():
    z = []
    for r in csv.DictReader(open("/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv")):
        if r["condition_id"] == "count_N50" and r.get("eval_reward_mean"):
            try: z.append(float(r["eval_reward_mean"]))
            except ValueError: pass
    per = {}
    csvf = sorted(glob.glob("/home/marzii/IRL3/experiments/gail_grid_noise_N50_*.csv"))[-1]
    for r in csv.DictReader(open(csvf)):
        L = int(r["noise_pct"]); npz = f"{GAIL_BASE}/{r['slurm_job_id']}/eval_data/agent_rollouts.npz"
        if not os.path.exists(npz): continue
        d = np.load(npz, allow_pickle=True)
        per.setdefault(L, []).append(float(np.mean([float(e.sum()) for e in d["rews"]])))
    out = {L: agg(v) for L, v in per.items()}
    if z: out[0] = agg(z)
    return out


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
    print("PT N=350:", {k: round(v[0]) for k, v in pt.items()})
    print("GAIL N=50:", {k: round(v[0]) for k, v in ga.items()})
    fig, ax = plt.subplots(figsize=(11, 6))
    gx, gy = curve(ax, ga, GA_C, "^", "GAIL noise (N=50, 30 seeds, nonFS expert 4615187)")
    curve(ax, pt, PT_C, "s", "PT noise (N=350, 30 seeds, per-seed corrupted pairs)")
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
    ax.set_title("Noise axis — PT (N=350) vs GAIL (N=50), coin random-replace (30 seeds, nonFS expert 4615187)")
    fig.tight_layout()
    out = f"{FIG}/noise_N350_N50_pt_vs_gail.png"
    fig.savefig(out, dpi=150); print("Saved:", out)


if __name__ == "__main__":
    main()
