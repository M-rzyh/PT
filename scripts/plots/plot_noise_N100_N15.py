"""Noise axis at the LOW budgets: PT (N=100 preferences) vs GAIL (N=15 demos),
random-replace COIN, 0-100% (11 levels), 30 seeds. Same style as the N=350/N=50 and
N=1000/N=100 plots. Run with the imitation-gail env (pyarrow for demo returns).

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_noise_N100_N15.py

DATA SOURCES (deliberately explicit, not globbed):
  PT   : grid_mixture_ms/lunarlander-grid-ms-N100-{clean,noise{p}}/seed_*/eval_summary.json
  GAIL noise 10-100% : the REBUILT index gail_grid_noise_N15_2026-07-23.csv (300 rows, one
         per (level,seed), eval-verified). Do NOT glob gail_grid_noise_N15_*.csv — the older
         dated files are partial submit logs (14 and 234 rows) and would double-count.
  GAIL 0%            : count_N15 in gail_grid_count_2026-07-22.csv  (the clean point; see the
         0%-convention note in IRL3/scripts/README_bash_files.md — 0% == count_N{N}).
"""
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.ipc

GRID = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
EXP = "/home/marzii/IRL3/experiments"
NOISE_IDX = f"{EXP}/gail_grid_noise_N15_2026-07-23.csv"   # rebuilt, authoritative (300 rows)
COUNT_IDX = f"{EXP}/gail_grid_count_2026-07-22.csv"       # holds count_N15 (the 0% point)
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
        cond = (f"{GRID}/lunarlander-grid-ms-N100-clean" if p == 0
                else f"{GRID}/lunarlander-grid-ms-N100-noise{p}")
        v = [json.load(open(f))["last10_eval_reward"]
             for f in glob.glob(f"{cond}/seed_*/eval_summary.json")]
        if v:
            out[p] = agg(v)
            print(f"  PT   {p:>3}%: n={len(v)}  mean {out[p][0]:+.0f}")
    return out


def _ep_return_mean(jobid):
    npz = f"{GAIL_BASE}/{jobid}/eval_data/agent_rollouts.npz"
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def gail_series():
    """0% from count_N15; 10-100% from the rebuilt noise index (one row per level,seed)."""
    per = {}
    for r in csv.DictReader(open(COUNT_IDX)):
        if r.get("condition_id") == "count_N15":
            v = _ep_return_mean(r["slurm_job_id"])
            if v is not None:
                per.setdefault(0, []).append(v)
    for r in csv.DictReader(open(NOISE_IDX)):
        v = _ep_return_mean(r["slurm_job_id"])
        if v is not None:
            per.setdefault(int(r["noise_pct"]), []).append(v)
    out = {}
    for p in LEVELS:
        if per.get(p):
            out[p] = agg(per[p])
            print(f"  GAIL {p:>3}%: n={len(per[p])}  mean {out[p][0]:+.0f}")
    return out


def demo_returns():
    """Mean return of the demos GAIL trained on, for the annotations."""
    out = {}
    for L in LEVELS:
        dirs = [_CLEAN] if L == 0 else [f"{_NONFS}/n100_p{L}_s{s}" for s in range(30)]
        r = []
        for d in dirs:
            for a in glob.glob(d + "/*.arrow"):
                try:
                    with pyarrow.ipc.open_stream(a) as rd:
                        t = rd.read_all()
                    r += [float(sum(ep)) for ep in t.column("rews").to_pylist()]
                except Exception:
                    pass
        if r:
            out[L] = float(np.mean(r))
    return out


def curve(ax, d, color, marker, label):
    x = np.array(sorted(d)); y = np.array([d[k][0] for k in x]); e = np.array([d[k][1] for k in x])
    ax.plot(x, y, marker + "-", color=color, lw=2.2, ms=7, label=label, zorder=4)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.13, lw=0, zorder=1)
    return x, y


def main():
    print("Loading:")
    pt, ga, dr = pt_series(), gail_series(), demo_returns()
    print("\nPT   N=100:", {k: round(v[0]) for k, v in pt.items()})
    print("GAIL N=15 :", {k: round(v[0]) for k, v in ga.items()})

    fig, ax = plt.subplots(figsize=(11, 6))
    gx, gy = curve(ax, ga, GA_C, "^", "GAIL noise (N=15 demos, 30 seeds, nonFS expert 4615187)")
    curve(ax, pt, PT_C, "s", "PT noise (N=100 preferences, 30 seeds, per-seed corrupted pairs)")
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
            ax.annotate(f"{yv:+.0f}", (ax.get_xlim()[0], yv), xytext=(3, 2),
                        textcoords="offset points", fontsize=9, fontweight="bold", color=c, va="bottom")
    ax.set_title("Noise axis — PT (N=100) vs GAIL (N=15), coin random-replace "
                 "(30 seeds, nonFS expert 4615187)")
    fig.tight_layout()
    out = f"{FIG}/noise_N100_N15_pt_vs_gail.png"
    fig.savefig(out, dpi=150); print("\nSaved:", out)


if __name__ == "__main__":
    main()
