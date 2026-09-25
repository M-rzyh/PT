"""Expert/oracle count axis REPLOTTED with the measured human-time rates, so its x-axis is
directly comparable to the human figure (count_axis_human_pt_vs_gail_retonly.png).

Same runs as figures/grid_count_pt_vs_gail.png (which is NOT replaced):
  GAIL : gail_grid_count_2026-06-18.csv -- 150 jobs (5292359..5292508), expert 4615187
         demos with shuffle, N in {1,5,10,50,100}, 30 seeds, eval = agent_rollouts.npz mean
  PT   : grid_mixture_ms/lunarlander-grid-ms-count-N{50..1000}/seed_0..29, scripted labels,
         eval = last10_eval_reward

Only the time conversion changes -- from the old estimates (43.2 s/demo, 5.81 s/label) to the
MEASURED means of the human pools: 11.431 s/demo (699-demo timing.csv) and 9.074 s/pair
(labels_n1000_clean.pkl). The supervision here is synthetic, so its "human time" is the
hypothetical cost had a human produced it, priced at the measured human rates.

    python scripts/plots/plot_grid_count_pt_vs_gail_humantime.py
"""
import csv, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
GAIL_CSV = "/home/marzii/IRL3/experiments/GAIL/gail_grid_count_2026-06-18.csv"
MS = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
DEMO_SEC, PAIR_SEC = 11.431, 9.074          # measured means; see docstring
GAIL_C, PT_C = "#2ca02c", "#1f77b4"


def main():
    fig, ax = plt.subplots(figsize=(11, 6))

    per = {}
    for r in csv.DictReader(open(GAIL_CSV)):
        f = f"{GAIL_BASE}/{r['slurm_job_id']}/eval_data/agent_rollouts.npz"
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            per.setdefault(int(r["N"]), []).append(np.mean([float(e.sum()) for e in d["rews"]]))
    print("GAIL expert demos (30 seeds, measured-rate axis):")
    xs, ys, es = [], [], []
    for N in [1, 5, 10, 50, 100]:
        v = np.array(per[N]); t = N * DEMO_SEC / 60
        xs.append(t); ys.append(v.mean()); es.append(v.std())
        print(f"  N={N:>4}  n={len(v):2d}  {v.mean():+7.2f} ±{v.std():5.2f}   {t:6.2f} min")
        ax.annotate(f"N={N}", (t, v.mean()), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=GAIL_C)
    ax.errorbar(xs, ys, yerr=es, fmt="-s", color=GAIL_C, ms=7, capsize=4, lw=1.6,
                label=f"GAIL count axis (30 seeds, expert demos, {DEMO_SEC} s/demo measured)")

    print("\nPT scripted labels (30 seeds, measured-rate axis):")
    xs, ys, es = [], [], []
    for N in [50, 100, 250, 500, 750, 1000]:
        fs = glob.glob(f"{MS}/lunarlander-grid-ms-count-N{N}/seed_*/eval_summary.json")
        v = np.array([json.load(open(f))["last10_eval_reward"] for f in fs])
        t = N * PAIR_SEC / 60
        xs.append(t); ys.append(v.mean()); es.append(v.std())
        print(f"  N={N:>4}  n={len(v):2d}  {v.mean():+7.2f} ±{v.std():5.2f}   {t:6.2f} min")
        ax.annotate(f"N={N}", (t, v.mean()), textcoords="offset points", xytext=(6, 8),
                    fontsize=8, color=PT_C, fontweight="bold")
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color=PT_C, ms=8, capsize=4, lw=1.8,
                label=f"PT count axis (30 seeds, scripted labels, {PAIR_SEC} s/pair measured)")

    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Human Time (min)  --  measured rates: 11.43 s/demo, 9.07 s/pair")
    ax.set_ylabel("Mean Eval Reward (mean over seeds)")
    ax.set_title("Count axis -- PT vs GAIL on EXPERT/oracle supervision\n"
                 "(same runs as grid_count_pt_vs_gail; x rescaled to the measured human rates)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "/home/marzii/PT/PreferenceTransformer/figures/grid_count_pt_vs_gail_humantime.png"
    fig.savefig(out, dpi=150); print("\nSaved:", out)


if __name__ == "__main__":
    main()
