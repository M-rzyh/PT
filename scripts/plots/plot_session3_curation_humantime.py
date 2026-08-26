"""GAIL count-vs-human-time plot for the session_3 curation experiment.

x-axis = human time (minutes) = N demos x sec-per-demo, same convention as the PT-vs-GAIL count
plot. Two GAIL series on the SAME axis:
  - HUMAN demos (session_3): N in {1(top-return), 5, 10, 50, 100, 200 random}, this experiment.
  - NON-HUMAN (expert 4615187) demos: borrowed from the PT-vs-GAIL count plot
    (gail_grid_count_2026-06-18.csv), plotted at the human-time-EQUIVALENT (agent demos actually
    cost ~0 human time; here x = what N demos would cost a human at the session_3 rate, so the
    two series are comparable at equal demo count).

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_session3_curation_humantime.py
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
S3_LOG = "/scratch/marzii/imitation_runs/record_human_demos/lunarlander/session_3/all_episodes_log.csv"
CURATION_IDX = glob.glob("/home/marzii/IRL3/experiments/gail_session3_curation_*.csv")[0]
EXPERT_COUNT_CSV = "/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv"
HUM_C, EXP_C = "#d62728", "#2ca02c"


def sec_per_demo():
    rows = list(csv.DictReader(open(S3_LOG)))
    return float(rows[-1]["cumulative_sec"]) / len(rows)   # total wall time / all demos in pool


def ep(j):
    f = f"{GA}/{j}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def human_series():
    per = defaultdict(list)
    for r in csv.DictReader(open(CURATION_IDX)):
        v = ep(int(r["slurm_job_id"]))
        if v is not None:
            per[int(r["N"])].append(v)
    return {N: (np.mean(v), np.std(v)) for N, v in per.items()}


def expert_series():
    per = defaultdict(list)
    for r in csv.DictReader(open(EXPERT_COUNT_CSV)):
        v = r.get("eval_reward_mean", "")
        if v:
            try: per[int(r["N"])].append(float(v))
            except ValueError: pass
    return {N: (np.mean(v), np.std(v)) for N, v in per.items()}


def main():
    spd = sec_per_demo()
    hum, exp = human_series(), expert_series()
    print(f"sec/demo (session_3) = {spd:.1f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for d, c, m, lbl, tag in [(exp, EXP_C, "^", "GAIL — non-human (expert 4615187) demos", "N"),
                              (hum, HUM_C, "o", "GAIL — human demos (session_3)", "N")]:
        Ns = sorted(d)
        x = [N * spd / 60 for N in Ns]              # minutes
        y = [d[N][0] for N in Ns]; e = [d[N][1] for N in Ns]
        ax.errorbar(x, y, yerr=e, fmt=m + "-", color=c, lw=2.2, ms=7, capsize=4, label=lbl, zorder=4)
        for N, xi, yi in zip(Ns, x, y):
            ax.annotate(f"N={N}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=c, fontweight="bold")

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel(f"Human time (minutes) = N demos x {spd:.1f}s "
                  f"(red = actual session_3 rate; green = same rate, estimated for agent demos)")
    ax.set_ylabel("Mean eval reward (true env)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("GAIL on human vs non-human demos — reward vs human time (session_3 curation)\n"
                 "human N=1 is the single highest-return demo; N>=5 are random; expert demos are agent-generated (x = human-time-equivalent)",
                 fontsize=10)
    fig.tight_layout()
    out = f"{FIG}/session3_curation_humantime_gail.png"
    fig.savefig(out, dpi=150); print("Saved:", out)
    print("HUMAN:", {N: round(v[0]) for N, v in hum.items()})
    print("EXPERT:", {N: round(v[0]) for N, v in exp.items()})


if __name__ == "__main__":
    main()
