"""Count/human-time axis — ALL methods on the 699 human-demo pool + PT preferences.
Same x-axis convention as the oracle count plot: x = human time (minutes).

  PT              : human preferences, N in {10,100,250,750,1000}, 30 seeds  (9.06 s/pref)
  GAIL 9-feature  : top-K of 699 by 9-feature ranking, 5 seeds              (10.8 s/demo)
  GAIL 2-feature  : top-K by return + action_divergence, 5 seeds
  AIRL 9-feature  : top-K by 9-feature ranking, 5 seeds
  BC-warmstart    : BC(top-K) -> PPO on true reward, K in {10,50,100,300,699}, 5 seeds

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_human_count_all_ext700.py
"""
import csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
BW = "/scratch/marzii/imitation_runs/bc_warmstart/lunarlander"
PT_MS = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"

PT_SEC_PER_PREF = 151 * 60 / 1000    # 9.06 s/preference
DEMO_SEC = 54 * 60 / 300             # 10.8 s/demo

PT_C, G9_C, G2_C, AIRL_C, BC_C = "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#e6820e"


def epret(j):
    f = f"{GA}/{j}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f): return None
    d = np.load(f, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def gail_curve(idx):
    per = defaultdict(list)
    for r in csv.DictReader(open(idx)):
        v = epret(r["slurm_job_id"].strip())
        if v is not None: per[int(r["K"])].append(v)
    return {K: (np.mean(v), np.std(v)) for K, v in sorted(per.items())}


def bc_curve(idx):
    per = defaultdict(list)
    for r in csv.DictReader(open(idx)):
        f = f"{BW}/{r['jobid'].strip()}/eval_data/meta.json"
        if os.path.exists(f): per[int(r["K"])].append(json.load(open(f))["ep_reward_mean"])
    return {K: (np.mean(v), np.std(v)) for K, v in sorted(per.items())}


def pt_curve(Ns):
    out = {}
    for N in Ns:
        v = [json.load(open(f))["last10_eval_reward"]
             for f in __import__("glob").glob(f"{PT_MS}/lunarlander-human-count-N{N}/seed_*/eval_summary.json")]
        if v: out[N] = (np.mean(v), np.std(v))
    return out


def draw(ax, d, sec, color, marker, lbl):
    xs = np.array([k * sec / 60 for k in sorted(d)])
    y = np.array([d[k][0] for k in sorted(d)]); e = np.array([d[k][1] for k in sorted(d)])
    ax.fill_between(xs, y - e, y + e, color=color, alpha=0.11, lw=0, zorder=1)
    ax.plot(xs, y, marker + "-", color=color, lw=2.2, ms=7, label=lbl, zorder=4)
    for k, xi, yi in zip(sorted(d), xs, y):
        ax.annotate(f"{k}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6.5, color=color)


def main():
    pt = pt_curve([10, 100, 250, 750, 1000])
    g9 = gail_curve(f"{EXP}/gail_session3_ext700_ksweep_2026-08-13.csv")
    g2 = gail_curve(f"{EXP}/gail_session3_ext700_RA_ksweep_2026-08-13.csv")
    a9 = gail_curve(f"{EXP}/airl_session3_ext700_ksweep_2026-08-13.csv")
    bc = bc_curve(f"{EXP}/bc_warmstart_ext700_2026-08-13.csv")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    draw(ax, g9, DEMO_SEC, G9_C, "s", "GAIL — 9-feature ranking")
    draw(ax, g2, DEMO_SEC, G2_C, "D", "GAIL — 2-feature (return + consistency)")
    draw(ax, a9, DEMO_SEC, AIRL_C, "^", "AIRL — 9-feature ranking")
    draw(ax, bc, DEMO_SEC, BC_C, "*", "BC-warmstart (BC top-K → PPO on true reward)")
    draw(ax, pt, PT_SEC_PER_PREF, PT_C, "o", "PT — human preferences")

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.axhline(200, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel(f"Human time (min)   —   PT: {PT_SEC_PER_PREF:.1f}s/pref,  demos: {DEMO_SEC:.1f}s/demo "
                  f"(point labels = N preferences / K demos)")
    ax.set_ylabel("Mean eval reward (true env)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Human supervision on LunarLander (699-demo pool) — reward vs human time\n"
                 "PT preferences vs GAIL(9-feat / 2-feat) vs AIRL vs BC-warmstart", fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/human_count_all_ext700.png"
    fig.savefig(out, dpi=150); print("Saved:", out)
    for nm, d in [("PT", pt), ("GAIL9", g9), ("GAIL2", g2), ("AIRL", a9), ("BC", bc)]:
        print(nm, {k: round(v[0]) for k, v in d.items()})


if __name__ == "__main__":
    main()
