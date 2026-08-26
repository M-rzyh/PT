"""Combined HUMAN-supervision comparison on LunarLander, styled like the oracle
plot_grid_pt_vs_gail.py count axis (x = human time in minutes).

Four methods, all on the SAME human data-collection effort:
  - PT   (blue  -o): reward from human PREFERENCES, N in {10,100,250,750,1000}, 30 seeds.
  - GAIL (green -s): imitation from human DEMOS, ranked top-K, 5 seeds  (25 extra were cancelled).
  - AIRL (purple -^): IRL+RL from human DEMOS, ranked top-K, 30 seeds.
  - BC-warmstart (orange *): BC(demos) -> PPO on TRUE reward. A=all 300, B=landed 180, 5 seeds.

Human-time rates measured THIS session:
  PT: 151 min / 1000 preferences = 9.06 s/pref ;  demos: 54 min / 300 = 10.8 s/demo.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_human_pt_gail_airl_bc.py
"""
import csv, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA = "/scratch/marzii/imitation_runs/gail/lunarlander"
BW = "/scratch/marzii/imitation_runs/bc_warmstart/lunarlander"
PT_MS = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"

PT_SEC_PER_PREF = 151 * 60 / 1000    # 9.06 s/preference
DEMO_SEC = 54 * 60 / 300             # 10.8 s/demo

PT_C, GAIL_C, AIRL_C, BC_C = "#1f77b4", "#2ca02c", "#9467bd", "#e6820e"


def ep_return(job):
    f = f"{GA}/{job}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f): return None
    d = np.load(f, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def demo_curve(idx_csv):
    per = defaultdict(list)
    for r in csv.reader(open(idx_csv)):
        if not r or r[0] in ("condition", "variant") or r[0].startswith("#"):
            continue
        K, job = int(r[1]), r[3].strip()
        v = ep_return(job)
        if v is not None: per[K].append(v)
    return {K: (np.mean(v), np.std(v), len(v)) for K, v in sorted(per.items())}


def pt_curve(Ns):
    out = {}
    for N in Ns:
        vals = []
        for s in range(30):
            f = f"{PT_MS}/lunarlander-human-count-N{N}/seed_{s}/eval_summary.json"
            if os.path.exists(f):
                vals.append(json.load(open(f))["last10_eval_reward"])
        if vals: out[N] = (np.mean(vals), np.std(vals), len(vals))
    return out


def bc_points():
    per = defaultdict(list)
    for r in csv.DictReader(open(f"{EXP}/bc_warmstart_2026-08-12.csv")):
        f = f"{BW}/{r['jobid'].strip()}/eval_data/meta.json"
        if os.path.exists(f):
            m = json.load(open(f)); per[r["variant"]].append(m["ep_reward_mean"])
    # variant -> (n_demos, mean, std)
    ndem = {"A": 300, "B": 180}
    return {v: (ndem[v], np.mean(x), np.std(x)) for v, x in per.items()}


def main():
    pt = pt_curve([10, 100, 250, 750, 1000])
    gail = demo_curve(f"{EXP}/gail_session3_ranked_ksweep_2026-08-12.csv")
    airl = demo_curve(f"{EXP}/airl_session3_ranked_ksweep_2026-08-12.csv")
    bc = bc_points()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # ---- demo-based series (x = K demos * DEMO_SEC) — line + shaded ±1 std ----
    for d, c, m, lbl in [
        (gail, GAIL_C, "-s", "GAIL — human demos (top-K, 5 seeds)"),
        (airl, AIRL_C, "-^", "AIRL — human demos (top-K, 30 seeds)"),
    ]:
        Ks = sorted(d)
        x = np.array([K * DEMO_SEC / 60 for K in Ks])
        y = np.array([d[K][0] for K in Ks]); e = np.array([d[K][1] for K in Ks])
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.12, zorder=2)
        ax.plot(x, y, m, color=c, lw=2.1, ms=7, label=lbl, zorder=4)
        for K, xi, yi in zip(Ks, x, y):
            ax.annotate(f"{K}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=c)

    # ---- PT (x = N prefs * PT_SEC_PER_PREF) — line + shaded ±1 std ----
    Ns = sorted(pt)
    x = np.array([N * PT_SEC_PER_PREF / 60 for N in Ns])
    y = np.array([pt[N][0] for N in Ns]); e = np.array([pt[N][1] for N in Ns])
    ax.fill_between(x, y - e, y + e, color=PT_C, alpha=0.15, zorder=3)
    ax.plot(x, y, "-o", color=PT_C, lw=2.4, ms=8, label="PT — human preferences (30 seeds)", zorder=5)
    for N, xi, yi in zip(Ns, x, y):
        ax.annotate(f"{N}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color=PT_C, fontweight="bold")

    # ---- BC-warmstart reference stars (x = n_demos * DEMO_SEC) ----
    for v, (nd, mean, std) in bc.items():
        xi = nd * DEMO_SEC / 60
        lab = {"A": "BC-warmstart (all 300)", "B": "BC-warmstart (landed 180)"}[v]
        ax.plot(xi, mean, "*", color=BC_C, ms=17, markeredgecolor="k", markeredgewidth=0.5,
                label=lab, zorder=6)
        ax.annotate(f"{mean:+.0f}±{std:.0f}", (xi, mean), textcoords="offset points", xytext=(10, 0),
                    ha="left", va="center", fontsize=8, color=BC_C, fontweight="bold")

    ax.axhline(0, color="gray", ls="--", alpha=0.4)
    ax.axhline(200, color="gray", ls=":", alpha=0.35)
    ax.annotate("solved (+200)", (ax.get_xlim()[1], 200), fontsize=7.5, color="gray",
                va="bottom", ha="right")
    ax.set_xlabel(f"Human time (min)   —   PT: {PT_SEC_PER_PREF:.1f}s/pref,  demos: {DEMO_SEC:.1f}s/demo")
    ax.set_ylabel("Mean eval reward (true env)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Human supervision on LunarLander — PT vs GAIL vs AIRL vs BC-warmstart\n"
                 "labels/demos on same human-time axis; markers annotated with N preferences / K demos",
                 fontsize=10.5)
    fig.tight_layout()
    out = f"{FIG}/human_pt_gail_airl_bc.png"
    fig.savefig(out, dpi=150); print("Saved:", out)
    for name, d in [("PT", pt), ("GAIL", gail), ("AIRL", airl)]:
        print(name, {k: round(v[0]) for k, v in d.items()})
    print("BC", {v: round(x[1]) for v, x in bc.items()})


if __name__ == "__main__":
    main()
