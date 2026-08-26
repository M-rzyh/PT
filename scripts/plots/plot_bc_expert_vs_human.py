"""BC realizability diagnostic: expert-15 vs human-flagged-15 vs human-first-15.

BC fits pi(a|s) by max-likelihood with NO adversary and NO RL, so its final policy return
measures how imitable a demo distribution is by a single Markov policy. If human-BC caps well
below expert-BC (which should ~recover the expert), the human->expert GAIL gap is realizability
(humans aren't a single stationary policy), not a GAIL artifact.

Reads eval_data/agent_rollouts.npz (true-env returns) for each BC run, keyed by the index CSV.
    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_bc_expert_vs_human.py
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BC_RUNS = "/scratch/marzii/imitation_runs/bc/lunarlander"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
CONDS = ["expert15", "human_flagged15", "human_first15"]
LABELS = {"expert15": "Expert-15\n(SAC)", "human_flagged15": "Human-15\nflagged (best)",
          "human_first15": "Human-15\nfirst (uncurated)"}
COLORS = {"expert15": "#2ca02c", "human_flagged15": "#d62728", "human_first15": "#ff7f0e"}


def ep_return(jid):
    f = f"{BC_RUNS}/{jid}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def load():
    best = {}
    for cf in sorted(glob.glob(f"{EXP}/bc_lunarlander_*.csv")):
        for r in csv.DictReader(open(cf)):
            k = (r["condition"], int(r["seed"])); j = int(r["slurm_job_id"])
            if k not in best or j > best[k]:
                best[k] = j
    per = {c: [] for c in CONDS}
    for (c, _s), j in best.items():
        if c not in per:
            continue
        v = ep_return(j)
        if v is not None:
            per[c].append(v)
    return per


def main():
    per = load()
    print("BC true-env return (mean over seeds):")
    for c in CONDS:
        v = per[c]
        print(f"  {c:16s}: n={len(v):2d}  mean {np.mean(v):+.1f}  std {np.std(v):.0f}" if v
              else f"  {c:16s}: (no runs yet)")

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = range(len(CONDS))
    for i, c in enumerate(CONDS):
        v = per[c]
        if not v:
            continue
        m, s = np.mean(v), np.std(v)
        ax.bar(i, m, yerr=s, capsize=6, color=COLORS[c], alpha=0.85,
               label=f"{c} (n={len(v)})")
        # scatter individual seeds for honesty about spread
        ax.scatter([i] * len(v), v, color="k", s=10, alpha=0.4, zorder=5)
        ax.annotate(f"{m:+.0f}", (i, m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABELS[c] for c in CONDS])
    ax.set_ylabel("BC policy — mean eval reward (true env, 50 eps/seed)")
    ax.set_title("Behavior cloning: how imitable is each demo set?\n"
                 "(BC = max-likelihood clone, no RL/adversary — measures realizability)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"{FIG}/bc_expert_vs_human.png"
    fig.savefig(out, dpi=150)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
