"""Frame-blanking (Exp 1) result: PT (preferences) vs GAIL (demonstrations) under
blind supervision. Two panels — automated (agent) supervisors and human supervisors.
Blanking is block b=10; agents solve the NORMAL 8-D task. 5 seeds.

PT reward  = last10_eval_reward (oracle: eval_summary.json; human: progress.txt).
GAIL reward= 50-ep eval_reward_mean from agent_rollouts.npz (via index CSVs).
Missing conditions (still training) are skipped, so this renders a partial preview.
"""
import csv
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FB = "/scratch/marzii/PT/lunarlander/frame_blanking"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C, GA_C = "#1f77b4", "#9467bd"


def agg(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def pt_oracle():
    out = {}
    for L, tag in [(0, "blindoracleclean"), (25, "blindoracle25"), (50, "blindoracle50"), (75, "blindoracle75")]:
        v = [json.load(open(f))["last10_eval_reward"]
             for f in glob.glob(f"{FB}/lunarlander-grid-ms-N350-{tag}/seed_*/eval_summary.json")]
        if v:
            out[L] = agg(v)
    return out


def _last10(seed_dir):
    ps = glob.glob(f"{seed_dir}/**/progress.txt", recursive=True)
    if not ps:
        return None
    data = np.loadtxt(ps[0])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return float(data[-10:, 1].mean())


def pt_human():
    out = {}
    for L, cond in [(0, "humanblock0"), (50, "humanblock50")]:
        v = [r for sd in glob.glob(f"{FB}/{cond}_mixture/seed_*") if (r := _last10(sd)) is not None]
        if v:
            out[L] = agg(v)
    return out


def gail_levels(csv_glob, jobcol):
    files = sorted(glob.glob(csv_glob))
    if not files:
        return {}
    per = {}
    for r in csv.DictReader(open(files[-1])):
        L = int(r["blank_pct"])
        npz = f"{GAIL_BASE}/{r[jobcol]}/eval_data/agent_rollouts.npz"
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        per.setdefault(L, []).append(float(np.mean([float(e.sum()) for e in d["rews"]])))
    return {L: agg(v) for L, v in per.items()}


def curve(ax, d, color, marker, label):
    if not d:
        return
    x = np.array(sorted(d))
    y = np.array([d[k][0] for k in x]); e = np.array([d[k][1] for k in x])
    ax.errorbar(x, y, yerr=e, fmt=marker + "-", color=color, ms=7, lw=2, capsize=4, label=label)


def main():
    pto = pt_oracle()
    gexp = gail_levels("/home/marzii/IRL3/experiments/gail_blind_expert_block_*.csv", "slurm_job_id")
    pth = pt_human()
    ghum = gail_levels("/home/marzii/IRL3/experiments/gail_blind_human_*.csv", "slurm_job_id")
    print("PT oracle:", {k: round(v[0]) for k, v in pto.items()})
    print("GAIL expert:", {k: round(v[0]) for k, v in gexp.items()})
    print("PT human:", {k: round(v[0]) for k, v in pth.items()})
    print("GAIL human:", {k: round(v[0]) for k, v in ghum.items()})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    curve(axL, pto, PT_C, "o", "PT blind-oracle")
    curve(axL, gexp, GA_C, "^", "GAIL blind-expert")
    axL.set_title("Automated (agent) blind supervision")
    curve(axR, pth, PT_C, "o", "PT blind-human")
    curve(axR, ghum, GA_C, "^", "GAIL blind-human")
    axR.set_title("Human blind supervision")
    for ax in (axL, axR):
        ax.axhline(0, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("Blanking (%)"); ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 79); ax.legend(loc="lower left", fontsize=9)
    axL.set_ylabel("Mean eval reward (5 seeds)")
    fig.suptitle("Frame-blanking: PT (preferences, N=350) vs GAIL (demos, N=50) — block b=10, 5 seeds",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{FIG}/frame_blanking_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
