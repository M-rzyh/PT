"""Human frame-blanking at BLOCK b=5 (0.25 s @ 20 fps): PT (N=350 preferences) vs GAIL (N=50 demos).

Levels are AUTO-DISCOVERED -- the plot draws whatever has finished training and
silently skips the rest. Re-run it after each new level lands and the point appears.

  0%  is blank-agnostic (b is irrelevant when nothing is blanked), so it reuses the
      existing clean runs: PT humanblock0_mixture / GAIL blind_human_blank0.
  25/50/75%  come from the b=5 runs: PT humanblock5p{P}_mixture, GAIL blank5_p{P}.

PT reward   = last10_eval_reward, recomputed from each IQL run's progress.txt
              (the humanblock pipeline does not write eval_summary.json).
GAIL reward = mean episode return over eval_data/agent_rollouts.npz.

Run with the imitation-gail env (pyarrow, for the demo-return annotations):
    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_frame_blanking_human_b5.py
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.ipc

PT_BASE = "/scratch/marzii/PT/lunarlander/frame_blanking"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C, GA_C = "#1f77b4", "#2ca02c"
LEVELS = [0, 25, 50, 75]

# 0% is the shared clean condition; 25/50/75 are the b=5 runs.
PT_CONDS = {0: "humanblock0_mixture"}
PT_CONDS.update({p: f"humanblock5p{p}_mixture" for p in (25, 50, 75)})

DEMO_DIRS = {0: "/scratch/marzii/imitation_runs/human_demos/lunarlander/session_2"}
DEMO_DIRS.update({
    p: f"/scratch/marzii/imitation_runs/frame_blanking/demos/human/blank5_p{p}/session_1_flagged"
    for p in (25, 50, 75)
})


def agg(v):
    return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def pt_series():
    """last10 = mean of the final 10 in-training IQL evals (progress.txt: 'step reward')."""
    out, nseeds = {}, {}
    for pct, cond in PT_CONDS.items():
        vals = []
        for seed_dir in sorted(glob.glob(f"{PT_BASE}/{cond}/seed_*")):
            hits = glob.glob(f"{seed_dir}/**/progress.txt", recursive=True)
            if not hits:
                continue
            rew = [float(p[1]) for p in (ln.split() for ln in open(hits[0]) if ln.strip())
                   if len(p) >= 2]
            if len(rew) >= 10:
                vals.append(float(np.mean(rew[-10:])))
        if vals:
            out[pct], nseeds[pct] = agg(vals), len(vals)
            print(f"  PT   {pct:>3}%: {len(vals)} seeds, mean {out[pct][0]:+.0f}")
        else:
            print(f"  PT   {pct:>3}%: (not trained yet -- skipped)")
    return out, nseeds


def gail_series():
    """0% from the clean blind_human csv; b=5 levels from gail_blind_human_b5_*.csv."""
    per = {}
    rows = []
    f0 = f"{EXP}/gail_blind_human_2026-07-03.csv"
    if os.path.exists(f0):
        rows += [r for r in csv.DictReader(open(f0)) if int(r["blank_pct"]) == 0]
    for f in sorted(glob.glob(f"{EXP}/gail_blind_human_b5_*.csv")):
        rows += list(csv.DictReader(open(f)))
    for r in rows:
        npz = f"{GAIL_BASE}/{r['slurm_job_id']}/eval_data/agent_rollouts.npz"
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        per.setdefault(int(r["blank_pct"]), []).append(
            float(np.mean([float(e.sum()) for e in d["rews"]])))
    out, nseeds = {}, {}
    for pct in LEVELS:
        v = per.get(pct, [])
        if v:
            out[pct], nseeds[pct] = agg(v), len(v)
            print(f"  GAIL {pct:>3}%: {len(v)} seeds, mean {out[pct][0]:+.0f}")
        else:
            print(f"  GAIL {pct:>3}%: (no demos/runs yet -- skipped)")
    return out, nseeds


def demo_stats():
    """Mean return of the human demos actually fed to GAIL, per level."""
    out = {}
    for pct, d in DEMO_DIRS.items():
        rets = []
        for a in glob.glob(d + "/*.arrow"):
            try:
                with pyarrow.ipc.open_stream(a) as rd:
                    t = rd.read_all()
                rets += [float(sum(ep)) for ep in t.column("rews").to_pylist()]
            except Exception:
                pass
        if rets:
            out[pct] = float(np.mean(rets))
            print(f"  demo {pct:>3}%: {len(rets)} eps, mean return {out[pct]:+.0f}")
    return out


def curve(ax, d, color, marker, label):
    x = np.array(sorted(d))
    y = np.array([d[k][0] for k in x])
    e = np.array([d[k][1] for k in x])
    style = marker + "-" if len(x) > 1 else marker
    ax.plot(x, y, style, color=color, lw=2.2, ms=8, label=label, zorder=4)
    if len(x) > 1:
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.13, lw=0, zorder=1)
    else:  # single point: show the seed spread as an errorbar instead of a band
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color, elinewidth=2, capsize=5, zorder=3)
    return x, y


def seed_txt(n):
    u = sorted(set(n.values()))
    return f"{u[0]} seeds" if len(u) == 1 else f"{min(u)}-{max(u)} seeds"


def main():
    print("Loading (missing levels are skipped, not errors):")
    pt, pt_n = pt_series()
    ga, ga_n = gail_series()
    dr = demo_stats()
    if not pt and not ga:
        raise SystemExit("nothing trained yet -- nothing to plot.")

    fig, ax = plt.subplots(figsize=(9, 6))
    if pt:
        curve(ax, pt, PT_C, "s", f"PT — human preferences (N=350 pairs, {seed_txt(pt_n)})")
    gx = gy = []
    if ga:
        gx, gy = curve(ax, ga, GA_C, "^", f"GAIL — human demos (N=50, {seed_txt(ga_n)})")

    for xi, yi in zip(gx, gy):
        v = dr.get(int(xi))
        if v is not None:
            ax.annotate(f"demo: {v:+.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, -22), ha="center", fontsize=9,
                        color=GA_C, fontweight="bold", zorder=8)

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Frames blanked (%)   —   block length b=5 (0.25 s @ 20 fps)")
    ax.set_ylabel("Mean eval reward")
    ax.set_xticks(LEVELS)
    ax.set_xlim(-6, 81)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    have = sorted(set(pt) | set(ga))
    missing = [p for p in LEVELS if p not in have]
    sub = f"levels shown: {have}" + (f"; not yet trained: {missing}" if missing else "")
    ax.set_title(f"Frame blanking, b=5 (human supervision) — PT vs GAIL\n{sub}", fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/frame_blanking_human_b5_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
