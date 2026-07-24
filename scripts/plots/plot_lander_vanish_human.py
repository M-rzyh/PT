"""Lander vanishing (human supervision): PT (N=100 preferences) vs GAIL (N=15 demos).

The perception axis where the LANDER is removed and replaced by the terrain behind it during
b=5 blocks — ground, pad and flags stay fully visible, so the human keeps the scene but loses
the craft. Contrast with frame blanking, where the whole display goes dark.

Levels are AUTO-DISCOVERED: the plot draws whatever has finished training and silently skips
the rest, so it can be re-run as each level lands and the point simply appears.

  0%   is the clean baseline. On the PT side it costs no human time — its labels are the
       first 100 of the existing clean 350 (scripts/preference/make_vanish0_labels.py), which
       is exactly why the 25/50/75 video sets were built from those same pairs. On the GAIL
       side it is a FRESH fps-20 collection, deliberately not the older fps-25 clean sessions,
       which would have put a speed difference on the baseline.
  25/50/75%  are the vanish runs on both arms.

PT reward   = last10_eval_reward, recomputed from each IQL run's progress.txt
              (the human pipeline does not write eval_summary.json).
GAIL reward = mean episode return over eval_data/agent_rollouts.npz.

Run with the imitation-gail env (pyarrow, for the demo-return annotations):
    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_lander_vanish_human.py
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.ipc

PT_BASE = "/scratch/marzii/PT/lunarlander/lander_vanish"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
DEMO_ROOT = "/scratch/marzii/imitation_runs/lander_vanish/demos/human"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C, GA_C = "#1f77b4", "#d62728"
LEVELS = [0, 25, 50, 75]
N_PREF, N_DEMO = 100, 15

PT_CONDS = {p: f"humanvanish{p}_mixture" for p in LEVELS}
DEMO_DIRS = {p: f"{DEMO_ROOT}/vanish_p{p}/session_1_flagged" for p in LEVELS}


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
    """All four levels come from the vanish index CSVs (0% included — it is its own run)."""
    per = {}
    rows = []
    for f in sorted(glob.glob(f"{EXP}/gail_lander_vanish_human_*.csv")):
        rows += list(csv.DictReader(open(f)))
    # A re-run supersedes the original: keep the highest job id per (level, seed) so a
    # resubmitted job never double-counts into the band.
    best = {}
    for r in rows:
        key = (int(r["vanish_pct"]), int(r["seed"]))
        jid = int(r["slurm_job_id"])
        if key not in best or jid > best[key]:
            best[key] = jid
    for (pct, _seed), jid in best.items():
        npz = f"{GAIL_BASE}/{jid}/eval_data/agent_rollouts.npz"
        if not os.path.exists(npz):
            continue                      # still running / failed — not an error here
        d = np.load(npz, allow_pickle=True)
        per.setdefault(pct, []).append(float(np.mean([float(e.sum()) for e in d["rews"]])))
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
    """Mean return of the human demos actually fed to GAIL, per level.

    Only the first N_DEMO are read, because that is what SHUFFLE=0 hands the trainer — the
    session may hold more, and averaging over all of them would annotate the plot with demos
    no run ever saw.
    """
    out = {}
    for pct, d in DEMO_DIRS.items():
        rets = []
        for a in sorted(glob.glob(d + "/*.arrow"))[:N_DEMO]:
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
        curve(ax, pt, PT_C, "s", f"PT — human preferences (N={N_PREF} pairs, {seed_txt(pt_n)})")
    gx = gy = []
    if ga:
        gx, gy = curve(ax, ga, GA_C, "^", f"GAIL — human demos (N={N_DEMO}, {seed_txt(ga_n)})")

    for xi, yi in zip(gx, gy):
        v = dr.get(int(xi))
        if v is not None:
            ax.annotate(f"demo: {v:+.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, -22), ha="center", fontsize=9,
                        color=GA_C, fontweight="bold", zorder=8)

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Lander hidden (%)   —   block length b=5 (0.25 s @ 20 fps), terrain visible")
    ax.set_ylabel("Mean eval reward")
    ax.set_xticks(LEVELS)
    ax.set_xlim(-6, 81)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    have = sorted(set(pt) | set(ga))
    missing = [p for p in LEVELS if p not in have]
    sub = f"levels shown: {have}" + (f"; not yet trained: {missing}" if missing else "")
    ax.set_title("Lander vanishing (human supervision) — PT vs GAIL\n" + sub, fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/lander_vanish_human_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
