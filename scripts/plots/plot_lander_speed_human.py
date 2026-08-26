"""Lander vanishing (human supervision): PT vs GAIL, at levels 0/25/50/75%.

Three curves, styled like the noise plots:
  PT           — N=100 preferences, 10 training seeds.
  GAIL flagged — N=15, the demos the human judged good. Held constant across levels, so this
                 is "same budget, same quality bar, more of the lander hidden".
  GAIL all     — every episode the human SAVED (crashes included). N is NOT constant across
                 levels (28 / 33 / 49 / 74 at 0 / 25 / 50 / 75%) because a harder level takes
                 more attempts to yield 15 good flights, so this arm varies demo count and
                 demo quality together. A different, more realistic question than the flagged
                 arm; do not read the two as the same curve.

The perception axis where the LANDER is removed and replaced by the terrain behind it during
b=5 blocks — ground, pad and flags stay fully visible, so the human keeps the scene but loses
the craft. 0% is the clean baseline (PT labels reuse the first 100 clean labels; the GAIL 0%
is its own fps-20 collection, not the older fps-25 sessions).

PT reward   = last10_eval_reward from each IQL run's progress.txt (no eval_summary.json here).
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

PT_BASE = "/scratch/marzii/PT/lunarlander/sim_speed"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
DEMO_ROOT = "/scratch/marzii/imitation_runs/sim_speed/demos/human"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C = "#1f77b4"          # blue   — PT
GA_FLAG_C = "#d62728"     # red    — GAIL flagged (curated 15)
GA_FIRST15_C = "#ff7f0e"  # orange — GAIL first-15 (uncurated, count-matched)
GA_ALL_C = "#ff9896"      # pink   — GAIL all-saved (uncurated, N varies)
LEVELS = [10, 20, 50]   # fps
N_PREF = 100

PT_CONDS = {p: f"humanspeed{p}_mixture" for p in LEVELS}
# The flagged arm always feeds 15; the all-saved arm feeds whatever the session held.
def _demo_dir(fps, sub):
    if fps == 20:
        return f"/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p0/{sub}"
    return f"{DEMO_ROOT}/speed_fps{fps}/{sub}"
DEMO_DIRS = {
    "flagged": {p: _demo_dir(p, "session_1_flagged") for p in LEVELS},
    "all":     {p: _demo_dir(p, "session_1") for p in LEVELS},
}


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


def _ep_return_mean(jobid):
    npz = f"{GAIL_BASE}/{jobid}/eval_data/agent_rollouts.npz"
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def gail_series(arm):
    """One GAIL curve for the given arm ('flagged' or 'all'), read from the index CSV(s)."""
    rows = []
    for f in sorted(glob.glob(f"{EXP}/gail_lander_speed_human_*.csv")):
        rows += list(csv.DictReader(open(f)))
    # A re-run supersedes the original: keep the highest job id per (level, seed, arm).
    best = {}
    for r in rows:
        if r.get("arm", "flagged") != arm:
            continue
        key = (int(r["speed_fps"]), int(r["seed"]))
        jid = int(r["slurm_job_id"])
        if key not in best or jid > best[key]:
            best[key] = jid
    per = {}
    for (pct, _seed), jid in best.items():
        v = _ep_return_mean(jid)
        if v is not None:
            per.setdefault(pct, []).append(v)
    out, nseeds = {}, {}
    for pct in LEVELS:
        v = per.get(pct, [])
        if v:
            out[pct], nseeds[pct] = agg(v), len(v)
            print(f"  GAIL[{arm:>7}] {pct:>3}%: {len(v)} seeds, mean {out[pct][0]:+.0f}")
        else:
            print(f"  GAIL[{arm:>7}] {pct:>3}%: (no runs yet -- skipped)")
    return out, nseeds


def demo_counts(arm):
    """How many episodes the given arm actually fed GAIL, per level (for the legend/x-note)."""
    out = {}
    for pct, d in DEMO_DIRS[arm].items():
        n = 0
        for a in glob.glob(d + "/*.arrow"):
            try:
                with pyarrow.ipc.open_stream(a) as rd:
                    n += rd.read_all().num_rows
            except Exception:
                pass
        if n:
            out[pct] = n
    return out


def curve(ax, d, color, marker, label, ls="-"):
    x = np.array(sorted(d))
    y = np.array([d[k][0] for k in x])
    e = np.array([d[k][1] for k in x])
    style = marker + ls if len(x) > 1 else marker
    ax.plot(x, y, style, color=color, lw=2.2, ms=8, label=label, zorder=4)
    if len(x) > 1:
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.12, lw=0, zorder=1)
    else:
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color, elinewidth=2, capsize=5, zorder=3)
    return x, y


def seed_txt(n):
    u = sorted(set(n.values()))
    return f"{u[0]} seeds" if len(u) == 1 else f"{min(u)}-{max(u)} seeds"


def main():
    print("Loading (missing levels are skipped, not errors):")
    pt, pt_n = pt_series()
    ga_f, ga_f_n = gail_series("flagged")
    ga_15, ga_15_n = gail_series("first15")
    ga_a, ga_a_n = gail_series("all")
    cnt_all = demo_counts("all")
    if not (pt or ga_f or ga_15 or ga_a):
        raise SystemExit("nothing trained yet -- nothing to plot.")

    fig, ax = plt.subplots(figsize=(9.5, 6))
    if pt:
        curve(ax, pt, PT_C, "s",
              f"PT — preferences (N={N_PREF}, {seed_txt(pt_n)})")
    if ga_f:
        curve(ax, ga_f, GA_FLAG_C, "^",
              f"GAIL — flagged demos (best 15, {seed_txt(ga_f_n)})")
    if ga_15:
        # same count as flagged (15) but uncurated -> isolates curation quality
        curve(ax, ga_15, GA_FIRST15_C, "D",
              f"GAIL — first 15 saved (uncurated, {seed_txt(ga_15_n)})", ls=":")
    if ga_a:
        # count-per-level annotation, since N varies on this arm
        cnt = ", ".join(f"{p}%:{cnt_all.get(p,'?')}" for p in sorted(ga_a))
        curve(ax, ga_a, GA_ALL_C, "v",
              f"GAIL — all saved (N varies [{cnt}], {seed_txt(ga_a_n)})", ls="--")

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Playback speed (fps)   —   10=0.2x, 20=0.4x, 50=1.0x real time")
    ax.set_ylabel("Mean eval reward")
    ax.set_xticks(LEVELS)
    ax.set_xlim(6, 54)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    have = sorted(set(pt) | set(ga_f) | set(ga_15) | set(ga_a))
    missing = [p for p in LEVELS if p not in have]
    sub = f"levels shown: {have}" + (f"; not yet trained: {missing}" if missing else "")
    ax.set_title("Simulation speed (human supervision) — PT vs GAIL\n" + sub, fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/lander_speed_human_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
