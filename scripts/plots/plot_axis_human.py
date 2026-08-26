"""Unified 'human supervision' plot for the vanish / speed / delay axes.

Top panel : mean eval reward (30 seeds) vs difficulty level — PT (N=100 prefs) and the three
            GAIL arms (flagged=best-15, first15=first-15-saved, all=every saved demo).
Mid panel : HUMAN TIME to produce the supervision at each level — PT labelling time and the
            three GAIL arms' EXACT times (time to reach the 15th flagged / first 15 saved /
            all episodes, from timing.csv + flags.json). "How long did it cost."
Table     : per level × arm, the mean DEMO return fed to GAIL (the "average demo performance").

Run with the imitation-gail env (pyarrow):
    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_axis_human.py --axis vanish
    ... --axis speed   |   --axis delay
"""
import argparse, csv, glob, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.ipc

GAIL_RUNS = "/scratch/marzii/imitation_runs/gail/lunarlander"
EXP = "/home/marzii/IRL3/experiments"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C = "#1f77b4"          # PT preferences
ALL_C = "#b30000"         # GAIL all-saved — the HEADLINE arm (bold solid)
FLAG_C = "#d62728"        # GAIL flagged (dashed)
F15_C = "#ff7f0e"         # GAIL first-15 (dashed)
CLEAN350 = "/scratch/marzii/PT/lunarlander/frame_blanking/pt_human/labels_n350_clean.pkl"
PT_CAP_S = 30             # cap per-pair labelling time: drops idle gaps (page left open),
                          # keeps genuine deliberation. Without it a few 100-240s idle pairs
                          # inflate the total (e.g. vanish25 22->~11 min).

# ---- per-axis configuration -------------------------------------------------
# demo_dir(level)  -> GAIL session dir ; pt_labels(level) -> PT web-label pickle (or ("clean100"))
def cfg_vanish():
    return dict(
        name="Lander vanishing", xlabel="Lander hidden (%)", levels=[0, 25, 50, 75],
        gail_levels=[0, 25, 50, 75], pt_base="/scratch/marzii/PT/lunarlander/lander_vanish",
        pt_cond=lambda p: f"humanvanish{p}_mixture", lvl_col="vanish_pct",
        csv_glob=f"{EXP}/gail_lander_vanish_human_*.csv",
        demo_dir=lambda p: f"/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p{p}",
        pt_labels=lambda p: ("clean100" if p == 0 else
            f"/scratch/marzii/PT/lunarlander/lander_vanish/pt_human/labels_n100_vanish{p}.pkl"))

def cfg_speed():
    vp0 = "/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p0"
    return dict(
        name="Simulation speed", xlabel="Playback speed (fps): 10=0.2x 20=0.4x 50=1.0x realtime",
        levels=[10, 20, 50], gail_levels=[10, 20, 50],
        pt_base="/scratch/marzii/PT/lunarlander/sim_speed",
        pt_cond=lambda p: f"humanspeed{p}_mixture", lvl_col="speed_fps",
        csv_glob=f"{EXP}/gail_lander_speed_human_*.csv",
        demo_dir=lambda p: (vp0 if p == 20 else
            f"/scratch/marzii/imitation_runs/sim_speed/demos/human/speed_fps{p}"),
        pt_labels=lambda p: ("clean100" if p == 20 else   # 20fps reuses vanish-0 = clean-100
            f"/scratch/marzii/PT/lunarlander/sim_speed/pt_human/labels_n100_speed{p}.pkl"))

def cfg_delay():
    vp0 = "/scratch/marzii/imitation_runs/lander_vanish/demos/human/vanish_p0"
    return dict(
        name="Action delay", xlabel="Action delay k (steps): k*20 ms game time",
        levels=[0, 5, 10, 20], gail_levels=[0, 5, 10],   # k=20 is PT-only
        pt_base="/scratch/marzii/PT/lunarlander/action_delay",
        pt_cond=lambda p: f"humandelay{p}_mixture", lvl_col="delay_k",
        csv_glob=f"{EXP}/gail_lander_delay_human_*.csv",
        demo_dir=lambda p: (vp0 if p == 0 else
            f"/scratch/marzii/imitation_runs/action_delay/demos/human/delay_k{p}"),
        pt_labels=lambda p: f"/scratch/marzii/PT/lunarlander/action_delay/pt_human/labels_n100_delay{p}.pkl")

CFG = {"vanish": cfg_vanish, "speed": cfg_speed, "delay": cfg_delay}


def agg(v): return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def pt_reward(cfg):
    out = {}
    for p in cfg["levels"]:
        vals = []
        for sd in sorted(glob.glob(f"{cfg['pt_base']}/{cfg['pt_cond'](p)}/seed_*")):
            h = glob.glob(f"{sd}/**/progress.txt", recursive=True)
            if not h: continue
            r = [float(x[1]) for x in (l.split() for l in open(h[0]) if l.strip()) if len(x) >= 2]
            if len(r) >= 10: vals.append(np.mean(r[-10:]))
        if vals: out[p] = (agg(vals), len(vals))
    return out


def _ep_ret(jid):
    f = f"{GAIL_RUNS}/{jid}/eval_data/agent_rollouts.npz"
    if not os.path.exists(f): return None
    d = np.load(f, allow_pickle=True)
    return float(np.mean([float(e.sum()) for e in d["rews"]]))


def gail_reward(cfg, arm):
    best = {}
    for f in sorted(glob.glob(cfg["csv_glob"])):
        for r in csv.DictReader(open(f)):
            if r.get("arm") != arm: continue
            k = (int(r[cfg["lvl_col"]]), int(r["seed"])); j = int(r["slurm_job_id"])
            if k not in best or j > best[k]: best[k] = j
    per = {}
    for (lvl, _s), j in best.items():
        v = _ep_ret(j)
        if v is not None: per.setdefault(lvl, []).append(v)
    return {lvl: (agg(per[lvl]), len(per[lvl])) for lvl in per}


def arrow_returns(path, limit=None):
    rets = []
    for a in sorted(glob.glob(path + "/*.arrow")):
        with pyarrow.ipc.open_stream(a) as rd:
            for ep in rd.read_all().column("rews").to_pylist():
                rets.append(float(sum(ep)))
    return rets[:limit] if limit else rets


def demo_perf(cfg):
    """Mean return of the demos each arm actually trains on, per level."""
    out = {}
    for p in cfg["gail_levels"]:
        d = cfg["demo_dir"](p)
        allr = arrow_returns(f"{d}/session_1")
        flag = arrow_returns(f"{d}/session_1_flagged")
        out[p] = {"flagged": np.mean(flag) if flag else None,
                  "first15": np.mean(allr[:15]) if len(allr) >= 15 else None,
                  "all":     np.mean(allr) if allr else None}
    return out


def gail_time(cfg):
    """EXACT human play-time per arm per level (minutes), from timing.csv + flags.json."""
    out = {}
    for p in cfg["gail_levels"]:
        d = cfg["demo_dir"](p)
        try:
            rows = list(csv.DictReader(open(f"{d}/session_1/timing.csv")))
            cum = [float(r["cumulative_sec"]) for r in rows]
            fl = json.load(open(f"{d}/session_1/flags.json"))["flagged_indices"]
            out[p] = {"flagged": cum[fl[14]] / 60 if len(fl) > 14 else cum[-1] / 60,
                      "first15": cum[14] / 60 if len(cum) > 14 else cum[-1] / 60,
                      "all":     cum[-1] / 60}
        except Exception:
            pass
    return out


def gail_flight_time(cfg):
    """REAL flight time (minutes) of each arm's OWN episodes = sum of their durations.
    flagged = the 15 flagged episodes' actual flight durations; first15 = the first 15 saved;
    all = every saved episode (the full effort spent to produce the flagged 15). So flagged
    and all genuinely differ, and 'all' shows how much flying it took to yield 15 good ones."""
    out = {}
    for p in cfg["gail_levels"]:
        d = cfg["demo_dir"](p)
        try:
            rows = list(csv.DictReader(open(f"{d}/session_1/timing.csv")))
            durs = [float(r["duration_sec"]) for r in rows]
            fl = json.load(open(f"{d}/session_1/flags.json"))["flagged_indices"]
            out[p] = {"flagged": sum(durs[i] for i in fl[:15]) / 60,
                      "first15": sum(durs[:15]) / 60,
                      "all":     sum(durs) / 60}
        except Exception:
            pass
    return out


def gail_ndemo(cfg):
    """N demos per level for the 'all' arm (this is the count that varies; flagged/first15=15)."""
    return {p: len(arrow_returns(f"{cfg['demo_dir'](p)}/session_1")) for p in cfg["gail_levels"]}


def pt_time(cfg):
    """PT labelling time (minutes) per level, from the web pickle's time_sec."""
    out = {}
    clean_first100 = None
    for p in cfg["levels"]:
        src = cfg["pt_labels"](p)
        if src == "clean100":
            if clean_first100 is None:
                ts = pickle.load(open(CLEAN350, "rb")).get("time_sec", [])
                clean_first100 = sum(min(x, PT_CAP_S) for x in ts[:100] if x) / 60
            out[p] = clean_first100
        elif os.path.exists(src):
            ts = pickle.load(open(src, "rb")).get("time_sec", [])
            out[p] = sum(min(x, PT_CAP_S) for x in ts if x) / 60
    return out


def curve(ax, d, color, marker, label, ls="-", lw=2.2, ms=7, zorder=4):
    x = np.array(sorted(d)); y = np.array([d[k][0][0] for k in x]); e = np.array([d[k][0][1] for k in x])
    ax.plot(x, y, marker, color=color, ls=ls, lw=lw, ms=ms, label=label, zorder=zorder)
    if len(x) > 1: ax.fill_between(x, y - e, y + e, color=color, alpha=0.12, lw=0)
    else: ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color, elinewidth=2, capsize=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(CFG))
    args = ap.parse_args()
    cfg = CFG[args.axis]()

    pt = pt_reward(cfg)
    arms = {a: gail_reward(cfg, a) for a in ("flagged", "first15", "all")}
    dperf = demo_perf(cfg); gtime = gail_time(cfg); ptime = pt_time(cfg)
    gflight = gail_flight_time(cfg)   # real per-arm flight time for the time table
    nall = gail_ndemo(cfg)            # N demos for the all arm (varies per level)

    print(f"[{args.axis}] PT levels:", {p: round(v[0][0]) for p, v in pt.items()})
    for a in arms: print(f"  GAIL {a}:", {p: round(v[0][0]) for p, v in arms[a].items()})

    fig = plt.figure(figsize=(12, 9.5))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3.2, 1.5, 1.5], hspace=0.4, wspace=0.28)
    axR = fig.add_subplot(gs[0, :])            # reward — full width
    axT = fig.add_subplot(gs[1, :])            # time line-panel — full width
    axTab1 = fig.add_subplot(gs[2, 0])         # Table 1 (returns) — bottom-left
    axTab2 = fig.add_subplot(gs[2, 1])         # Table 2 (time)    — bottom-right

    # ---- reward ----
    # all-saved is the HEADLINE arm -> bold solid; flagged & first15 dashed & thinner.
    if pt: curve(axR, pt, PT_C, "s", "PT — preferences (N=100)", lw=2.4)
    if arms["all"]:
        curve(axR, arms["all"], ALL_C, "v", "GAIL — all saved (N varies)",
              ls="-", lw=3.2, ms=9, zorder=6)
        # label each all-arm point with how many demos it trained on (the count that varies).
        # White bbox so the number stays readable even when it lands over another line.
        for p in sorted(arms["all"]):
            axR.annotate(f"N={nall.get(p, '?')}", (p, arms["all"][p][0][0]),
                         textcoords="offset points", xytext=(0, 16), ha="center",
                         fontsize=8, color=ALL_C, fontweight="bold", zorder=20,
                         bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=ALL_C,
                                   lw=0.8, alpha=0.95))
    if arms["flagged"]: curve(axR, arms["flagged"], FLAG_C, "^", "GAIL — flagged (best 15)",
                              ls="--", lw=1.7, ms=6, zorder=5)
    if arms["first15"]: curve(axR, arms["first15"], F15_C, "D", "GAIL — first 15 saved",
                              ls="--", lw=1.7, ms=6, zorder=5)
    axR.axhline(0, color="gray", ls=":", alpha=0.4); axR.grid(True, alpha=0.3)
    axR.set_ylabel("Mean eval reward\n(30 seeds)"); axR.legend(loc="lower left", fontsize=8)
    axR.set_title(f"{cfg['name']} (human supervision) — PT vs GAIL", fontsize=12)

    # ---- time panel ----
    lv = cfg["gail_levels"]
    for a, c, m, st in (("all", ALL_C, "v", dict(ls="-", lw=3.0, ms=8)),
                        ("flagged", FLAG_C, "^", dict(ls="--", lw=1.7, ms=6)),
                        ("first15", F15_C, "D", dict(ls="--", lw=1.7, ms=6))):
        xs = [p for p in lv if p in gtime]; ys = [gtime[p][a] for p in xs]
        if xs: axT.plot(xs, ys, m, color=c, label=f"GAIL {a}", **st)
    px = [p for p in cfg["levels"] if p in ptime]
    if px: axT.plot(px, [ptime[p] for p in px], "s-", color=PT_C, ms=6, lw=1.8, label="PT labelling")
    axT.grid(True, alpha=0.3); axT.set_ylabel("Human time\n(minutes)")
    axT.set_xlabel(cfg["xlabel"]); axT.legend(fontsize=7, ncol=2)

    # ---- Table 1: AVERAGE demo return per arm × level ----
    xcols = cfg["levels"]                      # full x (includes any PT-only level, e.g. k=20)
    cols = [str(p) for p in xcols]
    xlab = cfg["xlabel"].split(":")[0].split("(")[0].strip()

    def render(ax, rows, cell, title):
        ax.axis("off")
        t = ax.table(cellText=cell, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)
        ax.set_title(title, fontsize=9, pad=4)

    r1 = ["flagged (avg)", "first-15 (avg)", "all (avg)", "N demos (all)"]
    c1 = [[f"{dperf[p][k]:+.0f}" if dperf.get(p, {}).get(k) is not None else "–" for p in xcols]
          for k in ("flagged", "first15", "all")]
    c1.append([str(nall[p]) if p in nall else "–" for p in xcols])   # demo count for the all arm
    render(axTab1, r1, c1, f"GAIL avg demo return per arm  (flagged/first-15 = 15 demos)   (x = {xlab})")

    # ---- Table 2: REAL human time (min): GAIL = flight time of each arm's episodes ----
    r2 = ["GAIL flagged", "GAIL first-15", "GAIL all", "PT label"]
    c2 = [[f"{gflight[p][k]:.1f}" if p in gflight else "–" for p in xcols]
          for k in ("flagged", "first15", "all")]
    c2.append([f"{ptime[p]:.1f}" if p in ptime else "–" for p in xcols])
    render(axTab2, r2, c2, f"Human time (min) — real flight time   (x = {xlab})")

    out = f"{FIG}/{args.axis}_human_pt_vs_gail_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("Saved:", out)


if __name__ == "__main__":
    main()
