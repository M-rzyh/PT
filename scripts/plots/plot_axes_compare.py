"""Cross-axis comparison: overlay vanish / speed / delay with PT (left) vs GAIL (right).

x is a SHARED CATEGORICAL difficulty scale, not a normalized rank — because only speed has a
level EASIER than the standard baseline (10 fps), so a rank-normalized x would misalign the
baselines. Categories (left->right): easier · normal · medium · hard · super-hard, where
'normal' is the standard condition every axis shares (vanish 0%, speed 20 fps, delay k=0).

    easier   normal   medium   hard   super-hard
 vanish %      –        0       25      50       75
 speed  fps   10       20        –      50        –
 delay  k      –        0        5      10       20

Two versions:
  --gail all   : GAIL shows only the 'all saved' arm  (one line per axis)
  --gail arms  : GAIL shows all three arms            (color = axis, linestyle = arm)

The GAIL 'all' arm trains on EVERY saved demo, so its N varies by level — shown in a
reference box in the GAIL panel.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_axes_compare.py --gail all
    ... --gail arms
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_axis_human as P   # reuse CFG + loaders

FIG = "/home/marzii/PT/PreferenceTransformer/figures"
AXES = ["vanish", "speed", "delay"]
AXCOL = {"vanish": "#1f77b4", "speed": "#2ca02c", "delay": "#d62728"}
ARMSTYLE = {"all": dict(ls="-", lw=3.0, ms=8), "flagged": dict(ls="--", lw=1.8, ms=6),
            "first15": dict(ls=":", lw=1.8, ms=6)}
ARMMARK = {"all": "v", "flagged": "^", "first15": "D"}

# shared categorical x: level -> category index (0=easier ... 4=super-hard)
CATLABELS = ["easier", "normal", "medium", "hard", "super-hard"]
CATMAP = {
    "vanish": {0: 1, 25: 2, 50: 3, 75: 4},
    "speed":  {10: 0, 20: 1, 50: 3},
    "delay":  {0: 1, 5: 2, 10: 3, 20: 4},
}
UNIT = {"vanish": "%", "speed": "fps", "delay": "k"}
# per-axis label offset (points), so value labels don't collide where the axes coincide:
OFF = {"vanish": (5, -11), "speed": (-20, 6), "delay": (6, 6)}
# point-specific overrides {level: (dx,dy)} to pull individual crowded labels clear.
# PT panel: at 'normal' vanish(+260) & delay(+252) sat on top of speed(+264).
PT_OFFAT = {"vanish": {0: (2, -20)}, "delay": {0: (18, -3)}}
# GAIL panel: blue vanish(+30) at 'normal', and speed(-105) at 'hard' next to delay(-98).
GAIL_OFFAT = {"vanish": {0: (2, -20)}, "speed": {50: (-8, -16)}}


def label(ax, cfg):
    return f"{ax} ({'/'.join(str(p) for p in cfg['levels'])} {UNIT[ax]})"


def plot_series(pane, ax, d, color, marker, st, lbl=None, annotate=True, off=(5, 5), off_at=None):
    # off      : default label offset (points) for every point
    # off_at   : {level: (dx,dy)} to override the offset for specific points (by level value)
    off_at = off_at or {}
    xs = sorted(d); x = [CATMAP[ax][k] for k in xs]; y = [d[k][0][0] for k in xs]
    pane.plot(x, y, marker, color=color, label=lbl, **st)
    if annotate:                       # print the exact mean eval reward beside each point
        for k, xi, yi in zip(xs, x, y):
            pane.annotate(f"{yi:+.0f}", (xi, yi), textcoords="offset points",
                          xytext=off_at.get(k, off),
                          fontsize=7, color=color, fontweight="bold", zorder=10)
    return xs, x, y


def gail_n(cfg):
    """N saved demos per level for the 'all' arm (its N varies)."""
    out = {}
    for p in cfg["gail_levels"]:
        out[p] = len(P.arrow_returns(f"{cfg['demo_dir'](p)}/session_1"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gail", choices=["all", "arms"], required=True)
    args = ap.parse_args()

    data = {}
    for ax in AXES:
        cfg = P.CFG[ax]()
        data[ax] = dict(cfg=cfg, pt=P.pt_reward(cfg),
                        arms={a: P.gail_reward(cfg, a) for a in ("all", "flagged", "first15")},
                        n=gail_n(cfg))

    fig, (axP, axG) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # ---- PT panel ----
    for ax in AXES:
        plot_series(axP, ax, data[ax]["pt"], AXCOL[ax], "s",
                    dict(ls="-", lw=2.6, ms=7), lbl=label(ax, data[ax]["cfg"]),
                    off=OFF[ax], off_at=PT_OFFAT.get(ax))
    axP.set_title("PT — preferences (N=100, 30 seeds)", fontsize=13)
    axP.set_ylabel("Mean eval reward (30 seeds)"); axP.legend(fontsize=9, loc="lower left")

    # ---- GAIL panel ----
    if args.gail == "all":
        for ax in AXES:
            xs, xc, yc = plot_series(axG, ax, data[ax]["arms"]["all"], AXCOL[ax], "v",
                                     dict(ls="-", lw=2.8, ms=8), lbl=label(ax, data[ax]["cfg"]),
                                     off=OFF[ax], off_at=GAIL_OFFAT.get(ax))
        axG.set_title("GAIL — all-saved demos (30 seeds)", fontsize=13)
        axG.legend(fontsize=9, loc="lower left")
        tag = "gail_all"
    else:
        for ax in AXES:
            for arm in ("all", "flagged", "first15"):
                # label only the 'all' arm here — 9 lines is too dense to label them all
                plot_series(axG, ax, data[ax]["arms"][arm], AXCOL[ax], ARMMARK[arm],
                            ARMSTYLE[arm], annotate=(arm == "all"), off=OFF[ax])
        axG.set_title("GAIL — all 3 arms (color = axis, style = arm; 30 seeds)", fontsize=13)
        axleg = [Line2D([0], [0], color=AXCOL[a], lw=3, label=label(a, data[a]["cfg"])) for a in AXES]
        armleg = [Line2D([0], [0], color="gray", marker=ARMMARK[a], label=a, **ARMSTYLE[a])
                  for a in ("all", "flagged", "first15")]
        l1 = axG.legend(handles=axleg, fontsize=8.5, loc="lower left", title="axis")
        axG.add_artist(l1); axG.legend(handles=armleg, fontsize=8, loc="upper left", title="GAIL arm")
        tag = "gail_arms"

    # ---- reference box: GAIL 'all' N per level (it varies) ----
    lines = ["GAIL all — N demos:"]
    for ax in AXES:
        n = data[ax]["n"]
        lines.append(f"  {ax}: " + "/".join(str(n[p]) for p in data[ax]["cfg"]["gail_levels"]))
    axG.text(0.985, 0.97, "\n".join(lines), transform=axG.transAxes, fontsize=8,
             va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    for a in (axP, axG):
        a.axhline(0, color="gray", ls=":", alpha=0.4); a.grid(True, alpha=0.3)
        a.set_xticks(range(len(CATLABELS))); a.set_xticklabels(CATLABELS)
        a.set_xlim(-0.3, 4.3); a.set_xlabel("Supervision difficulty")
    fig.suptitle("Supervision difficulty across axes — vanish / speed / delay", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{FIG}/axes_compare_PT_vs_GAIL_{tag}.png"
    fig.savefig(out, dpi=150); print("Saved:", out)


if __name__ == "__main__":
    main()
