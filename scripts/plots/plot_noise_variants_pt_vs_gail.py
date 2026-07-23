"""Noise-injection VARIANT comparison, PT vs GAIL — same setting as the final
noise plot (PT N=1000, GAIL N=100, 30 seeds, nonFS expert 4615187).

Two panels, shared y:
  Panel 1 = RANDOM-REPLACE : PT {noise (coin), exnoise (exact)} + GAIL {coin_uniform, exact_uniform}
  Panel 2 = FLIP           : PT {flipnoise (coin), exflipnoise (exact)} + GAIL {coin_exclude, exact_exclude}

Encoding (dataviz): color = METHOD (PT blue, GAIL purple); the coin/exact distinction
is a SECONDARY encoding, never a new hue — exact = bold line + light band, coin =
faint dashed "shadow" (no band). If the shadow hides under the bold line, coin≈exact
and it can be dropped.

Metrics match the existing plot: PT last10_eval_reward; GAIL 50-ep eval_reward_mean.
Missing conditions (runs still in flight) are skipped, so this renders a partial
preview now and fills in as jobs finish.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.ipc

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
GRID_ROOT_MS = SCRATCH / "PT/lunarlander/grid_mixture_ms"
GAIL_BASE = Path("/scratch/marzii/imitation_runs/gail/lunarlander")
GAIL_NOISE_OLD = "/home/marzii/IRL3/experiments/gail_grid_noise_2026-06-18.csv"   # coin_uniform (N=100)
GAIL_COUNT_CSV = "/home/marzii/IRL3/experiments/gail_grid_count_2026-06-18.csv"   # count_N100 = 0% baseline
FIG_DIR = Path("/home/marzii/PT/PreferenceTransformer/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PT_COLOR = "#1f77b4"
GAIL_COLOR = "#9467bd"
LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def agg(vals):
    if not vals:
        return None, None, 0
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std()), len(a)


# ── PT ───────────────────────────────────────────────────────────────────────
def load_pt_ms():
    rows = defaultdict(list)
    for env_dir in GRID_ROOT_MS.glob("lunarlander-grid-ms-*"):
        for f in env_dir.glob("seed_*/eval_summary.json"):
            try:
                rows[env_dir.name].append(float(json.load(open(f))["last10_eval_reward"]))
            except Exception:
                pass
    return rows


def pt_series(pt_ms, noise_word, clean_word):
    xs, ys, es = [], [], []
    for p in LEVELS:
        cond = (f"lunarlander-grid-ms-N1000-{clean_word}" if p == 0
                else f"lunarlander-grid-ms-N1000-{noise_word}{p}")
        m, s, n = agg(pt_ms.get(cond, []))
        if m is None:
            continue
        xs.append(p); ys.append(m); es.append(s)
    return np.array(xs), np.array(ys), np.array(es)


# ── GAIL ─────────────────────────────────────────────────────────────────────
def load_gail_zero():
    vals = []
    for r in csv.DictReader(open(GAIL_COUNT_CSV)):
        if r["condition_id"] == "count_N100" and r.get("eval_reward_mean"):
            try: vals.append(float(r["eval_reward_mean"]))
            except ValueError: pass
    return vals


def load_gail_coin_uniform():
    rows = defaultdict(list)
    for r in csv.DictReader(open(GAIL_NOISE_OLD)):
        cond, v = r["condition_id"], r.get("eval_reward_mean", "")
        if v and cond.startswith("noise_p"):
            try: rows[int(cond[len("noise_p"):])].append(float(v))
            except ValueError: pass
    return rows


def load_gail_variants():
    gail = defaultdict(lambda: defaultdict(list))
    for csvp in sorted(glob.glob("/home/marzii/IRL3/experiments/gail_noise_variants_*.csv")):
        for r in csv.DictReader(open(csvp)):
            npz = GAIL_BASE / r["slurm_job_id"] / "eval_data" / "agent_rollouts.npz"
            if not npz.exists():
                continue
            try:
                d = np.load(npz, allow_pickle=True)
                ep = [float(e.sum()) for e in d["rews"]]
                gail[r["mode"]][int(r["noise_pct"])].append(float(np.mean(ep)))
            except Exception:
                pass
    return gail


def gail_series(per_level, zero_vals):
    xs, ys, es = [], [], []
    for p in LEVELS:
        vals = zero_vals if p == 0 else per_level.get(p, [])
        m, s, n = agg(vals)
        if m is None:
            continue
        xs.append(p); ys.append(m); es.append(s)
    return np.array(xs), np.array(ys), np.array(es)


# ── plotting ─────────────────────────────────────────────────────────────────
def bold_line(ax, x, y, e, color, marker, label):
    """coin = the reference curve: bold line WITH markers + band (grid_noise look)."""
    if len(x) == 0:
        return
    ax.plot(x, y, marker + "-", color=color, lw=2.2, ms=7, label=label, zorder=4)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.13, lw=0, zorder=1)


def shadow(ax, x, y, color, label):
    """exact = faint dashed shadow (rides under the coin line where they agree)."""
    if len(x) == 0:
        return
    ax.plot(x, y, "--", color=color, lw=1.0, alpha=0.45, label=label, zorder=3)


# ── GAIL demonstration returns (printed on the GAIL coin points, as in grid_noise) ──
_NONFS = "/scratch/marzii/imitation_runs/noisy_demos_online_nonFS/lunarlander/expert_4615187"
_CLEAN = "/scratch/marzii/imitation_runs/noisy_demos/lunarlander/expert_4615187/n100_p0_clean"
_COIN_EXCLUDE = "/scratch/marzii/imitation_runs/noise_variants/demos/coin_exclude"


def _pool_return(d):
    rets = []
    for arrow in glob.glob(str(d) + "/*.arrow"):
        try:
            with pyarrow.ipc.open_stream(arrow) as r:
                tbl = r.read_all()
            for ep in tbl.column("rews").to_pylist():
                rets.append(float(sum(ep)))
        except Exception:
            pass
    return rets


def demo_returns(pool_fn, levels):
    out = {}
    for L in levels:
        rets = []
        for d in pool_fn(L):
            rets += _pool_return(d)
        if rets:
            out[L] = float(np.mean(rets))
    return out


def random_pools(L):
    return [_CLEAN] if L == 0 else [f"{_NONFS}/n100_p{L}_s{s}" for s in range(30)]


def flip_pools(L):
    return [_CLEAN] if L == 0 else [f"{_COIN_EXCLUDE}/n100_p{L}_s{s}" for s in range(30)]


def annotate_demos(ax, x, y, demo_ret, color):
    for xi, yi in zip(x, y):
        dr = demo_ret.get(int(xi))
        if dr is not None:
            ax.annotate(f"{dr:+.0f}", (xi, yi), textcoords="offset points",
                        xytext=(5, -15), fontsize=9, color=color, fontweight="bold",
                        ha="left", zorder=8)


def mark_100(ax, x, y, color):
    """Dashed line to the y-axis at the 100%-noise value (labelled) for cross-N comparison."""
    for xi, yi in zip(x, y):
        if int(xi) == 100:
            ax.axhline(yi, color=color, ls="--", lw=1.1, alpha=0.6)
            ax.annotate(f"{yi:+.0f}", (ax.get_xlim()[0], yi), xytext=(3, 2),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color=color, va="bottom")
            return


def main():
    pt_ms = load_pt_ms()
    gz = load_gail_zero()
    g_coin_uniform = load_gail_coin_uniform()
    gvar = load_gail_variants()
    print(f"PT conditions: {len(pt_ms)} | GAIL 0% seeds: {len(gz)} | "
          f"coin_uniform levels: {len(g_coin_uniform)} | variant modes: {list(gvar)}")
    random_demo_ret = demo_returns(random_pools, LEVELS)
    flip_demo_ret = demo_returns(flip_pools, LEVELS)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Panel 1 — RANDOM-REPLACE  (coin = bold marked line + demo:… ; exact = faint dashed)
    gxL, gyL, geL = gail_series(g_coin_uniform, gz)
    bold_line(axL, gxL, gyL, geL, GAIL_COLOR, "^", "GAIL coin")
    ptL = pt_series(pt_ms, "noise", "clean")
    bold_line(axL, *ptL, PT_COLOR, "s", "PT coin")
    shadow(axL, *pt_series(pt_ms, "exnoise", "exclean")[:2], PT_COLOR, "PT exact")
    shadow(axL, *gail_series(gvar.get("exact_uniform", {}), gz)[:2], GAIL_COLOR, "GAIL exact")
    annotate_demos(axL, gxL, gyL, random_demo_ret, GAIL_COLOR)
    axL.set_title("Random-replace noise")

    # Panel 2 — FLIP  (coin = bold marked line + demo:… ; exact = faint dashed)
    gxR, gyR, geR = gail_series(gvar.get("coin_exclude", {}), gz)
    bold_line(axR, gxR, gyR, geR, GAIL_COLOR, "^", "GAIL coin")
    ptR = pt_series(pt_ms, "flipnoise", "flipclean")
    bold_line(axR, *ptR, PT_COLOR, "s", "PT coin")
    shadow(axR, *pt_series(pt_ms, "exflipnoise", "exflipclean")[:2], PT_COLOR, "PT exact")
    shadow(axR, *gail_series(gvar.get("exact_exclude", {}), gz)[:2], GAIL_COLOR, "GAIL exact")
    annotate_demos(axR, gxR, gyR, flip_demo_ret, GAIL_COLOR)
    axR.set_title("Flip noise")

    for ax in (axL, axR):
        ax.axhline(0, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("Noise (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 103)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    axL.set_ylabel("Mean eval reward (30 seeds)")
    axL.set_ylim(-700, 320)
    # 100%-noise dashed lines (coin curves) for cross-N comparison
    mark_100(axL, gxL, gyL, GAIL_COLOR); mark_100(axL, ptL[0], ptL[1], PT_COLOR)
    mark_100(axR, gxR, gyR, GAIL_COLOR); mark_100(axR, ptR[0], ptR[1], PT_COLOR)
    fig.suptitle("Noise injection: PT (N=1000) vs GAIL (N=100) — bold = coin (■ PT, ▲ GAIL), faint dashed = exact  "
                 "(30 seeds, nonFS expert 4615187; demo:… = GAIL demo return)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "noise_variants_pt_vs_gail.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
