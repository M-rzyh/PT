"""Mixture-v2 version of the comparison plot.

Same layout as plot_runs_a_vs_b.py / plot_runs_medium_replay.py:
3 PT bands + PEBBLE-oracle band + PEBBLE-human-web line + dataset baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
ORACLE_BASE = SCRATCH / "PT/lunarlander/iql_runs/oracle100_mixture"
HUMAN_BASE  = SCRATCH / "PT/lunarlander/iql_runs/human100_mixture"
PEBBLE_BASE = SCRATCH / "PT/lunarlander/iql_runs/pebble100_mixture"
PEBBLE_WEB_TB_PARTS = [
    "/scratch/marzii/compare_runs/pebble/lunarlander_web_full/4895573/seed_12345/pebble/tb",
    "/scratch/marzii/compare_runs/pebble/lunarlander_web_full_compare/4895573/seed_12345/pebble/tb",
]
PEBBLE_ORACLE_TB_DIRS = [
    f"/scratch/marzii/compare_runs/pebble/lunarlander/{j}/pebble/tb"
    for j in (4883370, 4883371, 4883373)
]
OUT = Path("/home/marzii/PT/PreferenceTransformer/figures/pt_iql_lunarlander_mixture_v2_no_pebblelabels.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_pt(seed_dir: Path, env_tag: str, comment: str):
    seed_num = seed_dir.name.split('_')[1]
    pat = f"tb/{env_tag}/reward_True_PrefTransformer/{comment}/{seed_num}/*/progress.txt"
    m = sorted(seed_dir.glob(pat))
    if not m: raise FileNotFoundError(f"{seed_dir} / {pat}")
    d = np.loadtxt(m[-1])
    return d[:, 0], d[:, 1]


def gather(base, tpl, comment, n=5):
    rows, common = [], None
    for s in range(n):
        x, y = load_pt(base / f"seed_{s}", tpl.format(s=s), comment)
        rows.append(y)
        if common is None: common = x
    return common, np.stack(rows)


def smooth(y, w=10):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode="same")


def smooth_pt(y, w=5):
    from scipy.ndimage import uniform_filter1d
    if len(y) < w: return y
    return uniform_filter1d(np.asarray(y, dtype=float), size=w, mode='nearest')


def plot_band(ax, x, arr, color, label_prefix, apply_smooth=True):
    mean = smooth_pt(arr.mean(axis=0)) if apply_smooth else arr.mean(axis=0)
    std  = smooth_pt(arr.std(axis=0))  if apply_smooth else arr.std(axis=0)
    i = int(arr.shape[1] * 0.9)
    last10 = arr[:, i:].mean(axis=1)
    m, sd = last10.mean(), last10.std()
    ax.plot(x, mean, color=color, linewidth=1.8,
            label=f"{label_prefix} (n={arr.shape[0]}, last10%={m:.1f}±{sd:.1f})")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def load_pebble_curve(tb_dir, tag="train/true_episode_reward"):
    ea = EventAccumulator(tb_dir); ea.Reload()
    evs = ea.Scalars(tag)
    return (np.array([e.step for e in evs]),
            np.array([e.value for e in evs]))


def stitch_pebble_curves(tb_dirs):
    all_s, all_v = [], []
    for d in tb_dirs:
        s, v = load_pebble_curve(d)
        all_s.append(s); all_v.append(v)
    s = np.concatenate(all_s); v = np.concatenate(all_v)
    order = np.argsort(s, kind="stable")
    s, v = s[order], v[order]
    _, keep = np.unique(s, return_index=True)
    return s[keep], v[keep]


def aggregate_seeds_on_common_x(tb_dirs, common_x):
    rows = []
    for d in tb_dirs:
        s, v = load_pebble_curve(d)
        if len(s) < 50: continue
        rows.append(np.interp(common_x, s, smooth(v, 100), left=np.nan, right=np.nan))
    return np.stack(rows)


def last10_str(steps, vals):
    if len(vals) == 0: return ""
    i = int(len(vals) * 0.9)
    seg = vals[i:]
    return f"{seg.mean():.1f}±{seg.std():.1f}"


x_o, oracle = gather(ORACLE_BASE, "lunarlander-mixture-v2-oracle-s{s}", "oracle100")
x_h, human  = gather(HUMAN_BASE,  "lunarlander-mixture-v2-human-s{s}",  "human100")

ps, pv = stitch_pebble_curves(PEBBLE_WEB_TB_PARTS)
pv_sm = smooth(pv, 200)
common_x_p = np.linspace(0, 1_000_000, 200)
oracle_arr_p = aggregate_seeds_on_common_x(PEBBLE_ORACLE_TB_DIRS, common_x_p)

import h5py
with h5py.File(SCRATCH / "PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5", "r") as f:
    r = f["rewards"][:]; ends = (f["terminals"][:] | f["timeouts"][:])
    cur = 0.0; ep_returns = []
    for i in range(len(r)):
        cur += float(r[i])
        if ends[i]: ep_returns.append(cur); cur = 0.0
ds_mean = float(np.mean(ep_returns)) if ep_returns else float("nan")

fig, ax = plt.subplots(figsize=(11, 6))
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
plot_band(ax, x_o, oracle, COLORS[0], "PT-oracle", apply_smooth=True)
plot_band(ax, x_h, human,  COLORS[3], "PT-human",  apply_smooth=True)
ax.plot(ps, pv_sm, color=COLORS[2], linewidth=1.8,
        label=f"PEBBLE-human-web (4895573+4895915, last10%={last10_str(ps, pv)})")
mean_o = np.nanmean(oracle_arr_p, axis=0)
std_o  = np.nanstd(oracle_arr_p,  axis=0)
i = int(oracle_arr_p.shape[1] * 0.9)
last10 = np.nanmean(oracle_arr_p[:, i:], axis=1)
m_o, sd_o = float(np.nanmean(last10)), float(np.nanstd(last10))
ax.plot(common_x_p, mean_o, color=COLORS[4], linewidth=1.8,
        label=f"PEBBLE-oracle (n={oracle_arr_p.shape[0]} seeds, max_feedback=100, last10%={m_o:.1f}±{sd_o:.1f})")
ax.fill_between(common_x_p, mean_o - std_o, mean_o + std_o, color=COLORS[4], alpha=0.18)

ax.axhline(ds_mean, color="grey", lw=1.0, ls="--", alpha=0.6,
           label=f"mixture dataset ep-return mean ({ds_mean:+.1f})")
ax.set_xlabel("Steps")
ax.set_ylabel("Episode Reward (ep_rew_mean)")
ax.set_title("PT vs PEBBLE on LunarLanderContinuous-v2 — MIXTURE dataset (100 labels)")
ax.set_ylim(-600, 350)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
