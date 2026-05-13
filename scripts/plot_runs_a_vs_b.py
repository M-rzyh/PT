"""Overlay PT IQL learning curves (Runs A/B/C) with PEBBLE's own learning curve.

Plot style follows /home/marzii/IRL3/scripts/plot_gail_multi_seed.py:
    figsize=(11, 6), dpi=150, default-matplotlib colors,
    linewidth=1.8, fill_between alpha=0.18, grid alpha=0.3,
    legend lower-right fontsize=9, last10% summary in each legend label.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
ORACLE_BASE = SCRATCH / "PT/lunarlander/iql_runs/oracle100"
HUMAN_BASE = SCRATCH / "PT/lunarlander/iql_runs/human100"
PEBBLE_PT_BASE = SCRATCH / "PT/lunarlander/iql_runs/pebble100"

# PEBBLE web run: stitch original 4895573 (steps 0..~74K = preference-based
# PEBBLE training) + resumed 4895915 (steps 70K..1M = pure online RL).
PEBBLE_WEB_TB_PARTS = [
    "/scratch/marzii/compare_runs/pebble/lunarlander_web_full/4895573/seed_12345/pebble/tb",
    "/scratch/marzii/compare_runs/pebble/lunarlander_web_full_compare/4895573/seed_12345/pebble/tb",
]

# PEBBLE-oracle multi-seed (max_feedback=100, scripted teacher).
# Only seeds {370, 371, 373} reached 1M steps; 372 and 374 stopped at
# 250K and 10K respectively, so they're excluded from the band.
PEBBLE_ORACLE_TB_DIRS = [
    f"/scratch/marzii/compare_runs/pebble/lunarlander/{j}/pebble/tb"
    for j in (4883370, 4883371, 4883373)
]

OUT = Path("/home/marzii/PT/PreferenceTransformer/figures/pt_iql_lunarlander_medium_v2.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def find_progress(base: Path, pat: str) -> Path | None:
    m = sorted(base.glob(pat))
    return m[-1] if m else None


def load_pt(seed_dir: Path, env_tag: str, comment: str) -> tuple[np.ndarray, np.ndarray]:
    seed_num = seed_dir.name.split('_')[1]
    pat = f"tb/{env_tag}/reward_True_PrefTransformer/{comment}/{seed_num}/*/progress.txt"
    p = find_progress(seed_dir, pat)
    if p is None:
        raise FileNotFoundError(f"no progress.txt under {seed_dir} matching {pat}")
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1]


def load_pebble_curve(tb_dir: str, tag: str = "train/true_episode_reward"
                      ) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(tb_dir); ea.Reload()
    evs = ea.Scalars(tag)
    return (np.array([e.step for e in evs]),
            np.array([e.value for e in evs]))


def stitch_pebble_curves(tb_dirs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Concat episodes from multiple TBs sorted by step, deduplicating
    overlaps so each step shows up at most once."""
    all_s, all_v = [], []
    for d in tb_dirs:
        s, v = load_pebble_curve(d)
        all_s.append(s); all_v.append(v)
    steps = np.concatenate(all_s); vals = np.concatenate(all_v)
    order = np.argsort(steps, kind="stable")
    steps, vals = steps[order], vals[order]
    _, keep = np.unique(steps, return_index=True)
    return steps[keep], vals[keep]


def aggregate_seeds_on_common_x(tb_dirs: list[str], common_x: np.ndarray,
                                tag: str = "train/true_episode_reward"
                                ) -> tuple[np.ndarray, list[str]]:
    """Per-seed: smooth + interpolate onto `common_x`. Returns (n_seeds,len)."""
    rows = []
    used = []
    for d in tb_dirs:
        s, v = load_pebble_curve(d, tag)
        if len(s) < 50: continue
        v_sm = smooth(v, window=100)
        rows.append(np.interp(common_x, s, v_sm,
                              left=np.nan, right=np.nan))
        used.append(d)
    return np.stack(rows), used


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window:
        return y
    k = np.ones(window) / window
    return np.convolve(y, k, mode="same")


def last10_str(steps: np.ndarray, vals: np.ndarray) -> str:
    if len(vals) == 0:
        return ""
    i = int(len(vals) * 0.9)
    seg = vals[i:]
    return f"{seg.mean():.1f}±{seg.std():.1f}"


def gather(base: Path, env_tag_tpl: str, comment: str, n_seeds: int = 5):
    curves, common_x = [], None
    for s in range(n_seeds):
        x, y = load_pt(base / f"seed_{s}", env_tag=env_tag_tpl.format(s=s),
                       comment=comment)
        curves.append(y)
        if common_x is None: common_x = x
    return common_x, np.stack(curves)


def plot_band(ax, x, arr, color, label_prefix):
    mean = smooth(arr.mean(axis=0), window=10)
    std  = smooth(arr.std(axis=0),  window=10)
    i = int(arr.shape[1] * 0.9)
    last10_per_seed = arr[:, i:].mean(axis=1)
    m, sd = last10_per_seed.mean(), last10_per_seed.std()
    ax.plot(x, mean, color=color, linewidth=1.8,
            label=f"{label_prefix} (n={arr.shape[0]} seeds, last10%={m:.1f}±{sd:.1f})")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


# --- gather ---------------------------------------------------------------
x_o, oracle = gather(ORACLE_BASE, "lunarlander-medium-v2-oracle-s{s}", "oracle100")
x_h, human  = gather(HUMAN_BASE,  "lunarlander-medium-v2-human-s{s}",  "human100")
x_p, pebble = gather(PEBBLE_PT_BASE, "lunarlander-pebble100-s{s}",     "pebble100")

ps, pv = stitch_pebble_curves(PEBBLE_WEB_TB_PARTS)
pv_sm = smooth(pv, window=200)  # PEBBLE's per-episode noisy curve

# PEBBLE-oracle 5-seed band (3 of 5 seeds completed).
common_x_oracle = np.linspace(0, 1_000_000, 200)
oracle_arr_p, _used = aggregate_seeds_on_common_x(PEBBLE_ORACLE_TB_DIRS, common_x_oracle)

# --- plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
plot_band(ax, x_o, oracle, COLORS[0], "PT-oracle")
plot_band(ax, x_h, human,  COLORS[3], "PT-human")
plot_band(ax, x_p, pebble, COLORS[1], "PT-PEBBLE-labels")

# PEBBLE web (single seed): 4895573 [0..74K] stitched with 4895915 [70K..1M]
ax.plot(ps, pv_sm, color=COLORS[2], linewidth=1.8,
        label=f"PEBBLE-human-web (4895573+4895915, train/true_ep_rew, last10%={last10_str(ps, pv)})")

# PEBBLE-oracle band (max_feedback=100, scripted teacher). Only the seeds
# that reached 1M are aggregated. nan-aware mean/std so partial seeds (none
# here) don't crash the plot.
mean_o = np.nanmean(oracle_arr_p, axis=0)
std_o  = np.nanstd(oracle_arr_p,  axis=0)
i = int(oracle_arr_p.shape[1] * 0.9)
last10_per = np.nanmean(oracle_arr_p[:, i:], axis=1)
m_o, sd_o = float(np.nanmean(last10_per)), float(np.nanstd(last10_per))
ax.plot(common_x_oracle, mean_o, color=COLORS[4], linewidth=1.8,
        label=f"PEBBLE-oracle (n={oracle_arr_p.shape[0]} seeds, "
              f"max_feedback=100, last10%={m_o:.1f}±{sd_o:.1f})")
ax.fill_between(common_x_oracle, mean_o - std_o, mean_o + std_o,
                color=COLORS[4], alpha=0.18)

# Reference dataset return (medium-v2 ep mean)
ax.axhline(137.0, color="grey", lw=1.0, ls="--", alpha=0.6,
           label="medium-v2 dataset ep-return mean (+137)")

ax.set_xlabel("Steps")
ax.set_ylabel("Episode Reward (ep_rew_mean)")
ax.set_title("PT vs PEBBLE on LunarLanderContinuous-v2 (medium dataset, 100 preference labels)")
ax.set_ylim(-600, 350)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
