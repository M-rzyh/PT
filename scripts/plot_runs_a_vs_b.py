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
PEBBLE_RUN_TB = ("/scratch/marzii/compare_runs/pebble/lunarlander_web_full/"
                  "4895573/seed_12345/pebble/tb")
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


# --- gather ---------------------------------------------------------------
oracle_curves, common_x = [], None
for s in (0, 1, 2):
    x, y = load_pt(ORACLE_BASE / f"seed_{s}",
                    env_tag=f"lunarlander-medium-v2-oracle-s{s}", comment="oracle100")
    oracle_curves.append(y)
    if common_x is None: common_x = x
oracle_arr = np.stack(oracle_curves)

xh, yh = load_pt(HUMAN_BASE / "seed_0",
                  env_tag="lunarlander-medium-v2-human-s0", comment="human100")
xp, yp = load_pt(PEBBLE_PT_BASE / "seed_0",
                  env_tag="lunarlander-pebble100-s0", comment="pebble100")

ps, pv = load_pebble_curve(PEBBLE_RUN_TB)
# Heavy smoothing for PEBBLE's per-episode online training rewards.
pv_sm = smooth(pv, window=200)

# --- plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# PT-oracle: mean ± std band
mean = smooth(oracle_arr.mean(axis=0), window=10)
std  = smooth(oracle_arr.std(axis=0),  window=10)
last10_m = oracle_arr[:, int(oracle_arr.shape[1]*0.9):].mean()
last10_s = oracle_arr[:, int(oracle_arr.shape[1]*0.9):].mean(axis=1).std()
ax.plot(common_x, mean, color=COLORS[0], linewidth=1.8,
        label=f"PT-oracle (n=3 seeds, last10%={last10_m:.1f}±{last10_s:.1f})")
ax.fill_between(common_x, mean - std, mean + std, color=COLORS[0], alpha=0.18)

# PT-human (1 seed)
yh_sm = smooth(yh, window=10)
ax.plot(xh, yh_sm, color=COLORS[3], linewidth=1.8,
        label=f"PT-human (n=1 seed, last10%={last10_str(xh, yh)})")

# PT-PEBBLE-labels (1 seed)
yp_sm = smooth(yp, window=10)
ax.plot(xp, yp_sm, color=COLORS[1], linewidth=1.8,
        label=f"PT-PEBBLE-labels (n=1 seed, last10%={last10_str(xp, yp)})")

# PEBBLE itself
ax.plot(ps, pv_sm, color=COLORS[2], linewidth=1.8,
        label=f"PEBBLE (job 4895573, train/true_ep_rew, last10%={last10_str(ps, pv)})")

# Reference dataset return (medium-v2 ep mean)
ax.axhline(137.0, color="grey", lw=1.0, ls="--", alpha=0.6,
           label="medium-v2 dataset ep-return mean (+137)")

ax.set_xlabel("Steps")
ax.set_ylabel("Episode Reward (ep_rew_mean)")
ax.set_title("PT vs PEBBLE on LunarLanderContinuous-v2 (medium dataset, 100 preference labels)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
