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

ps, pv = load_pebble_curve(PEBBLE_RUN_TB)
pv_sm = smooth(pv, window=200)  # PEBBLE's per-episode noisy curve

# --- plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
plot_band(ax, x_o, oracle, COLORS[0], "PT-oracle")
plot_band(ax, x_h, human,  COLORS[3], "PT-human")
plot_band(ax, x_p, pebble, COLORS[1], "PT-PEBBLE-labels")

# PEBBLE itself (single seed, dense online-training reward)
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
