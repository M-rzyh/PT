"""Plot IQL learning curves for Run A (PT-oracle, 3 seeds) and Run B (PT-human, 1 seed)."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
ORACLE_BASE = SCRATCH / "PT/lunarlander/iql_runs/oracle100"
HUMAN_BASE = SCRATCH / "PT/lunarlander/iql_runs/human100"
PEBBLE_BASE = SCRATCH / "PT/lunarlander/iql_runs/pebble100"
OUT = Path("/home/marzii/PT/PreferenceTransformer/figures/pt_iql_lunarlander_medium_v2.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def find_progress(base: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        m = sorted(base.glob(pat))
        if m:
            return m[-1]
    return None


def load(seed_dir: Path, env_tag: str, comment: str) -> tuple[np.ndarray, np.ndarray]:
    seed_num = seed_dir.name.split('_')[1]
    pat = f"tb/{env_tag}/reward_True_PrefTransformer/{comment}/{seed_num}/*/progress.txt"
    p = find_progress(seed_dir, [pat])
    if p is None:
        raise FileNotFoundError(f"no progress.txt under {seed_dir} matching {pat}")
    data = np.loadtxt(p)
    return data[:, 0], data[:, 1]


def smooth(y: np.ndarray, window: int = 10) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
ax.grid(True, color="#cbd5e1", linewidth=0.5, alpha=0.6)

# --- Run A: 3 oracle seeds (mean ± std band + thin per-seed lines) -----------
oracle_curves = []
common_x = None
for s in (0, 1, 2):
    seed_dir = ORACLE_BASE / f"seed_{s}"
    x, y = load(seed_dir, env_tag=f"lunarlander-medium-v2-oracle-s{s}", comment="oracle100")
    oracle_curves.append(y)
    if common_x is None:
        common_x = x
    ax.plot(x, smooth(y), color="#2563eb", alpha=0.25, lw=1)

oracle_arr = np.stack(oracle_curves)
mean = smooth(oracle_arr.mean(axis=0))
std = smooth(oracle_arr.std(axis=0))
ax.fill_between(common_x, mean - std, mean + std, color="#2563eb", alpha=0.18)
ax.plot(common_x, mean, color="#2563eb", lw=2.5,
        label="PT-oracle (3 seeds, mean ± std)")

# --- Run B: 1 human seed -----------------------------------------------------
seed_dir = HUMAN_BASE / "seed_0"
xh, yh = load(seed_dir, env_tag="lunarlander-medium-v2-human-s0", comment="human100")
ax.plot(xh, smooth(yh), color="#7c3aed", lw=2.5,
        label="PT-human (seed 0, 100 web labels)")

# --- Run C: 1 seed, PT trained on PEBBLE's 100 labels ------------------------
seed_dir = PEBBLE_BASE / "seed_0"
xp, yp = load(seed_dir, env_tag="lunarlander-pebble100-s0", comment="pebble100")
ax.plot(xp, smooth(yp), color="#dc2626", lw=2.5,
        label="PT-PEBBLE-labels (seed 0, 25 oracle + 75 human)")

# --- PEBBLE's own eval.csv (the source of those labels) -----------------------
peb_csv = "/scratch/marzii/compare_runs/pebble/lunarlander_web_full/4895573/seed_12345/pebble/eval.csv"
peb = np.genfromtxt(peb_csv, delimiter=",", names=True)
ps = peb["step"]; pr = peb["true_episode_reward"]
order = np.argsort(ps)
ps, pr = ps[order], pr[order]
ax.plot(ps, pr, color="#0f766e", lw=2.0, marker="o", markersize=8,
        label=f"PEBBLE (job 4895573, eval.csv, {len(ps)} points)")

# --- Annotations -------------------------------------------------------------
ax.axhline(137.0, color="#b45309", lw=1.2, ls="--", alpha=0.7,
           label="medium-v2 dataset ep-return mean (+137)")
ax.axhline(265.0, color="#15803d", lw=1.0, ls=":", alpha=0.7,
           label="approx. expert return (+265)")

ax.set_xlim(0, 1_000_000)
ax.set_xlabel("IQL gradient step")
ax.set_ylabel("Online eval return (10 ep)")
ax.set_title(
    "Preference Transformer on LunarLanderContinuous-v2 (medium dataset, 100 queries)",
    fontsize=12,
)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
