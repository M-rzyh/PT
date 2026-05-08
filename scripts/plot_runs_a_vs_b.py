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
OUT = Path("/home/marzii/PT/PreferenceTransformer/figures/pt_iql_lunarlander_medium_v2.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def find_progress(base: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        m = sorted(base.glob(pat))
        if m:
            return m[-1]
    return None


def load(seed_dir: Path, run_tag: str, comment: str) -> tuple[np.ndarray, np.ndarray]:
    pat = (
        f"tb/lunarlander-medium-v2-{run_tag}/reward_True_PrefTransformer/{comment}/"
        f"{seed_dir.name.split('_')[1]}/*/progress.txt"
    )
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
ax.set_facecolor("#0f172a")
fig.patch.set_facecolor("#0f172a")
for spine in ax.spines.values():
    spine.set_color("#475569")
ax.tick_params(colors="#cbd5e1")
ax.grid(True, color="#334155", linewidth=0.5, alpha=0.6)

# --- Run A: 3 oracle seeds (mean ± std band + thin per-seed lines) -----------
oracle_curves = []
common_x = None
for s in (0, 1, 2):
    seed_dir = ORACLE_BASE / f"seed_{s}"
    x, y = load(seed_dir, run_tag=f"oracle-s{s}", comment="oracle100")
    oracle_curves.append(y)
    if common_x is None:
        common_x = x
    ax.plot(x, smooth(y), color="#60a5fa", alpha=0.30, lw=1)

oracle_arr = np.stack(oracle_curves)
mean = smooth(oracle_arr.mean(axis=0))
std = smooth(oracle_arr.std(axis=0))
ax.fill_between(common_x, mean - std, mean + std, color="#60a5fa", alpha=0.15)
ax.plot(common_x, mean, color="#60a5fa", lw=2.5,
        label="PT-oracle (3 seeds, mean ± std)")

# --- Run B: 1 human seed -----------------------------------------------------
seed_dir = HUMAN_BASE / "seed_0"
xh, yh = load(seed_dir, run_tag="human-s0", comment="human100")
ax.plot(xh, smooth(yh), color="#c084fc", lw=2.5,
        label="PT-human (seed 0, 100 web labels)")

# --- Annotations -------------------------------------------------------------
# Dataset baseline: medium-v2's mean episode return (137 from earlier inspection)
ax.axhline(137.0, color="#facc15", lw=1.2, ls="--", alpha=0.6,
           label="medium-v2 dataset ep-return mean (+137)")
# Oracle ceiling: ~+260 (expert) for reference
ax.axhline(265.0, color="#22c55e", lw=1.0, ls=":", alpha=0.5,
           label="approx. expert return (+265)")

ax.set_xlim(0, 1_000_000)
ax.set_xlabel("IQL gradient step", color="#e2e8f0")
ax.set_ylabel("Online eval return (10 ep)", color="#e2e8f0")
ax.set_title(
    "Preference Transformer on LunarLanderContinuous-v2 (medium dataset, 100 queries)",
    color="#f8fafc", fontsize=12,
)
leg = ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155",
                labelcolor="#e2e8f0", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
