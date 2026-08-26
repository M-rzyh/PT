"""noise_N100_N15 plot + RANDOM-POLICY floor lines.

Reuses plot_noise_N100_N15's exact data loaders, then overlays two horizontal reference lines:
the mean return of a uniformly-RANDOM policy in each method's env (100 episodes, cap 1000) —
  PT   floor: random policy in LunarLanderContinuous-v2 (random thrust)
  GAIL floor: random policy in LunarLander-v2      (random 1-of-4 action)
so you can see whether the 100%-noise points bottom out at random supervision.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_noise_N100_N15_random.py
"""
import os, sys
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_noise_N100_N15 import pt_series, gail_series, demo_returns, curve, PT_C, GA_C, FIG


def random_floor(env_id, n_eps=100, cap=1000, seed=0):
    env = gym.make(env_id, max_episode_steps=cap)
    rets = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            R += r; done = term or trunc
        rets.append(R)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def main():
    pt, ga, dr = pt_series(), gail_series(), demo_returns()
    pt_rand = random_floor("LunarLanderContinuous-v2")   # PT env   Main lines
    ga_rand = random_floor("LunarLander-v2")             # GAIL env Main lines
    print(f"\nRANDOM floor  PT(continuous): {pt_rand[0]:+.1f} ± {pt_rand[1]:.1f}")
    print(f"RANDOM floor  GAIL(discrete): {ga_rand[0]:+.1f} ± {ga_rand[1]:.1f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    gx, gy = curve(ax, ga, GA_C, "^", "GAIL noise (N=15 demos, 30 seeds, nonFS expert 4615187)")
    curve(ax, pt, PT_C, "s", "PT noise (N=100 preferences, 30 seeds, per-seed corrupted pairs)")
    for xi, yi in zip(gx, gy):
        v = dr.get(int(xi))
        if v is not None:
            ax.annotate(f"{v:+.0f}", (xi, yi), textcoords="offset points", xytext=(5, -15),
                        fontsize=9, color=GA_C, fontweight="bold", zorder=8)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)

    # ---- RANDOM-POLICY floor lines ----
    for (m, sd), c, lbl in [(pt_rand, PT_C, "PT env"), (ga_rand, GA_C, "GAIL env")]:
        ax.axhline(m, color=c, ls="-.", lw=1.6, alpha=0.9, zorder=3)
        ax.annotate(f"random ({lbl}) {m:+.0f}", (103, m), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8.5, fontweight="bold", color=c,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, alpha=0.85))

    ax.set_xlabel("Noise (%)"); ax.set_ylabel("Mean eval reward (30 seeds)")
    ax.grid(True, alpha=0.3); ax.set_xlim(-3, 103); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Noise axis — PT (N=100) vs GAIL (N=15), coin random-replace\n"
                 "dash-dot = uniformly-random-policy floor (100 eps) in each method's env",
                 fontsize=11)
    fig.tight_layout()
    out = f"{FIG}/noise_N100_N15_pt_vs_gail_random.png"
    fig.savefig(out, dpi=150); print("Saved:", out)


if __name__ == "__main__":
    main()
