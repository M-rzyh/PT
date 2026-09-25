"""PT learning curves: dense eval (every 1000 gradient steps) vs the original (every 5000),
for both 100% noise and clean, 30 seeds each.

y = ground-truth env reward (progress.txt col1), mean over seeds, shaded = +/- 1 std.
NOTE: a re-run writes a NEW timestamped progress.txt next to the old truncated one, so we pick
the file with the MOST rows per seed, never just the first/last glob hit.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_learning_curve_ev1000_vs_ev5000.py
"""
import glob
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
CLEAN_C, NOISE_C = "#2ca02c", "#1f77b4"


def curves(cond):
    """One (steps, rewards) pair per seed, taking the LONGEST progress.txt in each seed dir."""
    xs, ys = [], []
    for d in sorted(glob.glob(f"{G}/{cond}/seed_*")):
        files = glob.glob(f"{d}/**/progress.txt", recursive=True)
        if not files:
            continue
        best = max(files, key=lambda f: sum(1 for _ in open(f)))   # most rows = the complete run
        a = np.loadtxt(best)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        xs.append(a[:, 0]); ys.append(a[:, 1])
    return xs, ys


def band(xs, ys, n=400):
    lo, hi = max(x.min() for x in xs), min(x.max() for x in xs)
    g = np.linspace(lo, hi, n)
    M = np.vstack([np.interp(g, x, y) for x, y in zip(xs, ys)])
    return g, M.mean(0), M.std(0), len(xs)


def smooth(v, k=9):
    pad = np.r_[np.full(k // 2, v[0]), v, np.full(k // 2, v[-1])]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def random_floor(env_id="LunarLanderContinuous-v2", n_eps=100, cap=1000, seed=0):
    env = gym.make(env_id, max_episode_steps=cap); env.action_space.seed(seed)
    rets = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + ep); done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample()); R += r; done = term or trunc
        rets.append(R)
    env.close(); return float(np.mean(rets))


def main():
    specs = [("lunarlander-grid-ms-N100-clean",           CLEAN_C, ":",  "clean, eval/5000 (200 pts)"),
             ("lunarlander-grid-ms-N100-clean-ev1000",    CLEAN_C, "-",  "clean, eval/1000 (1000 pts)"),
             ("lunarlander-grid-ms-N100-noise100",        NOISE_C, ":",  "100% noise, eval/5000 (200 pts)"),
             ("lunarlander-grid-ms-N100-noise100-ev1000", NOISE_C, "-",  "100% noise, eval/1000 (1000 pts)")]

    fig, ax = plt.subplots(figsize=(11, 6))
    for cond, c, ls, lbl in specs:
        xs, ys = curves(cond)
        if not xs:
            print("missing:", cond); continue
        g, m, s, n = band(xs, ys)
        ms, ss = smooth(m), smooth(s)
        if ls == "-":                      # band only on the dense curves, else it is unreadable
            ax.fill_between(g, ms - ss, ms + ss, color=c, alpha=0.12, lw=0, zorder=1)
        ax.plot(g, ms, ls, color=c, lw=2.2 if ls == "-" else 1.6, label=f"{lbl}, {n} seeds", zorder=4)
        print(f"{cond}: {n} seeds, final {m[-1]:+.0f}")

    rf = random_floor(); print(f"random floor (continuous) = {rf:+.1f}")
    ax.axhline(rf, color="gray", ls="-.", lw=1.6)
    ax.text(1.015e6, rf, f"random policy  {rf:+.0f}", color="gray", fontsize=9, va="center", clip_on=False)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)

    ax.set_xlim(0, 1e6)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground truth reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("PT learning curves — dense vs original evaluation interval (30 seeds)", fontsize=12.5)
    fig.subplots_adjust(right=0.8)
    out = f"{FIG}/learning_curve_pt_ev1000_vs_ev5000.png"
    fig.savefig(out, dpi=150); print("Saved:", out)


if __name__ == "__main__":
    main()
