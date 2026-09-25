"""Learning curves at 100% noise: PT (N=100 preferences) vs GAIL (N=15 demos), 30 seeds each.

x = fraction of each method's own training budget (PT: 1e6 IQL gradient steps, offline;
    GAIL: 1e6 env timesteps, online)
y = ground-truth env reward, mean across the 30 seeds; shaded = +/- 1 std across seeds
    PT   : progress.txt col1 (eval, 10 episodes every 5k steps)
    GAIL : TB mean/gen/rollout/ep_rew_mean (generator rollouts; NOT ep_rew_wrapped_mean)
Dash-dot = uniformly-random-policy floor (100 eps, 1000-step cap) in each method's env.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/plots/plot_learning_curve_noise100.py
"""
import csv, glob
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT_DIR = "/scratch/marzii/PT/lunarlander/grid_mixture_ms/lunarlander-grid-ms-N100-noise100"
GAIL_IDX = "/home/marzii/IRL3/experiments/GAIL/gail_grid_noise_N15_2026-07-23.csv"
GAIL_BASE = "/scratch/marzii/imitation_runs/gail/lunarlander"
FIG = "/home/marzii/PT/PreferenceTransformer/figures"
PT_C, GA_C, CLEAN_C = "#1f77b4", "#9467bd", "#2ca02c"
GRID = "/scratch/marzii/PT/lunarlander/grid_mixture_ms"


def random_floor(env_id, n_eps=100, cap=1000, seed=0):
    """Uniformly-random-action policy (no training, no reward, no dataset) in `env_id`.
    Returns the mean true-env return over n_eps episodes. Same routine as
    scripts/plots/plot_noise_N100_N15_random.py."""
    env = gym.make(env_id, max_episode_steps=cap)
    env.action_space.seed(seed)      # action sampling has its OWN rng; seed it or the
    rets = []                        # floor shifts a few points run-to-run
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            R += r; done = term or trunc
        rets.append(R)
    env.close()
    return float(np.mean(rets))


def pt_cond(cond):
    """Any PT condition dir -> per-seed (steps, reward). Picks the progress.txt with the MOST
    rows in each seed, because a re-run writes a NEW timestamped file beside the truncated one."""
    xs, ys = [], []
    for d in sorted(glob.glob(f"{GRID}/{cond}/seed_*")):
        files = glob.glob(f"{d}/**/progress.txt", recursive=True)
        if not files:
            continue
        a = np.loadtxt(max(files, key=lambda f: sum(1 for _ in open(f))))
        if a.ndim == 1:
            a = a.reshape(1, -1)
        xs.append(a[:, 0]); ys.append(a[:, 1])
    return xs, ys


def pt_curves():
    xs, ys = [], []
    for d in sorted(glob.glob(f"{PT_DIR}/seed_*")):
        p = glob.glob(f"{d}/**/progress.txt", recursive=True)
        if not p:
            continue
        a = np.loadtxt(p[-1])
        if a.ndim == 1:
            a = a.reshape(1, -1)
        xs.append(a[:, 0]); ys.append(a[:, 1])
    return xs, ys


def gail_curves():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    jobs = [r["slurm_job_id"] for r in csv.DictReader(open(GAIL_IDX)) if r["noise_pct"] == "100"]
    xs, ys = [], []
    for j in jobs:
        f = glob.glob(f"{GAIL_BASE}/{j}/log/events.out.tfevents*")
        if not f:
            continue
        ea = EventAccumulator(f[0], size_guidance={'scalars': 0}); ea.Reload()
        if "mean/gen/rollout/ep_rew_mean" not in ea.Tags()['scalars']:
            continue
        s = ea.Scalars("mean/gen/rollout/ep_rew_mean")
        # TB 'step' is the GAIL ROUND index (0..488), not env timesteps. 488 rounds x 2048
        # steps/round = the 1e6 total_timesteps the run was configured for -> rescale.
        st = np.array([e.step for e in s], float)
        xs.append(st / st.max() * 1e6)
        ys.append(np.array([e.value for e in s], float))
    return xs, ys


def band(xs, ys, n=200):
    """Interpolate every seed onto a common grid of ABSOLUTE training steps."""
    lo = max(x.min() for x in xs); hi = min(x.max() for x in xs)
    g = np.linspace(lo, hi, n)
    M = np.vstack([np.interp(g, x, y) for x, y in zip(xs, ys)])
    return g, M.mean(0), M.std(0), len(xs)


def smooth(v, k=9):
    pad = np.r_[np.full(k // 2, v[0]), v, np.full(k // 2, v[-1])]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    pg, pm, ps, pn = band(*pt_curves())
    gg, gm, gs, gn = band(*gail_curves())
    PT_RAND = random_floor("LunarLanderContinuous-v2")
    GA_RAND = random_floor("LunarLander-v2")
    print(f"random floors -> continuous {PT_RAND:+.1f} | discrete {GA_RAND:+.1f}")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    # ---- GAIL @ 100% noise ----
    gms, gss = smooth(gm), smooth(gs)
    ax.fill_between(gg, gms - gss, gms + gss, color=GA_C, alpha=0.13, lw=0, zorder=1)
    ax.plot(gg, gms, color=GA_C, lw=2.3, label=f"GAIL @ 100% noise (N=15 demos)", zorder=4)

    # ---- PT @ 100% noise: both evaluation intervals ----
    for cond, ls, lw, lbl in [("lunarlander-grid-ms-N100-noise100",        ":", 1.6, "PT 100% noise, eval/5000"),
                              ("lunarlander-grid-ms-N100-noise100-ev1000", "-", 2.3, "PT 100% noise, eval/1000")]:
        nx, ny = pt_cond(cond)
        if not nx:
            continue
        ng, nm, ns, nn = band(nx, ny)
        nms, nss = smooth(nm), smooth(ns)
        if ls == "-":
            ax.fill_between(ng, nms - nss, nms + nss, color=PT_C, alpha=0.13, lw=0, zorder=1)
        ax.plot(ng, nms, ls, color=PT_C, lw=lw, label=f"{lbl} ({nn} seeds)", zorder=4)

    # ---- PT with CLEAN labels (0% noise) -- the upper reference ----
    for cond, ls, lw, lbl in [("lunarlander-grid-ms-N100-clean",        ":", 1.6, "PT clean, eval/5000"),
                              ("lunarlander-grid-ms-N100-clean-ev1000", "-", 2.3, "PT clean, eval/1000")]:
        cx, cy = pt_cond(cond)
        if not cx:
            continue
        cg, cm, cs, cn = band(cx, cy)
        cms, css = smooth(cm), smooth(cs)
        if ls == "-":
            ax.fill_between(cg, cms - css, cms + css, color=CLEAN_C, alpha=0.12, lw=0, zorder=1)
        ax.plot(cg, cms, ls, color=CLEAN_C, lw=lw, label=f"{lbl} (0% noise, {cn} seeds)", zorder=4)

    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.set_xlim(0, 1e6)

    # random-policy floors: line inside, label plainly OUTSIDE on the right
    for yv, c, name in [(GA_RAND, GA_C, "random policy discrete"),
                        (PT_RAND, PT_C, "random policy continuous")]:
        ax.axhline(yv, color=c, ls="-.", lw=1.6, alpha=0.9, zorder=3)
        ax.text(1.015e6, yv, f"{name}  {yv:+.0f}", color=c, fontsize=9, va="center", ha="left",
                clip_on=False)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground truth reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Learning curves — PT vs GAIL at 100% noise, with clean PT as reference (30 seeds)", fontsize=12.5)
    fig.subplots_adjust(right=0.78)
    out = f"{FIG}/learning_curve_noise100_pt_vs_gail.png"
    fig.savefig(out, dpi=150); print("Saved:", out)


if __name__ == "__main__":
    main()
