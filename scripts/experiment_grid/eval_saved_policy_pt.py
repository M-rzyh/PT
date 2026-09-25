"""Deterministic post-training evaluation of a saved PT/IQL policy.

Loads <run_dir>/policy/actor.flax (written by train_offline.py --save_policy) and
rolls EPISODES episodes in the real environment with the deterministic policy
(temperature=0), scoring with the TRUE environment reward. Matches the protocol
used by the GAIL and BC runs: fixed per-episode seeds, 50 episodes by default.

The network shape comes from configs/lunarlander_config.py, exactly as in
training, so a mismatch fails loudly at load rather than silently.

Run from the repo root with the pt env, e.g.:

  python -m scripts.experiment_grid.eval_saved_policy_pt \
      --runs 'lunarlander-human-count-v2-N*-policy' --seeds 10 --episodes 50

  # a single run dir:
  python -m scripts.experiment_grid.eval_saved_policy_pt \
      --run_dir /scratch/marzii/PT/lunarlander/grid_mixture_ms/<cond>/seed_0
"""
import argparse
import json
from pathlib import Path

import gym
import numpy as np

import wrappers
from learner import Learner

RUNS_ROOT = Path("/scratch/marzii/PT/lunarlander/grid_mixture_ms")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    p.add_argument("--runs", default=None,
                   help="glob over condition dirs under runs_root, e.g. "
                        "'lunarlander-human-count-v2-N*-policy'")
    p.add_argument("--run_dir", type=Path, default=None,
                   help="evaluate exactly one seed dir instead of a glob")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--eval_seed_base", type=int, default=10000,
                   help="episode e uses env seed eval_seed_base + e")
    p.add_argument("--env_id", default="LunarLanderContinuous-v2")
    p.add_argument("--overwrite", action="store_true",
                   help="re-evaluate runs that already have a result file")
    return p.parse_args()


def build_agent(env, seed):
    """IQL learner with the training architecture; params are overwritten by load()."""
    from ml_collections import ConfigDict
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lunarlander_config", "configs/lunarlander_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg: ConfigDict = mod.get_config()
    kwargs = dict(cfg)
    return Learner(seed,
                   env.observation_space.sample()[np.newaxis],
                   env.action_space.sample()[np.newaxis],
                   max_steps=1,
                   **kwargs)


def evaluate_run(run_dir: Path, args) -> dict:
    actor_path = run_dir / "policy" / "actor.flax"
    if not actor_path.exists():
        return {}

    seed = int(run_dir.name.split("_")[-1])

    env = gym.make(args.env_id)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    agent = build_agent(env, seed)
    agent.actor = agent.actor.load(str(actor_path))

    returns, lengths = [], []
    for ep in range(args.episodes):
        env.seed(args.eval_seed_base + ep)
        obs, done = env.reset(), False
        ret, n = 0.0, 0
        while not done:
            action = agent.sample_actions(obs, temperature=0.0)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, r, done, _ = env.step(action)
            ret += float(r)
            n += 1
        returns.append(ret)
        lengths.append(n)
    env.close()

    r = np.asarray(returns)
    return dict(
        run_dir=str(run_dir),
        condition=run_dir.parent.name,
        seed=seed,
        episodes=args.episodes,
        deterministic=True,
        eval_seed_base=args.eval_seed_base,
        env_id=args.env_id,
        reward_source="ground_truth_env_reward",
        mean_return=float(r.mean()),
        sd_return=float(r.std(ddof=1)),
        se_return=float(r.std(ddof=1) / np.sqrt(len(r))),
        median_return=float(np.median(r)),
        frac_return_above_200=float((r > 200).mean()),
        mean_episode_length=float(np.mean(lengths)),
        ep_returns=returns,
    )


def main():
    args = parse_args()

    if args.run_dir:
        seed_dirs = [args.run_dir]
    else:
        if not args.runs:
            raise SystemExit("pass --runs '<glob>' or --run_dir <path>")
        seed_dirs = []
        for cond in sorted(args.runs_root.glob(args.runs)):
            for s in range(args.seeds):
                d = cond / f"seed_{s}"
                if d.exists():
                    seed_dirs.append(d)

    print(f"{len(seed_dirs)} run dirs to evaluate "
          f"({args.episodes} deterministic episodes each)")

    per_cond = {}
    for d in seed_dirs:
        out_file = d / f"final_eval_{args.episodes}ep.json"
        if out_file.exists() and not args.overwrite:
            res = json.load(open(out_file))
            print(f"  (cached) {d.parent.name}/{d.name}: {res['mean_return']:.1f}")
        else:
            res = evaluate_run(d, args)
            if not res:
                print(f"  ! no saved policy: {d}")
                continue
            out_file.write_text(json.dumps(res, indent=2))
            print(f"  {d.parent.name}/{d.name}: {res['mean_return']:.1f} "
                  f"(>200: {100*res['frac_return_above_200']:.0f}%)")
        per_cond.setdefault(res["condition"], []).append(res["mean_return"])

    print("\n=== summary (mean +/- sd over seeds) ===")
    for cond in sorted(per_cond):
        v = np.asarray(per_cond[cond])
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{cond:<52} n={len(v):<3} {v.mean():7.1f} +/- {sd:5.1f}")


if __name__ == "__main__":
    main()
