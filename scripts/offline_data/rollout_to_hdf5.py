"""Roll out a trained SAC policy and write a D4RL-format HDF5.

Usage
-----
    python -m scripts.offline_data.rollout_to_hdf5 \
        --actor $SCRATCH/PT/lunarlander/sac_run/expert/actor.zip \
        --output $SCRATCH/PT/lunarlander/lunarlander-expert-v2.hdf5 \
        --num_steps 1000000 --deterministic --seed 0

For the "random" variant pass `--actor random` to use uniform [-1, 1]^2
actions (no policy needed).

Schema (D4RL convention):
    observations      float32 (N, 8)
    actions           float32 (N, 2)
    rewards           float32 (N,)
    next_observations float32 (N, 8)
    terminals         bool    (N,)   env-natural episode end
    timeouts          bool    (N,)   gym TimeLimit truncation
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actor", required=True, help="Path to .zip checkpoint, or the literal string 'random'.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num_steps", type=int, default=1_000_000)
    p.add_argument("--deterministic", action="store_true", help="Argmax/mean action instead of sampling.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env_id", default="LunarLanderContinuous-v3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if args.actor == "random":
        actor = None
    else:
        from stable_baselines3 import SAC
        actor = SAC.load(args.actor, device="cpu")

    N = args.num_steps
    obs_buf = np.zeros((N, obs_dim), dtype=np.float32)
    act_buf = np.zeros((N, act_dim), dtype=np.float32)
    rew_buf = np.zeros((N,), dtype=np.float32)
    nxt_buf = np.zeros((N, obs_dim), dtype=np.float32)
    term_buf = np.zeros((N,), dtype=bool)
    time_buf = np.zeros((N,), dtype=bool)

    ep_returns: list[float] = []
    ep_return = 0.0
    ep_steps = 0

    for t in range(N):
        if actor is None:
            a = env.action_space.sample()
        else:
            a, _ = actor.predict(obs, deterministic=args.deterministic)
        nxt, r, terminated, truncated, _ = env.step(a)

        obs_buf[t] = obs
        act_buf[t] = a
        rew_buf[t] = r
        nxt_buf[t] = nxt
        term_buf[t] = bool(terminated)
        time_buf[t] = bool(truncated)

        ep_return += float(r)
        ep_steps += 1

        if terminated or truncated:
            ep_returns.append(ep_return)
            ep_return = 0.0
            ep_steps = 0
            obs, _ = env.reset()
        else:
            obs = nxt

    env.close()

    if ep_returns:
        ret = np.asarray(ep_returns, dtype=np.float32)
        print(
            f"[rollout] episodes={len(ep_returns)} "
            f"return mean={ret.mean():.1f} std={ret.std():.1f} "
            f"min={ret.min():.1f} max={ret.max():.1f}"
        )
    else:
        print("[rollout] WARNING: no completed episodes — single episode > num_steps?")

    print(f"[rollout] writing {N} transitions → {args.output}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("observations", data=obs_buf, compression="gzip")
        f.create_dataset("actions", data=act_buf, compression="gzip")
        f.create_dataset("rewards", data=rew_buf, compression="gzip")
        f.create_dataset("next_observations", data=nxt_buf, compression="gzip")
        f.create_dataset("terminals", data=term_buf, compression="gzip")
        f.create_dataset("timeouts", data=time_buf, compression="gzip")
        f.attrs["env_id"] = args.env_id
        f.attrs["actor"] = str(args.actor)
        f.attrs["num_steps"] = N
        f.attrs["deterministic"] = bool(args.deterministic)
        f.attrs["seed"] = int(args.seed)


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
