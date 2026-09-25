"""Render ONE deterministic episode of a saved PT/IQL policy to a frame stack.

The pt env has no imageio, so this writes the raw RGB frames to .npz and the
mp4 is encoded elsewhere (AAAI_Student_Abstract_27/scripts/render_policy_videos.py).

The policy is loaded exactly as scripts/experiment_grid/eval_saved_policy_pt.py
loads it for evaluation (same Learner, same config, deterministic actions), and
the episode uses one fixed env seed so paired conditions see the same episode.

Nothing is retrained and no existing file is written.

Usage (repo root, pt env):
    python -m scripts.plots.render_pt_policy_frames \
        --actor .../policy/actor.flax --env_seed 12345 --out /path/frames.npz
"""
import argparse
from pathlib import Path

import numpy as np
import gym

import wrappers                     # noqa: F401  repo-root module, as in the eval script
from scripts.experiment_grid.eval_saved_policy_pt import build_agent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--env_id", default="LunarLanderContinuous-v2")
    p.add_argument("--env_seed", type=int, default=12345)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None,
                   help="agent build seed (default: parsed from the run dir name)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    seed = a.seed
    if seed is None:
        seed = int(a.actor.parent.parent.name.split("_")[-1])

    env = gym.make(a.env_id)
    agent = build_agent(env, seed)
    agent.actor = agent.actor.load(str(a.actor))

    env.seed(a.env_seed)
    obs, done, ret, n = env.reset(), False, 0.0, 0
    frames = [env.render(mode="rgb_array")]
    while not done and n < a.max_steps:
        action = agent.sample_actions(obs, temperature=0.0)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, r, done, _ = env.step(action)
        ret += float(r)
        n += 1
        frames.append(env.render(mode="rgb_array"))
    env.close()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, frames=np.asarray(frames, dtype=np.uint8),
                        episode_return=ret, steps=n, env_seed=a.env_seed)
    print(f"{a.out.name}: return={ret:.1f} steps={n} frames={len(frames)}")


if __name__ == "__main__":
    main()
