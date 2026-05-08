"""Roll out a trained actor and emit BOTH a D4RL-format HDF5 AND per-episode
mp4s, captured in lockstep so step indices in the HDF5 map cleanly to frames.

Output layout
-------------
    <output_dir>/
        lunarlander-<variant>-v2.hdf5    # same schema as rollout_to_hdf5.py
        episodes/
            episode_00000.mp4            # one mp4 per episode (frames at fps)
            episode_00001.mp4
            ...
            index.pkl                    # list of dicts:
                                         #   {episode_idx, start_step, end_step,
                                         #    n_frames, mp4_path}

Use `extract_segment_videos.py` to slice these per-episode mp4s into the
seg_A.mp4 / seg_B.mp4 pairs the labeller consumes.

Usage
-----
    python -m scripts.offline_data.rollout_with_video \
        --actor $SCRATCH/PT/lunarlander/seed_0/sac_run/medium/actor.zip \
        --output_dir $SCRATCH/PT/lunarlander/labels/lunarlander-medium-v2/seed_0/rollout \
        --num_steps 100000 --fps 20 --seed 0
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import imageio_ffmpeg  # ensures ffmpeg path is set
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actor", required=True,
                   help="Path to .zip checkpoint, or 'random' for uniform actions.")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_steps", type=int, default=100_000)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env_id", default="LunarLanderContinuous-v3")
    p.add_argument("--variant", default="medium",
                   help="Variant name used in the output HDF5 filename.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = args.output_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")
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
    obs_buf  = np.zeros((N, obs_dim), dtype=np.float32)
    act_buf  = np.zeros((N, act_dim), dtype=np.float32)
    rew_buf  = np.zeros((N,),         dtype=np.float32)
    nxt_buf  = np.zeros((N, obs_dim), dtype=np.float32)
    term_buf = np.zeros((N,),         dtype=bool)
    time_buf = np.zeros((N,),         dtype=bool)

    episodes: list[dict] = []
    cur_frames: list[np.ndarray] = []
    ep_start_step = 0
    ep_idx = 0

    def flush_episode(end_step_inclusive: int) -> None:
        nonlocal cur_frames, ep_idx, ep_start_step
        if not cur_frames:
            return
        mp4_path = ep_dir / f"episode_{ep_idx:05d}.mp4"
        # macro_block_size=1 so encoder doesn't pad odd dimensions.
        writer = imageio.get_writer(
            str(mp4_path), fps=args.fps, codec="libx264",
            quality=7, macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        for fr in cur_frames:
            writer.append_data(fr)
        writer.close()
        episodes.append({
            "episode_idx": ep_idx,
            "start_step":  int(ep_start_step),
            "end_step":    int(end_step_inclusive),
            "n_frames":    len(cur_frames),
            "mp4_path":    str(mp4_path),
        })
        cur_frames = []
        ep_idx += 1

    ep_returns: list[float] = []
    ep_return = 0.0

    # We render the FRAME CORRESPONDING TO obs_buf[t], not the post-step frame.
    # That means at the start of each episode we render once before stepping.
    cur_frames.append(env.render())

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

        if terminated or truncated:
            ep_returns.append(ep_return); ep_return = 0.0
            flush_episode(end_step_inclusive=t)
            obs, _ = env.reset()
            cur_frames.append(env.render())
            ep_start_step = t + 1
        else:
            obs = nxt
            cur_frames.append(env.render())

    # Any partial trailing episode: flush so its frames aren't lost (it just
    # won't be queryable for full-length segments without crossing).
    if cur_frames:
        flush_episode(end_step_inclusive=N - 1)

    env.close()

    if ep_returns:
        ret = np.asarray(ep_returns, dtype=np.float32)
        print(f"[rollout] episodes={len(ep_returns)} "
              f"return mean={ret.mean():.1f} std={ret.std():.1f} "
              f"min={ret.min():.1f} max={ret.max():.1f}")

    hdf5_path = args.output_dir / f"lunarlander-{args.variant}-v2.hdf5"
    print(f"[rollout] writing {N} transitions → {hdf5_path}")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("observations",      data=obs_buf,  compression="gzip")
        f.create_dataset("actions",           data=act_buf,  compression="gzip")
        f.create_dataset("rewards",           data=rew_buf,  compression="gzip")
        f.create_dataset("next_observations", data=nxt_buf,  compression="gzip")
        f.create_dataset("terminals",         data=term_buf, compression="gzip")
        f.create_dataset("timeouts",          data=time_buf, compression="gzip")
        f.attrs["env_id"] = args.env_id
        f.attrs["actor"] = str(args.actor)
        f.attrs["num_steps"] = N
        f.attrs["deterministic"] = bool(args.deterministic)
        f.attrs["seed"] = int(args.seed)
        f.attrs["fps"] = int(args.fps)
        f.attrs["with_video"] = True

    with open(ep_dir / "index.pkl", "wb") as g:
        pickle.dump(episodes, g)
    print(f"[rollout] wrote {len(episodes)} episode mp4s + index.pkl → {ep_dir}")


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
