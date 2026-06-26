"""Train sb3 SAC from scratch on LunarLanderContinuous-v3 with rendering
enabled, capture per-episode mp4s + an HDF5 of the replay buffer at the
target step. Produces a "medium-replay-v2" dataset whose transitions are
in lockstep with the rendered frames, so segment-level queries can be
shown to a human labeller.

Output layout
-------------
    <save_dir>/
        actor.zip                          # sb3 SAC policy at --total_steps
        replay.pkl                         # sb3 replay buffer at --total_steps
        lunarlander-medium-replay-v2.hdf5  # D4RL-schema dataset of the buffer
        episodes/
            episode_00000.mp4
            episode_00001.mp4
            ...
            index.pkl                      # list of dicts:
                                           #   {episode_idx, start_step,
                                           #    end_step, n_frames, mp4_path}

Convention for the per-step ↔ per-frame mapping (matches
rollout_with_video.py): each captured frame `f` in an episode mp4 shows
the *observation* of HDF5 transition row `start_step + f`. So frame
index 0 of episode K's mp4 = obs at HDF5 row episodes[K].start_step.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import imageio_ffmpeg  # ensures ffmpeg is available
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class RenderCaptureCallback(BaseCallback):
    """Captures one frame per env.step() into the *current* episode's
    mp4. On done, the frame just captured (= new initial state of the
    next episode) is moved into the new episode."""

    def __init__(self, out_dir: Path, fps: int, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.episodes_dir = self.out_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.episodes: list[dict] = []
        self.episode_idx = 0
        self.episode_start_step = 0

    def _render(self) -> np.ndarray | None:
        env = self.training_env.envs[0]
        try:
            return env.render()
        except Exception:
            return None

    def _on_training_start(self) -> None:
        f = self._render()
        if f is not None:
            self.frames.append(f)

    def _on_step(self) -> bool:
        f = self._render()
        if f is not None:
            self.frames.append(f)
        dones = self.locals.get("dones", [False])
        if dones[0]:
            new_init = self.frames.pop() if self.frames else None
            self._save_episode()
            if new_init is not None:
                self.frames.append(new_init)
        return True

    def _save_episode(self) -> None:
        if not self.frames:
            return
        mp4_path = self.episodes_dir / f"episode_{self.episode_idx:05d}.mp4"
        writer = imageio.get_writer(
            str(mp4_path), fps=self.fps, codec="libx264",
            quality=7, macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        for fr in self.frames:
            writer.append_data(fr)
        writer.close()

        n_t = len(self.frames)  # frames = number of HDF5 transitions for this ep
        end_step = self.episode_start_step + n_t - 1
        self.episodes.append({
            "episode_idx": self.episode_idx,
            "start_step": int(self.episode_start_step),
            "end_step":   int(end_step),
            "n_frames":   n_t,
            "mp4_path":   str(mp4_path),
        })
        self.frames = []
        self.episode_idx += 1
        self.episode_start_step = end_step + 1

    def _on_training_end(self) -> None:
        if self.frames:
            self._save_episode()
        with open(self.episodes_dir / "index.pkl", "wb") as f:
            pickle.dump(self.episodes, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save_dir", required=True, type=Path)
    p.add_argument("--total_steps", type=int, default=70_000,
                   help="Number of env steps to train. Default 70K = the medium-replay checkpoint.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def replay_buffer_to_hdf5(rb, n: int, out_path: Path, fps: int = 20) -> None:
    """Write the first `n` populated entries of sb3's replay buffer as a
    D4RL-schema HDF5 (matches our scripts/offline_data/replay_to_hdf5.py)."""
    obs      = rb.observations[:n, 0, :].astype(np.float32)
    nxt      = rb.next_observations[:n, 0, :].astype(np.float32)
    act      = rb.actions[:n, 0, :].astype(np.float32)
    rew      = rb.rewards[:n, 0].astype(np.float32)
    dones    = rb.dones[:n, 0].astype(bool)
    timeouts = rb.timeouts[:n, 0].astype(bool)
    terminals = dones & ~timeouts
    with h5py.File(out_path, "w") as f:
        f.create_dataset("observations",      data=obs, compression="gzip")
        f.create_dataset("actions",           data=act, compression="gzip")
        f.create_dataset("rewards",           data=rew, compression="gzip")
        f.create_dataset("next_observations", data=nxt, compression="gzip")
        f.create_dataset("terminals",         data=terminals, compression="gzip")
        f.create_dataset("timeouts",          data=timeouts,  compression="gzip")
        f.attrs["n_transitions"] = n
        # IMPORTANT: extract_segment_videos.py reads attrs["fps"] when
        # slicing segments out of episode mp4s; omitting it triggers a
        # broken heuristic that yields fps=1 and seeks past the segment.
        f.attrs["fps"] = int(fps)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = Monitor(env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # Hyperparameters mirror scripts/offline_data/train_sac_for_offline.py.
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=7.3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[400, 300]),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    cb = RenderCaptureCallback(args.save_dir, args.fps)
    model.learn(total_timesteps=args.total_steps, callback=cb)

    model.save(args.save_dir / "actor.zip")
    model.save_replay_buffer(args.save_dir / "replay.pkl")

    rb = model.replay_buffer
    n = int(rb.size())
    hdf5_path = args.save_dir / "lunarlander-medium-replay-v2.hdf5"
    replay_buffer_to_hdf5(rb, n, hdf5_path, fps=args.fps)
    print(f"[done] {n} transitions in HDF5 → {hdf5_path}")
    print(f"[done] {len(cb.episodes)} episode mp4s → {args.save_dir / 'episodes'}")


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
