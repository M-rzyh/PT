"""Train SAC on LunarLanderContinuous and snapshot at "medium" + "expert".

Produces, in `--save_dir`:
    medium/actor.zip        sb3 SAC policy at `--medium_step` (default 250K)
    medium/replay.pkl       sb3 ReplayBuffer at the same step
    expert/actor.zip        sb3 SAC policy at `--total_steps`   (default 1M)
    expert/replay.pkl       sb3 ReplayBuffer at the same step

These four artefacts are the inputs to `rollout_to_hdf5.py` and
`replay_to_hdf5.py`, which produce the D4RL-format HDF5 datasets.

Usage
-----
    python -m scripts.offline_data.train_sac_for_offline \
        --save_dir $SCRATCH/PT/lunarlander/sac_run_$SLURM_JOB_ID \
        --total_steps 1000000 --medium_step 250000 --seed 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class SnapshotAtCallback(BaseCallback):
    """Saves the policy + replay buffer once when num_timesteps >= `at_step`."""

    def __init__(self, at_step: int, out_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.at_step = int(at_step)
        self.out_dir = Path(out_dir)
        self._fired = False

    def _on_step(self) -> bool:
        if self._fired or self.num_timesteps < self.at_step:
            return True
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.out_dir / "actor.zip")
        self.model.save_replay_buffer(self.out_dir / "replay.pkl")
        if self.verbose:
            print(
                f"[SnapshotAtCallback] step {self.num_timesteps}: "
                f"saved actor + replay → {self.out_dir}"
            )
        self._fired = True
        return True


def make_env(seed: int) -> gym.Env:
    env = gym.make("LunarLanderContinuous-v3")
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save_dir", required=True, type=Path)
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--medium_step", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tb", action="store_true",
                   help="Enable tensorboard logging (requires `pip install tensorboard`).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.seed)

    # Hyperparameters from the rl-baselines3-zoo `sac_LunarLanderContinuous`
    # tuned config (https://github.com/DLR-RM/rl-baselines3-zoo). These reach
    # ~+260 reward by step 1M and ~+100 reward by step 250K, matching the
    # "expert" / "medium" buckets we need.
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
        tensorboard_log=str(args.save_dir / "tb") if args.tb else None,
        verbose=1,
    )

    snapshot_cb = SnapshotAtCallback(
        at_step=args.medium_step,
        out_dir=args.save_dir / "medium",
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=snapshot_cb,
        log_interval=args.log_interval,
    )

    expert_dir = args.save_dir / "expert"
    expert_dir.mkdir(parents=True, exist_ok=True)
    model.save(expert_dir / "actor.zip")
    model.save_replay_buffer(expert_dir / "replay.pkl")
    print(f"[expert] step {model.num_timesteps}: saved actor + replay → {expert_dir}")


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
