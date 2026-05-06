"""Convert an sb3 ReplayBuffer pickle to a D4RL-format HDF5.

Used to produce the `medium-replay` variant: the contents of the SAC replay
buffer at the moment the medium actor was snapshotted.

Usage
-----
    python -m scripts.offline_data.replay_to_hdf5 \
        --replay $SCRATCH/PT/lunarlander/sac_run/medium/replay.pkl \
        --output $SCRATCH/PT/lunarlander/lunarlander-medium-replay-v2.hdf5

Schema matches `rollout_to_hdf5.py` (D4RL convention).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from stable_baselines3 import SAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", required=True, type=Path, help="sb3 replay buffer .pkl")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--actor", type=Path,
                   help="Optional sb3 actor .zip used to construct a model whose load_replay_buffer reads the pickle. "
                        "If omitted, falls back to a default SAC stub.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.actor is not None:
        model = SAC.load(args.actor, device="cpu")
    else:
        # Build a tiny stub model just to invoke load_replay_buffer; never trained.
        import gymnasium as gym
        env = gym.make("LunarLanderContinuous-v3")
        model = SAC("MlpPolicy", env, device="cpu", verbose=0)
    model.load_replay_buffer(args.replay)
    rb = model.replay_buffer

    n_used = int(rb.size())
    if n_used == 0:
        raise RuntimeError(f"replay buffer at {args.replay} reports size 0 — did training reach 'learning_starts'?")

    n_envs = rb.n_envs
    if n_envs != 1:
        raise NotImplementedError(f"this script assumes single-env training; got n_envs={n_envs}")

    pos = rb.pos if not rb.full else rb.buffer_size
    if rb.full:
        # Buffer wrapped: oldest entry is at rb.pos. We export in chronological order.
        idx = np.concatenate([np.arange(rb.pos, rb.buffer_size), np.arange(0, rb.pos)])
    else:
        idx = np.arange(pos)
    assert len(idx) == n_used, f"index length {len(idx)} != size {n_used}"

    obs_buf = rb.observations[idx, 0].astype(np.float32, copy=False)
    nxt_buf = rb.next_observations[idx, 0].astype(np.float32, copy=False)
    act_buf = rb.actions[idx, 0].astype(np.float32, copy=False)
    rew_buf = rb.rewards[idx, 0].astype(np.float32, copy=False)
    # sb3 stores `dones` (terminated|truncated, but with timeouts info logged separately
    # in `timeouts`). We map them to D4RL's terminals/timeouts semantics:
    #   terminals = dones & ~timeouts   (env-natural episode end)
    #   timeouts  = timeouts             (TimeLimit truncation)
    dones = rb.dones[idx, 0].astype(bool)
    timeouts = rb.timeouts[idx, 0].astype(bool)
    terminals = dones & ~timeouts

    print(
        f"[replay] {n_used} transitions  "
        f"reward mean={rew_buf.mean():.3f}  "
        f"#terminals={int(terminals.sum())}  "
        f"#timeouts={int(timeouts.sum())}"
    )

    print(f"[replay] writing → {args.output}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("observations", data=obs_buf, compression="gzip")
        f.create_dataset("actions", data=act_buf, compression="gzip")
        f.create_dataset("rewards", data=rew_buf, compression="gzip")
        f.create_dataset("next_observations", data=nxt_buf, compression="gzip")
        f.create_dataset("terminals", data=terminals, compression="gzip")
        f.create_dataset("timeouts", data=timeouts, compression="gzip")
        f.attrs["replay_pkl"] = str(args.replay)
        f.attrs["n_transitions"] = n_used


if __name__ == "__main__":
    main()
