"""Sample preference queries from a LunarLander HDF5 dataset.

For each of `--num_query` queries, picks a pair of segments (each `--query_len`
steps long) from random positions in the dataset, ensuring no segment crosses
an episode boundary. Writes the start indices in the pickle format that
`JaxPref/reward_transform.py` expects:

    <output_dir>/indices_num{N}_q{L}     pickled np.int32 array, shape (N,)
    <output_dir>/indices_2_num{N}_q{L}   pickled np.int32 array, shape (N,)
    <output_dir>/metadata.pkl            dict with hdf5 path, num_query, query_len, seed,
                                          and the trajectory split for replayability.

Usage
-----
    python -m scripts.preference.sample_query_indices \
        --hdf5 $SCRATCH/PT/lunarlander/seed_0/lunarlander-medium-replay-v2.hdf5 \
        --output_dir human_label/lunarlander-medium-replay-v2/seed_0 \
        --num_query 10 --query_len 100 --seed 42
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_query", type=int, default=10)
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def trajectory_bounds(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    """Return (n_trajs, 2) array of [start, end_inclusive] indices."""
    ends = (terminals | timeouts).astype(bool)
    bounds, start = [], 0
    for i, e in enumerate(ends):
        if e:
            bounds.append((start, i))
            start = i + 1
    if start <= len(ends) - 1:
        bounds.append((start, len(ends) - 1))
    return np.asarray(bounds, dtype=np.int64)


def sample_one_segment(rng: np.random.Generator, eligible_trajs: np.ndarray, query_len: int) -> int:
    """Pick a trajectory uniformly at random from `eligible_trajs` and a start
    inside it such that `[start, start+query_len)` fits."""
    while True:
        ti = rng.integers(0, len(eligible_trajs))
        s, e = eligible_trajs[ti]
        L = e - s + 1
        if L >= query_len:
            offset = rng.integers(0, L - query_len + 1)
            return int(s + offset)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.hdf5, "r") as f:
        terminals = f["terminals"][:].astype(bool)
        timeouts = f["timeouts"][:].astype(bool)

    bounds = trajectory_bounds(terminals, timeouts)
    lengths = bounds[:, 1] - bounds[:, 0] + 1
    eligible = bounds[lengths >= args.query_len]
    if len(eligible) == 0:
        raise RuntimeError(
            f"no trajectory in {args.hdf5} is at least {args.query_len} steps long; "
            f"try a smaller --query_len."
        )
    print(f"[sample] {len(bounds)} trajectories total, {len(eligible)} long enough for query_len={args.query_len}")

    indices_1 = np.zeros(args.num_query, dtype=np.int32)
    indices_2 = np.zeros(args.num_query, dtype=np.int32)
    for q in range(args.num_query):
        indices_1[q] = sample_one_segment(rng, eligible, args.query_len)
        indices_2[q] = sample_one_segment(rng, eligible, args.query_len)

    fn1 = args.output_dir / f"indices_num{args.num_query}_q{args.query_len}"
    fn2 = args.output_dir / f"indices_2_num{args.num_query}_q{args.query_len}"
    with open(fn1, "wb") as g: pickle.dump(indices_1, g)
    with open(fn2, "wb") as g: pickle.dump(indices_2, g)
    print(f"[sample] wrote {fn1.name} and {fn2.name}")

    meta = dict(
        hdf5=str(args.hdf5),
        num_query=args.num_query,
        query_len=args.query_len,
        seed=args.seed,
        n_trajectories=int(len(bounds)),
        n_eligible=int(len(eligible)),
    )
    with open(args.output_dir / "metadata.pkl", "wb") as g:
        pickle.dump(meta, g)
    print(f"[sample] wrote metadata.pkl  → {meta}")


if __name__ == "__main__":
    main()
