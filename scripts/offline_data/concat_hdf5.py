"""Concatenate two D4RL-format HDF5 files (used for the medium-expert variant).

Usage
-----
    python -m scripts.offline_data.concat_hdf5 \
        --inputs $SCRATCH/PT/lunarlander/lunarlander-medium-v2.hdf5 \
                 $SCRATCH/PT/lunarlander/lunarlander-expert-v2.hdf5 \
        --num_each 500000 \
        --output $SCRATCH/PT/lunarlander/lunarlander-medium-expert-v2.hdf5

Behavior:
    - Slices the first `num_each` transitions from each input (in input order).
      Pass `--num_each -1` to keep all transitions from each file.
    - All inputs must share the same keys and dtypes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

KEYS = ("observations", "actions", "rewards", "next_observations", "terminals", "timeouts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num_each", type=int, default=-1, help="-1 = use the full file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if len(args.inputs) < 2:
        raise SystemExit("--inputs needs at least two HDF5 files")

    pieces: dict[str, list[np.ndarray]] = {k: [] for k in KEYS}
    sizes: list[int] = []

    for path in args.inputs:
        with h5py.File(path, "r") as f:
            n = f["observations"].shape[0]
            take = n if args.num_each < 0 else min(args.num_each, n)
            sizes.append(take)
            for k in KEYS:
                pieces[k].append(f[k][:take])
            print(f"[concat] {path}  using {take}/{n} transitions")

    print(f"[concat] writing {sum(sizes)} transitions → {args.output}")
    with h5py.File(args.output, "w") as f:
        for k in KEYS:
            arr = np.concatenate(pieces[k], axis=0)
            f.create_dataset(k, data=arr, compression="gzip")
        f.attrs["sources"] = "\n".join(str(p) for p in args.inputs)
        f.attrs["per_source_take"] = sizes


if __name__ == "__main__":
    main()
