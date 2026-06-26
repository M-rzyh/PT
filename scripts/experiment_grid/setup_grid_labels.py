"""Materialize the 14 label dirs for the PT count/noise grid experiment.

Once. Reads the offline dataset, samples 1000 query pairs with data_seed=0,
computes oracle labels (compare segment reward sums), and writes:

    human_label/lunarlander-grid-N{N}-clean/        (count axis, N in {50, 100, 500, 1000})
        indices_num{N}_q100        first N of the 1000 shared start indices for seg-A
        indices_2_num{N}_q100      first N for seg-B
        label_human                first N clean oracle labels in {0, 1, -1}
        label_alignment.json       sanity (= 1.0 for clean)

    human_label/lunarlander-grid-N1000-noise{P}/    (noise axis, P in {10,...,100})
        indices_num1000_q100       all 1000
        indices_2_num1000_q100     all 1000
        label_human                clean labels with `p = P/100` re-sampled uniformly
        label_alignment.json       fraction of labels matching clean
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--label_root", required=True, type=Path,
                   help="e.g. /home/marzii/PT/PreferenceTransformer/human_label")
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--num_total", type=int, default=1000,
                   help="Total query pairs to sample once.")
    p.add_argument("--data_seed", type=int, default=0,
                   help="Seed for the one-shot 1000-pair sampling.")
    p.add_argument("--noise_seed", type=int, default=0,
                   help="Seed for the noise re-sampling per noise level.")
    p.add_argument("--n_counts", nargs="+", type=int, default=[50, 100, 500, 1000])
    p.add_argument("--noise_pcts", nargs="+", type=int,
                   default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return p.parse_args()


def trajectory_bounds(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    ends = (terminals | timeouts).astype(bool)
    bounds, start = [], 0
    for i, e in enumerate(ends):
        if e:
            bounds.append((start, i))
            start = i + 1
    if start <= len(ends) - 1:
        bounds.append((start, len(ends) - 1))
    return np.asarray(bounds, dtype=np.int64)


def sample_one_segment(rng: np.random.Generator, eligible: np.ndarray, qlen: int) -> int:
    while True:
        ti = rng.integers(0, len(eligible))
        s, e = eligible[ti]
        L = e - s + 1
        if L >= qlen:
            return int(s + rng.integers(0, L - qlen + 1))


def oracle_label(seg_a_start: int, seg_b_start: int, qlen: int, rewards: np.ndarray) -> int:
    """Return PT-format label in {0, 1, -1}: 0 if A preferred, 1 if B, -1 if tie."""
    a = float(rewards[seg_a_start:seg_a_start + qlen].sum())
    b = float(rewards[seg_b_start:seg_b_start + qlen].sum())
    if a > b: return 0
    if b > a: return 1
    return -1


def apply_noise(clean: list[int], p: float, rng: np.random.Generator) -> list[int]:
    """For each label, with prob p, replace with uniform sample from {-1, 0, 1}."""
    classes = np.array([-1, 0, 1], dtype=int)
    noisy = list(clean)
    flips = rng.random(len(clean)) < p
    for i, do_flip in enumerate(flips):
        if do_flip:
            noisy[i] = int(rng.choice(classes))
    return noisy


def write_condition_dir(out_dir: Path, idx_1: np.ndarray, idx_2: np.ndarray,
                        labels: list[int], num_query: int, query_len: int,
                        meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Slice to first num_query (count axis truncates; noise axis uses all 1000).
    idx_1_s = idx_1[:num_query].astype(np.int32)
    idx_2_s = idx_2[:num_query].astype(np.int32)
    lbls_s  = labels[:num_query]
    with open(out_dir / f"indices_num{num_query}_q{query_len}", "wb") as g:
        pickle.dump(idx_1_s, g)
    with open(out_dir / f"indices_2_num{num_query}_q{query_len}", "wb") as g:
        pickle.dump(idx_2_s, g)
    with open(out_dir / "label_human", "wb") as g:
        pickle.dump(lbls_s, g)
    # Metadata lives in a sibling dir, NOT inside out_dir, because PT's
    # loader hard-codes `sorted(os.listdir(base_path))` to unpack exactly
    # 3 entries — any 4th file in the dir breaks the destructuring.
    meta_dir = out_dir.parent / "_grid_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / f"{out_dir.name}.label_alignment.json", "w") as g:
        json.dump(meta, g, indent=2)


def main() -> None:
    args = parse_args()

    print(f"[setup] HDF5 = {args.hdf5}")
    with h5py.File(args.hdf5, "r") as f:
        terminals = f["terminals"][:].astype(bool)
        timeouts  = f["timeouts"][:].astype(bool)
        rewards   = f["rewards"][:].astype(np.float32)

    bounds = trajectory_bounds(terminals, timeouts)
    lengths = bounds[:, 1] - bounds[:, 0] + 1
    eligible = bounds[lengths >= args.query_len]
    if len(eligible) == 0:
        raise RuntimeError(f"no trajectory long enough for query_len={args.query_len}")
    print(f"[setup] {len(bounds)} trajs total, {len(eligible)} eligible (len >= {args.query_len})")

    # Sample 1000 query pairs once.
    rng = np.random.default_rng(args.data_seed)
    idx_1 = np.zeros(args.num_total, dtype=np.int32)
    idx_2 = np.zeros(args.num_total, dtype=np.int32)
    for q in range(args.num_total):
        idx_1[q] = sample_one_segment(rng, eligible, args.query_len)
        idx_2[q] = sample_one_segment(rng, eligible, args.query_len)
    print(f"[setup] sampled {args.num_total} query pairs with data_seed={args.data_seed}")

    # Compute oracle labels (one per pair).
    clean: list[int] = []
    for q in range(args.num_total):
        clean.append(oracle_label(int(idx_1[q]), int(idx_2[q]), args.query_len, rewards))
    counts = {v: clean.count(v) for v in (-1, 0, 1)}
    print(f"[setup] clean oracle label counts (over {args.num_total}): "
          f"A-pref={counts[0]}  B-pref={counts[1]}  tie={counts[-1]}")

    # --- Count axis (clean labels, varying N) ---
    for N in args.n_counts:
        env_tag = f"lunarlander-grid-N{N}-clean"
        out_dir = args.label_root / env_tag
        meta = dict(
            condition_id=env_tag, num_query=N, noise_pct=0,
            data_seed=args.data_seed, noise_seed=None,
            label_counts={-1: clean[:N].count(-1),
                          0: clean[:N].count(0),
                          1: clean[:N].count(1)},
            label_alignment=1.0,
        )
        write_condition_dir(out_dir, idx_1, idx_2, clean, N, args.query_len, meta)
        print(f"[count] wrote {env_tag} (N={N})")

    # --- Noise axis (N=1000, varying noise) ---
    for P in args.noise_pcts:
        p = P / 100.0
        rng_noise = np.random.default_rng(args.noise_seed + P)  # deterministic per P
        noisy = apply_noise(clean, p, rng_noise)
        alignment = float(sum(int(a == b) for a, b in zip(clean, noisy)) / len(clean))
        env_tag = f"lunarlander-grid-N{args.num_total}-noise{P}"
        out_dir = args.label_root / env_tag
        meta = dict(
            condition_id=env_tag, num_query=args.num_total, noise_pct=P,
            data_seed=args.data_seed, noise_seed=args.noise_seed + P,
            label_counts={-1: noisy.count(-1),
                          0: noisy.count(0),
                          1: noisy.count(1)},
            label_alignment=alignment,
        )
        write_condition_dir(out_dir, idx_1, idx_2, noisy, args.num_total, args.query_len, meta)
        print(f"[noise] wrote {env_tag} (P={P}%, alignment={alignment:.3f})")

    print(f"\n[done] {len(args.n_counts) + len(args.noise_pcts)} condition dirs under "
          f"{args.label_root}/lunarlander-grid-*/")


if __name__ == "__main__":
    main()
