"""Generate per-seed noise label directories for the multi-seed noise grid.

Unlike setup_grid_labels.py (which fixes one noise draw shared by all training
seeds), this script ties the noise corruption RNG to the training seed.  Each
(training_seed, noise_pct) pair gets a unique set of corrupted pairs, so
different training seeds see different noise instantiations.

Noise RNG seed formula: (training_seed + 10) * 100 + noise_pct
  e.g. training_seed=0, noise_pct=20  →  rng_seed = 1020
       training_seed=1, noise_pct=20  →  rng_seed = 1120
       training_seed=9, noise_pct=20  →  rng_seed = 1920
All well above the existing seeds (0 + noise_pct < 100).

Output dirs:
    human_label/lunarlander-grid-ms-N1000-noise{P}-s{S}/
    human_label/lunarlander-grid-ms-N1000-clean-s{S}/   (P=0, labels identical
                                                          across seeds but kept
                                                          for uniform naming)

Metadata:
    human_label/_grid_metadata/lunarlander-grid-ms-N1000-noise{P}-s{S}.label_alignment.json
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--label_root", required=True, type=Path)
    p.add_argument("--training_seed", type=int, required=True,
                   help="The PT/IQL training seed this label set is for (0-9).")
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--num_total", type=int, default=1000)
    p.add_argument("--data_seed", type=int, default=None,
                   help="Override the pair-sampling RNG seed. If omitted, uses "
                        "data_seed_base + training_seed.")
    p.add_argument("--data_seed_base", type=int, default=1000,
                   help="Base for per-seed pair sampling: actual seed = base + training_seed. "
                        "Default 1000 keeps count-axis seeds (1000-1009) well away from "
                        "existing noise-axis seeds (data_seed=0).")
    p.add_argument("--noise_pcts", nargs="+", type=int, default=[],
                   help="Noise percentages to generate (0-100). Default: none.")
    p.add_argument("--n_counts", nargs="+", type=int, default=[],
                   help="Label counts for count axis (e.g. 50 100 250 500 750 1000).")
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


def sample_one_segment(rng, eligible, qlen):
    while True:
        ti = rng.integers(0, len(eligible))
        s, e = eligible[ti]
        L = e - s + 1
        if L >= qlen:
            return int(s + rng.integers(0, L - qlen + 1))


def oracle_label(seg_a_start, seg_b_start, qlen, rewards):
    a = float(rewards[seg_a_start:seg_a_start + qlen].sum())
    b = float(rewards[seg_b_start:seg_b_start + qlen].sum())
    if a > b: return 0
    if b > a: return 1
    return -1


def apply_noise(clean, p, rng):
    classes = np.array([-1, 0, 1], dtype=int)
    noisy = list(clean)
    flips = rng.random(len(clean)) < p
    for i, do_flip in enumerate(flips):
        if do_flip:
            noisy[i] = int(rng.choice(classes))
    return noisy


def write_condition_dir(out_dir, idx_1, idx_2, labels, num_query, query_len, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_1_s = idx_1[:num_query].astype(np.int32)
    idx_2_s = idx_2[:num_query].astype(np.int32)
    lbls_s  = labels[:num_query]
    with open(out_dir / f"indices_num{num_query}_q{query_len}", "wb") as g:
        pickle.dump(idx_1_s, g)
    with open(out_dir / f"indices_2_num{num_query}_q{query_len}", "wb") as g:
        pickle.dump(idx_2_s, g)
    with open(out_dir / "label_human", "wb") as g:
        pickle.dump(lbls_s, g)
    meta_dir = out_dir.parent / "_grid_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / f"{out_dir.name}.label_alignment.json", "w") as g:
        json.dump(meta, g, indent=2)


def main() -> None:
    args = parse_args()
    S = args.training_seed

    with h5py.File(args.hdf5, "r") as f:
        terminals = f["terminals"][:].astype(bool)
        timeouts  = f["timeouts"][:].astype(bool)
        rewards   = f["rewards"][:].astype(np.float32)

    bounds = trajectory_bounds(terminals, timeouts)
    lengths = bounds[:, 1] - bounds[:, 0] + 1
    eligible = bounds[lengths >= args.query_len]

    # Per-seed data_seed: each training seed samples a different pool of pairs.
    actual_data_seed = args.data_seed if args.data_seed is not None \
                       else args.data_seed_base + S
    rng = np.random.default_rng(actual_data_seed)
    idx_1 = np.zeros(args.num_total, dtype=np.int32)
    idx_2 = np.zeros(args.num_total, dtype=np.int32)
    for q in range(args.num_total):
        idx_1[q] = sample_one_segment(rng, eligible, args.query_len)
        idx_2[q] = sample_one_segment(rng, eligible, args.query_len)

    clean = [oracle_label(int(idx_1[q]), int(idx_2[q]), args.query_len, rewards)
             for q in range(args.num_total)]

    # --- Count axis (clean labels, first N pairs from the per-seed pool) ------
    for N in args.n_counts:
        env_tag = f"lunarlander-grid-ms-count-N{N}-s{S}"
        out_dir = args.label_root / env_tag
        meta = dict(
            condition_id=env_tag,
            num_query=N,
            noise_pct=0,
            training_seed=S,
            actual_data_seed=actual_data_seed,
            label_alignment=1.0,
        )
        write_condition_dir(out_dir, idx_1, idx_2, clean, N, args.query_len, meta)
        print(f"[s={S}] wrote {env_tag}  (N={N}, clean)")

    # --- Noise axis -----------------------------------------------------------
    for P in args.noise_pcts:
        # Unique noise RNG seed per (training_seed, noise_pct).
        noise_rng_seed = (S + 10) * 100 + P
        rng_noise = np.random.default_rng(noise_rng_seed)

        if P == 0:
            noisy = list(clean)
            env_tag = f"lunarlander-grid-ms-N{args.num_total}-clean-s{S}"
        else:
            noisy = apply_noise(clean, P / 100.0, rng_noise)
            env_tag = f"lunarlander-grid-ms-N{args.num_total}-noise{P}-s{S}"

        alignment = float(sum(int(a == b) for a, b in zip(clean, noisy)) / len(clean))
        out_dir = args.label_root / env_tag
        meta = dict(
            condition_id=env_tag,
            num_query=args.num_total,
            noise_pct=P,
            training_seed=S,
            noise_rng_seed=noise_rng_seed,
            data_seed=args.data_seed,
            label_alignment=alignment,
        )
        write_condition_dir(out_dir, idx_1, idx_2, noisy, args.num_total,
                            args.query_len, meta)
        print(f"[s={S}] wrote {env_tag}  (noise={P}%, alignment={alignment:.3f})")

    print(f"[done] training_seed={S}: {len(args.noise_pcts)} label dirs written.")


if __name__ == "__main__":
    main()
