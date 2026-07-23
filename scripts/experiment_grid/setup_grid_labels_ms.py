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
    p.add_argument("--noise_mode", choices=["random_replace", "deterministic_flip"],
                   default="random_replace",
                   help="How a corrupted label is generated (noise axis only). "
                        "random_replace (default): uniform draw from {-1,0,1}; "
                        "deterministic_flip: invert the preference (0<->1). "
                        "flip outputs use 'flipnoise{P}'/'flipclean' tags so they "
                        "never collide with random_replace's 'noise{P}'/'clean'.")
    p.add_argument("--noise_selection", choices=["exact", "coin"], default="exact",
                   help="exact = corrupt exactly n%% of labels (default; tags 'exnoise'/'exclean'); "
                        "coin = per-label Bernoulli(n%%) (tags 'noise'/'clean'). coin's corrupted "
                        "count wobbles, worst at small N.")
    # --- Frame-blanking difficulty (Exp 1): blind-oracle labels ---------------
    # Opt-in; default empty = unchanged. Oracle sees reward only on VISIBLE frames.
    p.add_argument("--blind_oracle_pcts", nargs="+", type=int, default=[],
                   help="Blank percentages for BLIND-ORACLE labeling (frame-blanking "
                        "difficulty). Oracle labels from visible-frame reward only. "
                        "Tags 'blindoracle{P}'/'blindoracleclean'. Default: none (off).")
    p.add_argument("--blind_blank_mode", choices=["stochastic", "deterministic", "block"],
                   default="stochastic",
                   help="Blank schedule for blind-oracle: coin(p), every-k, or contiguous block runs.")
    p.add_argument("--blind_block_len", type=int, default=10,
                   help="block mode: contiguous run length (frames) for blind-oracle blanking "
                        "(match the human videos' --block_len).")
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


def _select_flips(n, p, rng, selection):
    """Which labels get corrupted. exact = exactly round(p*n) (no count wobble);
    coin = independent Bernoulli(p) per label (count wobbles, worst at small n)."""
    if selection == "coin":
        return rng.random(n) < p
    flips = np.zeros(n, dtype=bool)
    flips[rng.choice(n, size=int(round(p * n)), replace=False)] = True
    return flips


def apply_noise(clean, p, rng, selection="exact"):
    """random_replace: a corrupted label is replaced by a uniform draw from
    {-1, 0, 1}. A corruption lands on the same / opposite / tie answer each
    with prob 1/3, so alignment = 1 - (2/3)p and even p=1.0 leaves ~1/3 of
    labels accidentally correct (information destruction, not inversion)."""
    classes = np.array([-1, 0, 1], dtype=int)
    noisy = list(clean)
    flips = _select_flips(len(clean), p, rng, selection)
    for i, do_flip in enumerate(flips):
        if do_flip:
            noisy[i] = int(rng.choice(classes))
    return noisy


def apply_noise_flip(clean, p, rng, selection="exact"):
    """deterministic_flip: a corrupted label becomes the EXACT opposite
    preference (0<->1). Ties (-1) have no opposite and are left unchanged.
    This injects anti-signal rather than destroying it: alignment = 1 - p
    (over decisive pairs), and at p=1.0 every decisive label is inverted."""
    noisy = list(clean)
    flips = _select_flips(len(clean), p, rng, selection)
    for i, do_flip in enumerate(flips):
        if do_flip:
            if clean[i] == 0:
                noisy[i] = 1
            elif clean[i] == 1:
                noisy[i] = 0
            # clean[i] == -1 (tie): no opposite, leave unchanged
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


def blank_mask(n, mode, p, k, rng, block_len=10):
    """Boolean (n,) of blanked frames. stochastic=coin(p); deterministic=every k-th;
    block=black out contiguous runs of block_len (~p of the blocks) — matches the
    human videos' block-blanking (blank_segment_videos.py)."""
    if mode == "deterministic":
        return (np.arange(n) % int(k)) == 0
    if mode == "block":
        nblocks = (n + int(block_len) - 1) // int(block_len)
        blk = rng.random(nblocks) < float(p)
        return blk[np.arange(n) // int(block_len)]
    return rng.random(n) < float(p)


def blind_oracle_label(sa, sb, qlen, rewards, mask_a, mask_b):
    """Oracle that sees reward only on VISIBLE (non-blanked) frames of each segment.

    Frame-blanking difficulty: when the decisive reward falls on a blanked frame the
    label can flip, so labels degrade as the blank fraction rises.
    """
    a = float(rewards[sa:sa + qlen][~mask_a].sum())
    b = float(rewards[sb:sb + qlen][~mask_b].sum())
    if a > b:
        return 0
    if b > a:
        return 1
    return -1


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
    _ex = args.noise_selection == "exact"
    if args.noise_mode == "deterministic_flip":
        noise_word = "exflipnoise" if _ex else "flipnoise"
        clean_word = "exflipclean" if _ex else "flipclean"
        noise_fn = apply_noise_flip
    else:
        noise_word = "exnoise" if _ex else "noise"
        clean_word = "exclean" if _ex else "clean"
        noise_fn = apply_noise
    for P in args.noise_pcts:
        # Unique noise RNG seed per (training_seed, noise_pct).
        noise_rng_seed = (S + 10) * 100 + P
        rng_noise = np.random.default_rng(noise_rng_seed)

        if P == 0:
            noisy = list(clean)
            env_tag = f"lunarlander-grid-ms-N{args.num_total}-{clean_word}-s{S}"
        else:
            noisy = noise_fn(clean, P / 100.0, rng_noise, args.noise_selection)
            env_tag = f"lunarlander-grid-ms-N{args.num_total}-{noise_word}{P}-s{S}"

        alignment = float(sum(int(a == b) for a, b in zip(clean, noisy)) / len(clean))
        out_dir = args.label_root / env_tag
        meta = dict(
            condition_id=env_tag,
            num_query=args.num_total,
            noise_pct=P,
            noise_mode=args.noise_mode,
            training_seed=S,
            noise_rng_seed=noise_rng_seed,
            data_seed=args.data_seed,
            label_alignment=alignment,
        )
        write_condition_dir(out_dir, idx_1, idx_2, noisy, args.num_total,
                            args.query_len, meta)
        print(f"[s={S}] wrote {env_tag}  (noise={P}%, alignment={alignment:.3f})")

    # --- Blind-oracle axis (frame-blanking; labels from VISIBLE-frame reward) --
    for P in args.blind_oracle_pcts:
        bo_rng = np.random.default_rng((S + 20) * 1000 + P)  # distinct RNG stream
        labels = []
        for q in range(args.num_total):
            sa, sb = int(idx_1[q]), int(idx_2[q])
            if P == 0:
                ma = np.zeros(args.query_len, dtype=bool)
                mb = np.zeros(args.query_len, dtype=bool)
            else:
                ma = blank_mask(args.query_len, args.blind_blank_mode, P / 100.0, 2, bo_rng, args.blind_block_len)
                mb = blank_mask(args.query_len, args.blind_blank_mode, P / 100.0, 2, bo_rng, args.blind_block_len)
            labels.append(blind_oracle_label(sa, sb, args.query_len, rewards, ma, mb))
        tag = "blindoracleclean" if P == 0 else f"blindoracle{P}"
        env_tag = f"lunarlander-grid-ms-N{args.num_total}-{tag}-s{S}"
        alignment = float(sum(int(a == b) for a, b in zip(clean, labels)) / len(clean))
        meta = dict(
            condition_id=env_tag, num_query=args.num_total, blank_pct=P,
            label_mode="blind_oracle", blank_mode=args.blind_blank_mode,
            training_seed=S, label_alignment=alignment,
        )
        write_condition_dir(args.label_root / env_tag, idx_1, idx_2, labels,
                            args.num_total, args.query_len, meta)
        print(f"[s={S}] wrote {env_tag}  (blind_oracle blank={P}%, alignment={alignment:.3f})")

    print(f"[done] training_seed={S}: {len(args.noise_pcts)} label dirs written.")


if __name__ == "__main__":
    main()
