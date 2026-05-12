"""Convert the 100 (25 oracle + 75 human) PEBBLE labels stored in a BPref3
checkpoint into the file layout the PT pipeline already understands —
no PT-side code changes.

Inputs:
    The PEBBLE checkpoint at
      `<pebble_run>/seed_<S>/pebble/checkpoint.pt`
    contains a `reward.pref_buffer` dict with:
        seg1, seg2  float32 (BUFFER, 50, 10)   = obs(8) + action(2), padded
        label       float32 (BUFFER, 1)        in {0.0=A pref, 1.0=B pref}
        len1, len2  int32   (BUFFER,)          true segment length (always 50 here)
        index       int                        # of populated entries

Outputs (per seed):
    <output_root>/<env_tag>.hdf5     A synthetic D4RL-schema dataset where each
                                      saved segment occupies a contiguous 51-step
                                      span (50 real steps + 1 dummy terminal so
                                      PT's assert max(trj_len) > query_len passes).
    human_label/<env_tag>/
        indices_num100_q50           pickled np.int32 (100,)  start of seg-A
        indices_2_num100_q50         pickled np.int32 (100,)  start of seg-B
        label_human                  pickled list[int] in {0,1,-1}  PT convention
        provenance.json              source = "oracle" for first 25, "human" for rest

PT's `load_queries_with_indices` already handles this exact layout (it slices
`dataset[start:start+query_len]` for each pair), so the only flags we change
in `new_preference_reward_main.py` invocation are `--query_len=50` and
`--num_query=100`.
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
    p.add_argument("--pebble_checkpoint", required=True, type=Path,
                   help="Path to BPref3 PEBBLE checkpoint.pt")
    p.add_argument("--n_pairs", type=int, default=100)
    p.add_argument("--n_oracle", type=int, default=25,
                   help="First N pairs are tagged oracle in provenance.json")
    p.add_argument("--query_len", type=int, default=50)
    p.add_argument("--env_tag", default="lunarlander-pebble100-s0",
                   help="Used as both the synthetic-HDF5 basename and the "
                        "human_label/<env_tag>/ subdir name.")
    p.add_argument("--hdf5_out_dir", required=True, type=Path,
                   help="Directory to write the synthetic HDF5 into.")
    p.add_argument("--label_out_root", default=Path("human_label"), type=Path,
                   help="Root for human_label/<env_tag>/.")
    return p.parse_args()


def load_pebble_pref_buffer(ckpt_path: Path, n: int):
    """Return (seg1, seg2, label) sliced to the first `n` entries."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pb = ck["reward"]["pref_buffer"]
    seg1 = np.asarray(pb["seg1"])[:n]
    seg2 = np.asarray(pb["seg2"])[:n]
    lbls = np.asarray(pb["label"])[:n].flatten()
    return seg1, seg2, lbls


def main() -> None:
    args = parse_args()

    seg1, seg2, raw_labels = load_pebble_pref_buffer(args.pebble_checkpoint, args.n_pairs)
    assert seg1.shape == (args.n_pairs, args.query_len, 10), (
        f"unexpected segment shape {seg1.shape}; expected ({args.n_pairs}, {args.query_len}, 10)"
    )
    obs_dim, act_dim = 8, 2

    # ---- 1. Label-convention conversion --------------------------------------
    # PEBBLE: 0.0 = A preferred, 1.0 = B preferred (no ties in this run).
    # PT:     0   = A preferred, 1   = B preferred, -1 = tie.
    # The two conventions agree on 0 and 1; ties become -1 if any exist.
    pt_labels: list[int] = []
    for lf in raw_labels:
        if lf == 0.0:
            pt_labels.append(0)
        elif lf == 1.0:
            pt_labels.append(1)
        else:
            pt_labels.append(-1)  # treat anything else as tie
    n_oracle = args.n_oracle
    print(f"[labels] {len(pt_labels)} pairs  "
          f"counts: A-pref={pt_labels.count(0)}  B-pref={pt_labels.count(1)}  tie={pt_labels.count(-1)}")
    print(f"[labels] first {n_oracle} tagged oracle, last {len(pt_labels)-n_oracle} tagged human")

    # ---- 2. Synthetic HDF5 ---------------------------------------------------
    # Pack 100 pairs * 2 segments = 200 segments. Each segment occupies a
    # length-(query_len+1) trajectory so that max(trj_len) > query_len, which
    # is required by JaxPref/reward_transform.py:load_queries_with_indices.
    # Layout:
    #     [seg_A_0  pad   seg_B_0  pad   seg_A_1  pad   ...]
    #         |<--+51-->|     |<--+51-->|
    # indices_1[k] = (2k)   * (query_len + 1)
    # indices_2[k] = (2k+1) * (query_len + 1)
    seg_block = args.query_len + 1                                # 51
    N = args.n_pairs * 2 * seg_block                              # 100*2*51 = 10200

    obs = np.zeros((N, obs_dim), dtype=np.float32)
    act = np.zeros((N, act_dim), dtype=np.float32)
    nxt = np.zeros((N, obs_dim), dtype=np.float32)
    rew = np.zeros((N,), dtype=np.float32)
    terms = np.zeros((N,), dtype=bool)
    times = np.zeros((N,), dtype=bool)

    indices_1 = np.zeros(args.n_pairs, dtype=np.int32)
    indices_2 = np.zeros(args.n_pairs, dtype=np.int32)

    for k in range(args.n_pairs):
        for which, sa in (("A", seg1[k]), ("B", seg2[k])):
            traj_start = (2 * k + (0 if which == "A" else 1)) * seg_block
            seg_end_exclusive = traj_start + args.query_len      # 50 real steps
            # Real frames
            obs[traj_start:seg_end_exclusive] = sa[:, :obs_dim]
            act[traj_start:seg_end_exclusive] = sa[:, obs_dim:obs_dim + act_dim]
            # next_observations: shift obs forward by 1 within the segment;
            # last step's next_obs = last obs (boundary).
            nxt[traj_start:seg_end_exclusive - 1] = sa[1:, :obs_dim]
            nxt[seg_end_exclusive - 1] = sa[-1, :obs_dim]
            # Dummy pad transition that carries the trajectory terminal.
            obs[seg_end_exclusive] = sa[-1, :obs_dim]
            nxt[seg_end_exclusive] = sa[-1, :obs_dim]
            terms[seg_end_exclusive] = True
            # rewards stay zero — PT in --use_human_label=True mode uses
            # `label_human` rather than reward sums for the labels.
            if which == "A":
                indices_1[k] = traj_start
            else:
                indices_2[k] = traj_start

    hdf5_path = args.hdf5_out_dir / f"{args.env_tag}.hdf5"
    args.hdf5_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hdf5] writing N={N} transitions across {args.n_pairs*2} length-{seg_block} trajectories → {hdf5_path}")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("observations",      data=obs,   compression="gzip")
        f.create_dataset("actions",           data=act,   compression="gzip")
        f.create_dataset("rewards",           data=rew,   compression="gzip")
        f.create_dataset("next_observations", data=nxt,   compression="gzip")
        f.create_dataset("terminals",         data=terms, compression="gzip")
        f.create_dataset("timeouts",          data=times, compression="gzip")
        f.attrs["source"]   = str(args.pebble_checkpoint)
        f.attrs["n_pairs"]  = args.n_pairs
        f.attrs["query_len"] = args.query_len
        f.attrs["seg_block"] = seg_block

    # ---- 3. PT-style label / indices files -----------------------------------
    label_dir = args.label_out_root / args.env_tag
    label_dir.mkdir(parents=True, exist_ok=True)
    fn1 = label_dir / f"indices_num{args.n_pairs}_q{args.query_len}"
    fn2 = label_dir / f"indices_2_num{args.n_pairs}_q{args.query_len}"
    fl  = label_dir / "label_human"
    with open(fn1, "wb") as g: pickle.dump(indices_1, g)
    with open(fn2, "wb") as g: pickle.dump(indices_2, g)
    with open(fl,  "wb") as g: pickle.dump(pt_labels, g)
    provenance = dict(
        source=str(args.pebble_checkpoint),
        env_tag=args.env_tag,
        n_pairs=args.n_pairs,
        query_len=args.query_len,
        per_pair_source=(["oracle"] * args.n_oracle
                         + ["human"] * (args.n_pairs - args.n_oracle)),
        counts=dict(
            A_pref=pt_labels.count(0),
            B_pref=pt_labels.count(1),
            tie=pt_labels.count(-1),
        ),
        hdf5=str(hdf5_path),
    )
    # Provenance lives next to the HDF5 (NOT inside label_dir), because PT's
    # loader hard-codes `sorted(os.listdir(base_path))` to unpack into exactly
    # 3 filenames — any 4th entry breaks the destructuring.
    prov_path = args.hdf5_out_dir / f"{args.env_tag}.provenance.json"
    with open(prov_path, "w") as g:
        json.dump(provenance, g, indent=2)
    print(f"[labels] wrote {fn1.name}, {fn2.name}, label_human → {label_dir}")
    print(f"[labels] wrote provenance.json → {prov_path}")


if __name__ == "__main__":
    main()
