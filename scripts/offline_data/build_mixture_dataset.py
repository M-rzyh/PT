"""Build a 1M-transition mixture HDF5 from the 5 existing source variants.

Recipe (per the user's spec):
- 70K from medium-replay (= entire seed-0 medium-replay HDF5).
- 232,500 from each of {random, medium, medium-expert, expert}
  → 4 * 232,500 = 930,000, plus 70K from medium-replay = 1,000,000.
- The base "200K from each + 130K spread back across the 4 non-mr variants"
  collapses to "232,500 from each non-mr variant" → same outcome.

Sampling is at the TRAJECTORY level (not per-transition), so PT can still
draw contiguous `query_len` segments without crossing episode boundaries.

Output:
    $SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s{SEED}.hdf5
    $SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s{SEED}.metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


VARIANT_NAMES = ["random", "medium", "medium-replay", "medium-expert", "expert"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0,
                   help="Which source-seed's HDF5s to draw from (default 0).")
    p.add_argument("--out_dir", type=Path,
                   default=Path(os.environ.get("SCRATCH", "/scratch/marzii")) / "PT/lunarlander/mixture")
    p.add_argument("--mix_seed", type=int, default=0,
                   help="RNG seed for trajectory sampling.")
    p.add_argument("--total_target", type=int, default=1_000_000)
    p.add_argument("--mr_take_all", action="store_true", default=True,
                   help="medium-replay contributes its entire dataset (~70K).")
    return p.parse_args()


def trajectory_bounds(terminals: np.ndarray, timeouts: np.ndarray) -> list[tuple[int, int]]:
    ends = (terminals | timeouts).astype(bool)
    out, start = [], 0
    for i, e in enumerate(ends):
        if e:
            out.append((start, i))
            start = i + 1
    if start <= len(ends) - 1:
        out.append((start, len(ends) - 1))
    return out


def select_trajectories(bounds: list[tuple[int, int]], target: int, rng) -> tuple[list[tuple[int, int]], int]:
    perm = rng.permutation(len(bounds))
    selected, total = [], 0
    for ti in perm:
        s, e = bounds[ti]
        n = e - s + 1
        selected.append((s, e))
        total += n
        if total >= target:
            break
    return selected, total


def load_source(hdf5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        return {
            "observations":      f["observations"][:],
            "actions":           f["actions"][:],
            "rewards":           f["rewards"][:],
            "next_observations": f["next_observations"][:],
            "terminals":         f["terminals"][:].astype(bool),
            "timeouts":          f["timeouts"][:].astype(bool),
        }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scratch = Path(os.environ.get("SCRATCH", "/scratch/marzii"))
    rng = np.random.default_rng(args.mix_seed)

    # Total = 1M. medium-replay takes whatever it has (~70K). The remaining
    # budget is split equally across the 4 non-mr variants.
    mr_src = scratch / f"PT/lunarlander/seed_{args.seed}/lunarlander-medium-replay-v2.hdf5"
    with h5py.File(mr_src, "r") as f:
        mr_size = f["rewards"].shape[0]
    non_mr_target_per = (args.total_target - mr_size) // 4
    print(f"[plan] total target = {args.total_target}")
    print(f"[plan] medium-replay: take all {mr_size} transitions")
    print(f"[plan] each of {{random, medium, medium-expert, expert}}: target {non_mr_target_per} transitions")
    print(f"[plan] expected total ≈ {mr_size + 4 * non_mr_target_per}")

    # Per-source: load, choose trajectories, slice + concat.
    chunks: dict[str, dict[str, np.ndarray]] = {}
    chosen_meta: list[dict] = []
    for v in VARIANT_NAMES:
        src = scratch / f"PT/lunarlander/seed_{args.seed}/lunarlander-{v}-v2.hdf5"
        print(f"[load] {v}: {src}")
        d = load_source(src)
        bounds = trajectory_bounds(d["terminals"], d["timeouts"])

        if v == "medium-replay":
            # Take everything in original order.
            selected = bounds
            total = sum(e - s + 1 for s, e in selected)
        else:
            selected, total = select_trajectories(bounds, non_mr_target_per, rng)

        print(f"  {v}: {len(selected)} trajs, {total} transitions")
        chosen_meta.append(dict(variant=v, n_trajectories=len(selected), n_transitions=int(total)))

        # Slice each chosen trajectory and concatenate.
        chunks[v] = {k: np.concatenate([arr[s:e + 1] for s, e in selected])
                     for k, arr in d.items()}

    # Concatenate variants in fixed order, tagging source per row.
    mix = {k: [] for k in chunks["random"].keys()}
    sources = []
    for vi, v in enumerate(VARIANT_NAMES):
        for k in mix:
            mix[k].append(chunks[v][k])
        sources.append(np.full(chunks[v]["rewards"].shape[0], vi, dtype=np.int8))
    for k in mix:
        mix[k] = np.concatenate(mix[k])
    sources_arr = np.concatenate(sources)
    N = mix["rewards"].shape[0]
    print(f"[mix] N = {N} transitions")

    out_hdf5 = args.out_dir / f"lunarlander-mixture-v2-s{args.seed}.hdf5"
    with h5py.File(out_hdf5, "w") as f:
        f.create_dataset("observations",      data=mix["observations"].astype(np.float32),      compression="gzip")
        f.create_dataset("actions",           data=mix["actions"].astype(np.float32),           compression="gzip")
        f.create_dataset("rewards",           data=mix["rewards"].astype(np.float32),           compression="gzip")
        f.create_dataset("next_observations", data=mix["next_observations"].astype(np.float32), compression="gzip")
        f.create_dataset("terminals",         data=mix["terminals"].astype(bool),               compression="gzip")
        f.create_dataset("timeouts",          data=mix["timeouts"].astype(bool),                compression="gzip")
        # Provenance: row → variant idx (0..4) per VARIANT_NAMES
        f.create_dataset("sources",           data=sources_arr,                                 compression="gzip")
        f.attrs["n_transitions"] = N
        f.attrs["mix_seed"] = args.mix_seed
        f.attrs["source_seed"] = args.seed
        f.attrs["variant_names"] = "|".join(VARIANT_NAMES)
        f.attrs["per_variant_counts"] = [m["n_transitions"] for m in chosen_meta]

    meta = dict(
        out_hdf5=str(out_hdf5),
        total_transitions=int(N),
        source_seed=args.seed,
        mix_seed=args.mix_seed,
        variants=chosen_meta,
    )
    out_meta = args.out_dir / f"lunarlander-mixture-v2-s{args.seed}.metadata.json"
    with open(out_meta, "w") as g:
        json.dump(meta, g, indent=2)
    print(f"[done] wrote {out_hdf5}\n[done] wrote {out_meta}")


if __name__ == "__main__":
    main()
