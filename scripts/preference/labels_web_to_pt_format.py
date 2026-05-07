"""Convert the web labeller's pickle to the PT human_label/<env>/ format.

Web labeller output (from `label_web.py`):
    {
        'labels':        [int|None, ...],   # -1 / 0 / 1 / None(=skip)
        'pair_starts_a': [int, ...],
        'pair_starts_b': [int, ...],
        'time_sec':      [float, ...],
        'metadata':      {hdf5, query_len, fps, ...},
    }

PT expects in `human_label/<env>/seed_<N>/`:
    indices_num{M}_q{L}      pickled np.int32 (M,)   start indices of seg-A
    indices_2_num{M}_q{L}    pickled np.int32 (M,)   start indices of seg-B
    label_human              pickled list[int] of length M, in {-1, 0, 1}

Skipped pairs are dropped. M = number of labelled (non-None) pairs.

Usage
-----
    python -m scripts.preference.labels_web_to_pt_format \
        --labels $SCRATCH/PT/lunarlander/labels/lunarlander-medium-replay-v2/seed_0/labels.pkl \
        --output_dir human_label/lunarlander-medium-replay-v2/seed_0
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels, "rb") as f:
        web = pickle.load(f)

    labels = web["labels"]
    a = web["pair_starts_a"]
    b = web["pair_starts_b"]
    keep = [i for i, l in enumerate(labels) if l is not None]
    if not keep:
        raise SystemExit("no labelled (non-Skip) pairs in this file.")

    sel_labels = [int(labels[i]) for i in keep]
    sel_a = np.asarray([a[i] for i in keep], dtype=np.int32)
    sel_b = np.asarray([b[i] for i in keep], dtype=np.int32)

    M = len(sel_labels)
    L = web.get("metadata", {}).get("query_len", "?")
    suffix = f"num{M}_q{L}"
    fa = args.output_dir / f"indices_{suffix}"
    fb = args.output_dir / f"indices_2_{suffix}"
    fl = args.output_dir / "label_human"

    with open(fa, "wb") as g: pickle.dump(sel_a, g)
    with open(fb, "wb") as g: pickle.dump(sel_b, g)
    with open(fl, "wb") as g: pickle.dump(sel_labels, g)

    n_skipped = len(labels) - M
    counts = {v: sel_labels.count(v) for v in (-1, 0, 1)}
    print(f"[convert] kept {M} labelled, dropped {n_skipped} skips. counts={counts}")
    print(f"[convert] wrote {fa.name}, {fb.name}, label_human → {args.output_dir}")


if __name__ == "__main__":
    main()
