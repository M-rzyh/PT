#!/usr/bin/env python3
"""Derive the PT 0% (clean) vanish labels for free from the existing clean 350.

The vanish study's budget is N=100 preferences. The 0% condition is the same task the human
already labelled at N=350 on clean video, so instead of re-labelling 100 clean pairs we take
the FIRST 100 of that set — which is exactly why the 25/50/75 vanish video sets were built
from pair_000..pair_099 of the same clean batch. Same pairs, same clips, same human, the only
difference across the four conditions being how much of the lander was visible.

The correspondence is checked, not assumed: the subset's segment start indices must match
indices_1/indices_2 in the vanish sets' metadata.pkl entry-for-entry. If the clean label file
ever contained a skipped pair the 1:1 alignment would silently break, so that is verified too
(M must equal the pair count).

Writes lunarlander-mixture-v2-humanvanish0-s{0..9}/ with the same labels in every seed dir —
the supervision is collected once and the 10 seeds measure training variance, matching the
convention on both arms.

    /scratch/marzii/envs/pt/bin/python scripts/preference/make_vanish0_labels.py
    /scratch/marzii/envs/pt/bin/python scripts/preference/make_vanish0_labels.py --dry_run
"""
import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np

LABEL_ROOT = Path("/scratch/marzii/PT/human_label")
SRC = "lunarlander-mixture-v2-humanblock0-s0"          # clean human labels, N=350
DST_FMT = "lunarlander-mixture-v2-humanvanish0-s{}"
VANISH_SET = Path("/scratch/marzii/PT/lunarlander/lander_vanish/pt_human"
                  "/videos_human_n100_vanish25/batch_000/metadata.pkl")
N, QLEN, SEEDS = 100, 100, 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry_run", action="store_true", help="verify only, write nothing")
    p.add_argument("--force", action="store_true", help="overwrite existing seed dirs")
    return p.parse_args()


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    src = LABEL_ROOT / SRC
    lab = load(src / "label_human")
    ia = load(src / f"indices_num350_q100")
    ib = load(src / f"indices_2_num350_q100")
    print(f"source {SRC}: M={len(lab)} labels, indices {ia.shape}/{ib.shape}")

    if not (len(lab) == len(ia) == len(ib) == 350):
        raise SystemExit(f"expected 350 aligned entries, got {len(lab)}/{len(ia)}/{len(ib)} — "
                         "a skipped pair would break the 1:1 pair correspondence")

    # --- the subset must be the SAME pairs the vanish videos were built from ---
    meta = load(VANISH_SET)
    m1, m2 = np.asarray(meta["indices_1"]), np.asarray(meta["indices_2"])
    if not (len(m1) == N and np.array_equal(ia[:N], m1) and np.array_equal(ib[:N], m2)):
        raise SystemExit(
            "MISMATCH: the first 100 clean labels do not correspond to the vanish sets' "
            f"pairs.\n  labels  a[:3]={ia[:3]} b[:3]={ib[:3]}\n  videos  a[:3]={m1[:3]} "
            f"b[:3]={m2[:3]}\nThe 0% point would not be comparable to 25/50/75.")
    print(f"OK: first {N} labels match videos_human_n100_vanish25 pair_000..pair_{N-1:03d}")

    sub_l, sub_a, sub_b = list(lab[:N]), ia[:N].astype(np.int32), ib[:N].astype(np.int32)
    counts = {v: sub_l.count(v) for v in (-1, 0, 1)}
    print(f"subset: {N} labels, counts={counts}  (PT convention: 0=A, 1=B, -1=tie)")

    if args.dry_run:
        print("\n--dry_run: nothing written")
        return

    suffix = f"num{N}_q{QLEN}"
    for s in range(SEEDS):
        d = LABEL_ROOT / DST_FMT.format(s)
        if d.exists():
            if not args.force:
                print(f"  {d.name} EXISTS — skipping (pass --force to overwrite)")
                continue
            shutil.rmtree(d)
        d.mkdir(parents=True)
        with open(d / f"indices_{suffix}", "wb") as g:
            pickle.dump(sub_a, g)
        with open(d / f"indices_2_{suffix}", "wb") as g:
            pickle.dump(sub_b, g)
        with open(d / "label_human", "wb") as g:
            pickle.dump(sub_l, g)
    print(f"\nwrote {SEEDS} seed dirs: {DST_FMT.format('{0..%d}' % (SEEDS - 1))}")


if __name__ == "__main__":
    main()
