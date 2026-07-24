#!/usr/bin/env python3
"""Verify the built lander-vanish preference sets before any human time is spent on them.

A bad set is expensive: the human labels 100 pairs and only afterwards do we find the clips
were never masked, or that the lander is still traceable. Every check below corresponds to a
failure that actually happened during development:

  geometry      600x400 / 20 fps / 100 frames — imageio pads to 608 without macro_block_size=1
  index.pkl     must point at the LOCAL batch, else label_web silently plays the CLEAN videos
  metadata      n_pairs must match the pairs on disk (label_web loops range(n_pairs))
  hull removed  no hull-coloured blob on blanked frames
  no sky dots   engine particles used to survive when the lander was off-screen — the worst
                leak, since a trail of dots points straight at a lander you can't see
  terrain kept  the whole point of vanishing (vs blanking) is that the ground stays visible

READING THE OUTPUT: these clips are re-encoded to H.264, which sprinkles isolated 1-3 px
specks around high-contrast edges. A real remnant is a cluster (the grey particles that once
slipped through were 12 px each), so only blobs of MIN_BLOB or more fail the set; loose
specks are reported for information.

Reports the ACHIEVED blanked fraction per set: with b=5 over 100 frames there are only 20
coin flips, so the realised fraction scatters around the nominal level (the p50 proof-of-
concept came out 40%). That is variance, not a bug — but it belongs on the record.

    /scratch/marzii/envs/pt/bin/python scripts/preference/verify_vanish_sets.py
    /scratch/marzii/envs/pt/bin/python scripts/preference/verify_vanish_sets.py --levels 25
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import imageio_ffmpeg
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blank_lander_videos import HULL, blanked_frames  # noqa: E402  (same-dir helper)

MIN_BLOB = 5       # a few-px fleck is encoder ringing; a real remnant is a cluster (12+ px)

ROOT = Path("/scratch/marzii/PT/lunarlander/lander_vanish/pt_human")
LEVELS = [25, 50, 75]
N, QLEN, BLOCK, SEED = 100, 100, 5, 0
SAMPLE_PAIRS = 12          # clips are ~identical in kind; a sample catches systematic faults


def read(path):
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = rd.__next__()
    w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd], meta


def check_set(pct):
    d = ROOT / f"videos_human_n100_vanish{pct}"
    print(f"\n=== {d.name} ===")
    if not d.is_dir():
        print("  MISSING — not built"); return False
    ok = True

    # --- index.pkl must reference the local batch, not the clean source ---
    idx = pickle.load(open(d / "index.pkl", "rb"))
    local = all(str(d.resolve()) in str(p) for p in idx)
    print(f"  index.pkl -> {idx} {'OK' if local else 'STALE (points outside this set!)'}")
    ok &= local

    # --- metadata must describe exactly the pairs on disk ---
    meta = pickle.load(open(d / "batch_000" / "metadata.pkl", "rb"))
    on_disk = len(list((d / "batch_000").glob("pair_*")))
    print(f"  metadata n_pairs={meta['n_pairs']}  pairs on disk={on_disk}  "
          f"{'OK' if meta['n_pairs'] == on_disk == N else 'MISMATCH'}")
    ok &= (meta["n_pairs"] == on_disk == N)

    tot_blank = tot_frames = 0
    hull_left = sky_dots = strong_dots = 0
    ground_missing = 0
    geom_bad = []

    for q in range(0, N, max(1, N // SAMPLE_PAIRS)):
        for side, tag in ((0, "A"), (1, "B")):
            seg = d / "batch_000" / f"pair_{q:03d}" / f"seg_{tag}.mp4"
            frames, m = read(seg)
            if (m["size"], round(m["fps"]), len(frames)) != ((600, 400), 20, QLEN):
                geom_bad.append((seg.name, m["size"], m["fps"], len(frames)))

            # recompute the schedule with the SAME seed derivation the masker used
            seed = (SEED * 100003 + 0 * 1009 + q * 7 + side) % (2 ** 32)
            idxs = set(blanked_frames(QLEN, pct / 100.0, np.random.default_rng(seed), BLOCK))
            tot_blank += len(idxs); tot_frames += len(frames)

            for i in idxs:
                a = frames[i].astype(int)
                # a stray hull-coloured pixel is encoder ringing; a BLOB is a real remnant
                hull_px = np.abs(a - HULL).sum(2) < 90
                if hull_px.any():
                    lab, nlab = ndimage.label(hull_px)
                    if nlab and max(ndimage.sum(hull_px, lab, range(1, nlab + 1))) >= MIN_BLOB:
                        hull_left += int(hull_px.sum())
                # A "sky dot" is ANY lit pixel left in the upper half — deliberately NOT a
                # tint test. The mask used to key on tint, and this check did too, so both
                # were blind to the same thing: particles fade to pure grey (51,51,51), and
                # a check that shares the mask's assumption cannot catch the mask's mistake.
                # The upper half holds no legitimate scenery, so anything lit there is a
                # remnant. Only blobs count; isolated specks are H.264 ringing.
                s = a.sum(2)
                lit = (s > 40) & (s < 720)
                sky_dots += int(lit[:200].sum())
                sky = lit[:200]
                if sky.any():
                    lab, nlab = ndimage.label(sky)
                    if nlab:
                        sizes = ndimage.sum(sky, lab, range(1, nlab + 1))
                        strong_dots += int(sum(x for x in sizes if x >= MIN_BLOB))
            for i in range(0, len(frames), 20):     # terrain must survive everywhere
                if (frames[i].astype(int).sum(2) > 720).sum() < 500:
                    ground_missing += 1

    frac = tot_blank / max(1, tot_frames)
    print(f"  blanked fraction: {frac:.1%} (nominal {pct}%) — b={BLOCK} gives only "
          f"{QLEN // BLOCK} flips/clip, so scatter is expected")
    print(f"  hull blobs left on blanked     : {hull_left} px  {'OK' if hull_left == 0 else 'LEAK'}")
    print(f"  visible particle remnants      : {strong_dots} px  "
          f"{'OK' if strong_dots == 0 else 'LEAK'}")
    print(f"    (weak H.264 speckle, ignored : {sky_dots} px)")
    print(f"  frames with no visible ground  : {ground_missing}  "
          f"{'OK' if ground_missing == 0 else 'TERRAIN DESTROYED'}")
    if geom_bad:
        print(f"  GEOMETRY WRONG on {len(geom_bad)}: {geom_bad[:3]}")
    ok &= (hull_left == 0 and strong_dots == 0 and ground_missing == 0 and not geom_bad)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=LEVELS,
                    help="check only these levels (useful while the build is still running)")
    args = ap.parse_args()
    results = {p: check_set(p) for p in args.levels}
    print("\n" + "=" * 60)
    for p, r in results.items():
        print(f"  vanish{p}: {'PASS' if r else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
