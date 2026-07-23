#!/usr/bin/env python3
"""Spatially mask already-extracted REAL-frame preference videos (region masking, Sweep 1).

Post-processes extract_segment_videos.py output IN PLACE: for each seg_{A,B}.mp4 it paints a
SINGLE fixed black rectangle over EVERY frame of that segment (via ffmpeg drawbox), so the human
labeler has a persistent spatial blind spot -> degraded ("occluded") preferences. This is the
spatial twin of blank_segment_videos.py (which blacks whole frames in time); here one rectangle
is black in space, constant across the whole segment. NEW file; does not modify
extract_segment_videos.py, blank_segment_videos.py, or the mixture.

Region geometry (Sweep 1):
  size     : rectangle area = region_frac * frame area -> each side scaled by sqrt(region_frac)
             (0.25 -> 300x200, 0.50 -> 424x283, 0.75 -> 520x346 on a 600x400 frame).
  location : top-left drawn uniformly in [0,W-w] x [0,H-h], FIXED per segment, seeded
             deterministically from (region_seed, batch, pair, side) so it is identical across
             all training seeds (seeds = training variance only, exactly like the labels).

  python scripts/preference/mask_region_videos.py \
     --videos_dir <extract output_dir> --num_query N \
     --region_frac 0.5 --region_seed 0 [--frame_w 600 --frame_h 400 --batch_idx 0]
"""
import argparse
import math
import pickle
import subprocess
from pathlib import Path

import numpy as np
import imageio_ffmpeg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_dir", required=True, type=Path)
    p.add_argument("--num_query", type=int, default=10)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--region_frac", type=float, required=True,
                   help="fraction of the frame AREA to cover (0.25/0.5/0.75). 0 = no mask.")
    p.add_argument("--frame_w", type=int, default=600)
    p.add_argument("--frame_h", type=int, default=400)
    p.add_argument("--region_seed", type=int, default=0)
    p.add_argument("--outline", type=str, default=None, metavar="COLOR",
                   help="Draw an outline (e.g. 'red') around the masked box so it is "
                        "distinguishable from the black sky. Default None = no outline "
                        "(unchanged). Cosmetic only: it does not change what is hidden.")
    p.add_argument("--outline_thickness", type=int, default=3,
                   help="Outline thickness in pixels (default 3). Drawn INSIDE the box, so the "
                        "occluded area is unchanged.")
    return p.parse_args()


def region_box(frac, W, H, rng):
    """Return (x0, y0, w, h): a rectangle of area ~frac*W*H, placed uniformly inside the frame."""
    s = math.sqrt(float(frac))
    w = max(1, min(W, int(round(s * W))))
    h = max(1, min(H, int(round(s * H))))
    x0 = int(rng.integers(0, W - w + 1))
    y0 = int(rng.integers(0, H - h + 1))
    return x0, y0, w, h


def mask_video(ffmpeg, path, box, outline=None, outline_thickness=3):
    """Paint `box` (x0,y0,w,h) black over ALL frames of `path`, in place, one ffmpeg pass.

    `outline` (e.g. 'red') adds a border drawn INSIDE the box, on top of the black fill, so
    the human can tell the mask apart from the black sky. The occluded pixels are the same
    either way — the ring is opaque too, it just isn't black.
    """
    x0, y0, w, h = box
    vf = f"drawbox=x={x0}:y={y0}:w={w}:h={h}:color=black:t=fill"
    if outline:
        vf += f",drawbox=x={x0}:y={y0}:w={w}:h={h}:color={outline}:t={outline_thickness}"
    tmp = path.with_suffix(".mask.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], check=True)
    tmp.replace(path)


def main():
    args = parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    batch = args.videos_dir / f"batch_{args.batch_idx:03d}"
    covered = 0.0
    n = 0
    for q in range(args.num_query):
        for side, tag in ((0, "A"), (1, "B")):
            seg = batch / f"pair_{q:03d}" / f"seg_{tag}.mp4"
            if not seg.exists():
                print(f"  MISSING {seg}"); continue
            seed = (args.region_seed * 100003 + args.batch_idx * 1009 + q * 7 + side) % (2 ** 32)
            rng = np.random.default_rng(seed)
            box = region_box(args.region_frac, args.frame_w, args.frame_h, rng)
            mask_video(ffmpeg, seg, box, outline=args.outline,
                       outline_thickness=args.outline_thickness)
            covered += (box[2] * box[3]) / (args.frame_w * args.frame_h); n += 1
    # CRITICAL (same as blank_segment_videos.py): repoint index.pkl at the LOCAL masked batch,
    # else label_web follows stale absolute paths back to the clean originals and shows no mask.
    local_batches = sorted(str(d.resolve()) for d in args.videos_dir.glob("batch_*") if d.is_dir())
    with open(args.videos_dir / "index.pkl", "wb") as g:
        pickle.dump(local_batches, g)

    print(f"[mask] {n} segments, region_frac={args.region_frac} "
          f"-> mean coverage {covered/max(1,n):.1%}; index.pkl repointed to local masked batch")


if __name__ == "__main__":
    main()
