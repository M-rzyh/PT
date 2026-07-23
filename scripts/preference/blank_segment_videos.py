#!/usr/bin/env python3
"""Blank frames in already-extracted REAL-frame preference videos (frame-blanking, Exp 1).

Post-processes extract_segment_videos.py output IN PLACE: for each seg_{A,B}.mp4 it blacks
out a reproducible subset of frames (via ffmpeg drawbox), so the human labeler can't see the
lander on those frames -> degraded ("blind") preferences. NEW file; does not modify
extract_segment_videos.py or the mixture.

  python scripts/preference/blank_segment_videos.py \
     --videos_dir <extract output_dir> --num_query N --query_len 100 \
     --blank_mode stochastic --blank_prob 0.5 --blank_seed 0 [--batch_idx 0]
"""
import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np
import imageio_ffmpeg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_dir", required=True, type=Path)
    p.add_argument("--num_query", type=int, default=10)
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--blank_mode", choices=["stochastic", "deterministic", "block"], default="block")
    p.add_argument("--blank_prob", type=float, default=0.5)
    p.add_argument("--blank_k", type=int, default=2)
    p.add_argument("--block_len", type=int, default=10,
                   help="block mode: black out contiguous runs of this many frames "
                        "(~blank_prob of the blocks). At 20 fps, B=10 -> 0.5 s blackouts.")
    p.add_argument("--blank_seed", type=int, default=0)
    return p.parse_args()


def blanked_frames(L, mode, p, k, rng, block_len=10):
    if mode == "deterministic":
        return [i for i in range(L) if i % int(k) == 0]
    if mode == "block":
        nblocks = (L + block_len - 1) // block_len
        blank_blk = rng.random(nblocks) < float(p)
        return [i for i in range(L) if blank_blk[i // block_len]]
    return [i for i in range(L) if rng.random() < float(p)]


def blank_video(ffmpeg, path, idxs):
    """Black out frames `idxs` (0-based) of `path`, in place, via a single ffmpeg pass."""
    if not idxs:
        return
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{expr}'"
    tmp = path.with_suffix(".blank.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], check=True)
    tmp.replace(path)


def main():
    args = parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    batch = args.videos_dir / f"batch_{args.batch_idx:03d}"
    total_blanked = total_frames = 0
    for q in range(args.num_query):
        for side, tag in ((0, "A"), (1, "B")):
            seg = batch / f"pair_{q:03d}" / f"seg_{tag}.mp4"
            if not seg.exists():
                print(f"  MISSING {seg}"); continue
            seed = (args.blank_seed * 100003 + args.batch_idx * 1009 + q * 7 + side) % (2 ** 32)
            rng = np.random.default_rng(seed)
            idxs = blanked_frames(args.query_len, args.blank_mode, args.blank_prob,
                                  args.blank_k, rng, args.block_len)
            blank_video(ffmpeg, seg, idxs)
            total_blanked += len(idxs); total_frames += args.query_len
    # CRITICAL: repoint index.pkl at the LOCAL (blanked) batch dirs. extract_segment_videos
    # writes absolute paths there; if we blanked a COPY, label_web would otherwise follow
    # those paths back to the original CLEAN videos and show no blanking.
    local_batches = sorted(str(d.resolve()) for d in args.videos_dir.glob("batch_*") if d.is_dir())
    with open(args.videos_dir / "index.pkl", "wb") as g:
        pickle.dump(local_batches, g)

    print(f"[blank] {args.num_query} pairs, {args.blank_mode} p={args.blank_prob} "
          f"-> blanked {total_blanked}/{total_frames} frames ({total_blanked/max(1,total_frames):.1%}); "
          f"index.pkl repointed to local blanked batch")


if __name__ == "__main__":
    main()
