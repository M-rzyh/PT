"""Extract preference-query seg_A.mp4 / seg_B.mp4 from per-episode rollout
mp4s produced by `scripts/offline_data/rollout_with_video.py`.

For each pair (start_a, start_b) drawn from `indices_num{N}_q{L}` /
`indices_2_num{N}_q{L}`, the script:
  1. Looks up which episode mp4 contains [start, start+L) using the
     `episodes/index.pkl` written by rollout_with_video.
  2. Uses ffmpeg `-ss …/-vframes L` to slice that span out into the
     BPref3-compatible layout that `label_web.py` consumes:
         <out>/batch_NNN/pair_NNN/seg_A.mp4
         <out>/batch_NNN/pair_NNN/seg_B.mp4
         <out>/batch_NNN/metadata.pkl
         <out>/index.pkl

Usage
-----
    python -m scripts.preference.extract_segment_videos \
        --rollout_dir $SCRATCH/PT/lunarlander/labels/lunarlander-medium-v2/seed_0/rollout \
        --query_dir   human_label/lunarlander-medium-v2/seed_0/queries \
        --output_dir  $SCRATCH/PT/lunarlander/labels/lunarlander-medium-v2/seed_0/videos \
        --num_query 10 --query_len 100
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout_dir", required=True, type=Path,
                   help="Dir from rollout_with_video.py (contains episodes/ and an HDF5).")
    p.add_argument("--query_dir", required=True, type=Path,
                   help="Dir with indices_num{N}_q{L} pickles.")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_query", type=int, default=10)
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--batch_idx", type=int, default=0)
    return p.parse_args()


def find_episode(episodes: list[dict], step: int, length: int) -> dict:
    """Return the episode dict containing [step, step+length-1], else raise."""
    for ep in episodes:
        if ep["start_step"] <= step and step + length - 1 <= ep["end_step"]:
            return ep
    raise RuntimeError(
        f"no single episode contains the segment [{step}, {step + length - 1}] — "
        f"sample_query_indices.py should guarantee this; was the HDF5 from a "
        f"different rollout than the rollout_dir?"
    )


def slice_segment(ffmpeg: str, src_mp4: Path, frame_offset: int, length: int,
                  fps: int, out_mp4: Path) -> None:
    """Re-encode a length-`length` segment starting at frame `frame_offset`.

    Re-encode (not stream-copy) keeps timing exact and the output seekable.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    start_sec = frame_offset / fps
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.4f}", "-i", str(src_mp4),
        "-vframes", str(length),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # Load query indices.
    suffix = f"num{args.num_query}_q{args.query_len}"
    fn1 = args.query_dir / f"indices_{suffix}"
    fn2 = args.query_dir / f"indices_2_{suffix}"
    with open(fn1, "rb") as g: i1 = np.asarray(pickle.load(g), dtype=np.int64)
    with open(fn2, "rb") as g: i2 = np.asarray(pickle.load(g), dtype=np.int64)
    assert len(i1) == args.num_query and len(i2) == args.num_query

    # Load episode index.
    ep_index_path = args.rollout_dir / "episodes" / "index.pkl"
    with open(ep_index_path, "rb") as g: episodes = pickle.load(g)
    fps = int(episodes[0]["n_frames"] / max(1, episodes[0]["end_step"] - episodes[0]["start_step"] + 1) + 0.5) or 20
    # The above is fragile; the rollout HDF5 has fps in its attrs — read that.
    import h5py
    hdf5_glob = list(args.rollout_dir.glob("lunarlander-*.hdf5"))
    if hdf5_glob:
        with h5py.File(hdf5_glob[0], "r") as f:
            fps = int(f.attrs.get("fps", fps))
    print(f"[extract] {len(episodes)} episodes, fps={fps}")

    batch_dir = args.output_dir / f"batch_{args.batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for q in range(args.num_query):
        for tag, start in (("A", int(i1[q])), ("B", int(i2[q]))):
            ep = find_episode(episodes, start, args.query_len)
            frame_offset = start - ep["start_step"]
            pair_dir = batch_dir / f"pair_{q:03d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = pair_dir / f"seg_{tag}.mp4"
            print(f"[extract] pair {q:03d} seg {tag}: step={start} → "
                  f"episode {ep['episode_idx']:>5d} (frames {frame_offset}…{frame_offset + args.query_len - 1})")
            slice_segment(ffmpeg, Path(ep["mp4_path"]), frame_offset,
                          args.query_len, fps, out_mp4)

    metadata = dict(
        batch_idx=args.batch_idx,
        n_pairs=args.num_query,
        query_len=args.query_len,
        fps=fps,
        rollout_dir=str(args.rollout_dir),
        query_dir=str(args.query_dir),
        indices_1=i1.tolist(),
        indices_2=i2.tolist(),
    )
    with open(batch_dir / "metadata.pkl", "wb") as g:
        pickle.dump(metadata, g)
    with open(args.output_dir / "index.pkl", "wb") as g:
        pickle.dump([str(batch_dir.resolve())], g)
    print(f"[extract] wrote {args.num_query * 2} mp4s to {batch_dir}")


if __name__ == "__main__":
    main()
