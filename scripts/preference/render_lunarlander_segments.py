"""Render LunarLander preference-query segments to mp4 (matplotlib animation).

Reads the indices files produced by `sample_query_indices.py` plus the source
HDF5, and writes one mp4 per segment in the BPref3-compatible layout:

    <out_dir>/batch_000/pair_NNN/seg_A.mp4
    <out_dir>/batch_000/pair_NNN/seg_B.mp4
    <out_dir>/batch_000/metadata.pkl   {batch_idx, n_pairs, query_len, hdf5, ...}
    <out_dir>/index.pkl                 [<absolute path of batch_000>]

The web labeller (label_web.py) consumes exactly this layout, so this script
+ the labeller cover the whole render-and-label half of Phase C.

Each frame draws:
    - the lander as a triangle, rotated by `obs[4]` (angle)
    - both legs (red when in contact)
    - a green ground line at y=0 with a goalpad in [-0.2, 0.2]
    - a fading position trail
    - a header with the segment's cumulative reward so far

Usage
-----
    python -m scripts.preference.render_lunarlander_segments \
        --hdf5 $SCRATCH/PT/lunarlander/seed_0/lunarlander-medium-replay-v2.hdf5 \
        --query_dir human_label/lunarlander-medium-replay-v2/seed_0 \
        --output_dir $SCRATCH/PT/lunarlander/labels/lunarlander-medium-replay-v2/seed_0/videos \
        --num_query 10 --query_len 100 --fps 20
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import h5py
import imageio_ffmpeg  # ensures ffmpeg is available
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--query_dir", required=True, type=Path,
                   help="Dir containing indices_num{N}_q{L} and indices_2_num{N}_q{L}.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where to write batch_NNN/pair_NNN/seg_{A,B}.mp4.")
    p.add_argument("--num_query", type=int, default=10)
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--batch_idx", type=int, default=0)
    return p.parse_args()


def load_indices(query_dir: Path, num_query: int, query_len: int) -> tuple[np.ndarray, np.ndarray]:
    fn1 = query_dir / f"indices_num{num_query}_q{query_len}"
    fn2 = query_dir / f"indices_2_num{num_query}_q{query_len}"
    with open(fn1, "rb") as g: i1 = pickle.load(g)
    with open(fn2, "rb") as g: i2 = pickle.load(g)
    return np.asarray(i1, dtype=np.int64), np.asarray(i2, dtype=np.int64)


def render_segment(obs: np.ndarray, rew: np.ndarray, out_path: Path, fps: int) -> None:
    """obs: (T, 8), rew: (T,) — write an mp4 to out_path."""
    T = len(obs)
    xs, ys = obs[:, 0], obs[:, 1]
    angles = obs[:, 4]
    leg_l, leg_r = obs[:, 6] > 0.5, obs[:, 7] > 0.5
    cum_r = np.cumsum(rew)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.2, 1.6)
    ax.set_aspect("equal"); ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values(): spine.set_color("#334155")

    # ground
    ax.plot([-1.6, -0.2, -0.2, 0.2, 0.2, 1.6], [0.05, 0.05, 0, 0, 0.05, 0.05],
            color="#22c55e", lw=2)
    ax.add_patch(mpatches.Rectangle((-0.2, 0.0), 0.4, 0.04, color="#facc15"))

    trail, = ax.plot([], [], color="#64748b", lw=1, alpha=0.6)
    body = mpatches.RegularPolygon((0, 0), 3, radius=0.08, color="#60a5fa")
    ax.add_patch(body)
    leg_l_art, = ax.plot([], [], lw=3, color="#94a3b8")
    leg_r_art, = ax.plot([], [], lw=3, color="#94a3b8")
    title = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                    color="#e2e8f0", va="top", ha="left", fontsize=10,
                    family="monospace")

    def update(t: int):
        x, y, ang = xs[t], ys[t], angles[t]
        body.xy = (x, y)
        body.orientation = ang + np.pi / 2  # nose up at angle=0
        # Legs splayed ±25°, length 0.12
        for sign, art, contact in ((-1, leg_l_art, leg_l[t]), (+1, leg_r_art, leg_r[t])):
            base_a = ang - np.pi / 2 + sign * np.deg2rad(25)
            x2, y2 = x + 0.12 * np.cos(base_a), y + 0.12 * np.sin(base_a)
            art.set_data([x, x2], [y, y2])
            art.set_color("#ef4444" if contact else "#94a3b8")
        trail.set_data(xs[:t + 1], ys[:t + 1])
        title.set_text(
            f"t={t:>3d}/{T-1}  pos=({x:+.2f},{y:+.2f})  ang={np.rad2deg(ang):+5.1f}°\n"
            f"r_t={rew[t]:+6.2f}  Σr={cum_r[t]:+7.1f}"
        )
        return body, leg_l_art, leg_r_art, trail, title

    a = anim.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = anim.FFMpegWriter(fps=fps, codec="libx264", bitrate=800,
                                extra_args=["-pix_fmt", "yuv420p"])
    a.save(out_path, writer=writer)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    os.environ.setdefault(
        "IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe()
    )
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    i1, i2 = load_indices(args.query_dir, args.num_query, args.query_len)
    assert len(i1) == args.num_query and len(i2) == args.num_query

    with h5py.File(args.hdf5, "r") as f:
        obs_all = f["observations"][:]
        rew_all = f["rewards"][:]

    batch_dir = args.output_dir / f"batch_{args.batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for q in range(args.num_query):
        pair_dir = batch_dir / f"pair_{q:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        s1, s2 = int(i1[q]), int(i2[q])
        seg1_obs = obs_all[s1:s1 + args.query_len]
        seg1_rew = rew_all[s1:s1 + args.query_len]
        seg2_obs = obs_all[s2:s2 + args.query_len]
        seg2_rew = rew_all[s2:s2 + args.query_len]
        print(f"[render] pair {q:03d}: A start={s1} ΣrA={float(seg1_rew.sum()):+.1f}  "
              f"B start={s2} ΣrB={float(seg2_rew.sum()):+.1f}")
        render_segment(seg1_obs, seg1_rew, pair_dir / "seg_A.mp4", args.fps)
        render_segment(seg2_obs, seg2_rew, pair_dir / "seg_B.mp4", args.fps)

    metadata = dict(
        batch_idx=args.batch_idx,
        n_pairs=args.num_query,
        query_len=args.query_len,
        fps=args.fps,
        hdf5=str(args.hdf5),
        query_dir=str(args.query_dir),
        indices_1=i1.tolist(),
        indices_2=i2.tolist(),
    )
    with open(batch_dir / "metadata.pkl", "wb") as g:
        pickle.dump(metadata, g)
    with open(args.output_dir / "index.pkl", "wb") as g:
        pickle.dump([str(batch_dir.resolve())], g)
    print(f"[render] wrote {args.num_query * 2} mp4s to {batch_dir}, plus metadata.pkl + index.pkl")


if __name__ == "__main__":
    main()
