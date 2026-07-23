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
    # --- Frame-blanking difficulty (Exp 1): blank frames the HUMAN sees --------
    # Opt-in; default 'none' = unchanged. Blanked frames hide the lander so the
    # labeler can't see it (harder to give preferences).
    p.add_argument("--blank_mode", choices=["none", "stochastic", "deterministic"],
                   default="none", help="Blank displayed frames (human difficulty). Default off.")
    p.add_argument("--blank_prob", type=float, default=0.5)
    p.add_argument("--blank_k", type=int, default=2)
    p.add_argument("--blank_seed", type=int, default=0,
                   help="Base seed; each segment gets a distinct reproducible mask.")
    return p.parse_args()


def load_indices(query_dir: Path, num_query: int, query_len: int) -> tuple[np.ndarray, np.ndarray]:
    fn1 = query_dir / f"indices_num{num_query}_q{query_len}"
    fn2 = query_dir / f"indices_2_num{num_query}_q{query_len}"
    with open(fn1, "rb") as g: i1 = pickle.load(g)
    with open(fn2, "rb") as g: i2 = pickle.load(g)
    return np.asarray(i1, dtype=np.int64), np.asarray(i2, dtype=np.int64)


def _triangle_vertices(x: float, y: float, ang: float, size: float = 0.15) -> np.ndarray:
    """Return (3, 2) lander-body vertices for centre (x, y) at orientation `ang`
    (radians; 0 = nose up). Built locally then translated, so it works
    regardless of matplotlib's patch internals."""
    # local frame: nose pointing +y
    local = np.array([
        [0.0,        +size],          # nose
        [-0.7 * size, -0.6 * size],   # bottom-left
        [+0.7 * size, -0.6 * size],   # bottom-right
    ])
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (local @ R.T) + np.array([x, y])


def render_segment(obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                   out_path: Path, fps: int, blank_frames: np.ndarray = None) -> None:
    """obs: (T, 8), act: (T, 2), rew: (T,) — write an mp4 to out_path.

    blank_frames: optional (T,) bool mask. True frames hide the lander/trail/velocity/
    state HUD → the human sees a blank 'no-signal' frame (frame-blanking difficulty).
    None = no blanking (unchanged).
    """
    T = len(obs)
    xs, ys = obs[:, 0], obs[:, 1]
    vxs, vys = obs[:, 2], obs[:, 3]
    angles = obs[:, 4]
    leg_l, leg_r = obs[:, 6] > 0.5, obs[:, 7] > 0.5
    cum_r = np.cumsum(rew)

    # Fixed bounds chosen to cover the data's 1st…99th-percentile range
    # observed in lunarlander-medium-replay-v2 (x ∈ [-1, 1], y ∈ [-0.4, 2.0])
    # plus margin.
    XL, XH = -1.5, 1.5
    YL, YH = -0.4, 2.2

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_xlim(XL, XH); ax.set_ylim(YL, YH)
    ax.set_aspect("equal"); ax.set_facecolor("#0f172a")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, 1, 2])
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Ground + landing pad
    ax.plot([XL, -0.2, -0.2, 0.2, 0.2, XH], [0.06, 0.06, 0, 0, 0.06, 0.06],
            color="#22c55e", lw=2)
    ax.add_patch(mpatches.Rectangle((-0.2, 0.0), 0.4, 0.04,
                                     color="#facc15", zorder=1))
    # Goal flags
    ax.plot([-0.2, -0.2], [0, 0.18], color="#facc15", lw=1.5)
    ax.plot([+0.2, +0.2], [0, 0.18], color="#facc15", lw=1.5)

    trail, = ax.plot([], [], color="#94a3b8", lw=1.2, alpha=0.7, zorder=2)
    body = mpatches.Polygon(_triangle_vertices(0, 0, 0),
                             closed=True, color="#60a5fa", zorder=4)
    ax.add_patch(body)
    leg_l_art, = ax.plot([], [], lw=3, color="#94a3b8", zorder=3,
                          solid_capstyle="round")
    leg_r_art, = ax.plot([], [], lw=3, color="#94a3b8", zorder=3,
                          solid_capstyle="round")
    # Velocity arrow (semi-transparent, scaled for visibility)
    vel_arrow = mpatches.FancyArrowPatch((0, 0), (0, 0),
                                          arrowstyle="-|>",
                                          mutation_scale=10,
                                          color="#fbbf24",
                                          alpha=0.8, zorder=5)
    ax.add_patch(vel_arrow)

    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                  color="#e2e8f0", va="top", ha="left", fontsize=9,
                  family="monospace",
                  bbox=dict(facecolor="#0f172a", edgecolor="#334155",
                            alpha=0.85, pad=4))

    leg_offset = 0.13  # leg length

    def update(t: int):
        if blank_frames is not None and bool(blank_frames[t]):
            # blanked frame: hide the lander & per-frame state -> human sees no signal
            for art in (body, leg_l_art, leg_r_art, trail, vel_arrow):
                art.set_visible(False)
            hud.set_text("(frame blanked)")
            return body, leg_l_art, leg_r_art, trail, vel_arrow, hud
        for art in (body, leg_l_art, leg_r_art, trail, vel_arrow):
            art.set_visible(True)
        x, y = float(xs[t]), float(ys[t])
        ang = float(angles[t])
        # Body triangle
        body.set_xy(_triangle_vertices(x, y, ang))
        # Legs: angled at ±35° from "down" in body frame
        for sign, art, contact in ((-1, leg_l_art, leg_l[t]),
                                    (+1, leg_r_art, leg_r[t])):
            leg_dir = ang + sign * np.deg2rad(35) - np.pi / 2  # 0 angle => down
            x2 = x + leg_offset * np.cos(leg_dir)
            y2 = y + leg_offset * np.sin(leg_dir)
            art.set_data([x, x2], [y, y2])
            art.set_color("#ef4444" if contact else "#94a3b8")
            art.set_linewidth(4 if contact else 2.5)
        # Trail
        trail.set_data(xs[:t + 1], ys[:t + 1])
        # Velocity arrow (scale for visibility, cap length)
        vmag = float(np.hypot(vxs[t], vys[t]))
        if vmag > 1e-3:
            vx_s = float(vxs[t]) * 0.15
            vy_s = float(vys[t]) * 0.15
            vel_arrow.set_positions((x, y), (x + vx_s, y + vy_s))
            vel_arrow.set_alpha(min(0.9, 0.3 + vmag / 3.0))
        else:
            vel_arrow.set_positions((x, y), (x, y))
        hud.set_text(
            f"t={t:>3d}/{T-1}  x={x:+.2f}  y={y:+.2f}\n"
            f"v=({vxs[t]:+.2f},{vys[t]:+.2f})  ang={np.rad2deg(ang):+5.1f}°\n"
            f"a=({act[t,0]:+.2f},{act[t,1]:+.2f})\n"
            f"r_t={rew[t]:+6.2f}   Σr={cum_r[t]:+7.1f}"
        )
        return body, leg_l_art, leg_r_art, trail, vel_arrow, hud

    a = anim.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = anim.FFMpegWriter(fps=fps, codec="libx264", bitrate=800,
                                extra_args=["-pix_fmt", "yuv420p"])
    a.save(out_path, writer=writer)
    plt.close(fig)


def _seg_blank_mask(args, batch, pair, side, T):
    """Reproducible (T,) blank mask for one segment; None if blanking is off."""
    if args.blank_mode == "none":
        return None
    if args.blank_mode == "deterministic":
        return (np.arange(T) % int(args.blank_k)) == 0
    seed = (args.blank_seed * 100003 + batch * 1009 + pair * 7 + side) % (2 ** 32)
    return np.random.default_rng(seed).random(T) < float(args.blank_prob)


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
        act_all = f["actions"][:]
        rew_all = f["rewards"][:]

    batch_dir = args.output_dir / f"batch_{args.batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for q in range(args.num_query):
        pair_dir = batch_dir / f"pair_{q:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        s1, s2 = int(i1[q]), int(i2[q])
        L = args.query_len
        seg1 = (obs_all[s1:s1 + L], act_all[s1:s1 + L], rew_all[s1:s1 + L])
        seg2 = (obs_all[s2:s2 + L], act_all[s2:s2 + L], rew_all[s2:s2 + L])
        print(f"[render] pair {q:03d}: A start={s1} ΣrA={float(seg1[2].sum()):+.1f}  "
              f"B start={s2} ΣrB={float(seg2[2].sum()):+.1f}")
        bf1 = _seg_blank_mask(args, args.batch_idx, q, 0, L)
        bf2 = _seg_blank_mask(args, args.batch_idx, q, 1, L)
        render_segment(*seg1, pair_dir / "seg_A.mp4", args.fps, blank_frames=bf1)
        render_segment(*seg2, pair_dir / "seg_B.mp4", args.fps, blank_frames=bf2)

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
