#!/usr/bin/env python3
"""EXAMPLE CLIPS ONLY — sticky actions / action delay, PT (preference) side.

Rolls the expert policy through the sticky/delay wrappers and renders a 100-frame,
20 fps clip (5.000 s, 0.40x real time) — the exact shape of a real PT preference
segment — so the difficulty can be shown to a supervisor before committing to a
dataset. This does NOT build preference pairs, labels, or training data, and it
touches nothing under frame_blanking/ or region_mask/.

On the PT side the difficulty is injected INDIRECTLY: the compared trajectories are
*generated* under the modified dynamics, then rendered. The human labeller watches at
normal speed with a perfect view — so for PT this is a data-distribution change, not a
perceptual difficulty. (For GAIL it is a genuine control difficulty, applied live.)
State that asymmetry honestly when presenting.

Env note: uses gymnasium + SB3 (the expert's home), i.e. the imitation-gail env, NOT the
PT gym-0.23 env. That is fine for an illustrative clip; obs/reward are never persisted.

    /scratch/marzii/envs/imitation-gail/bin/python \
        scripts/preference/gen_control_example_clips.py --out_dir figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_wrappers import make_control_corruptor

POLICY = "/scratch/marzii/imitation_runs/expert/lunarlander/4615187/policies/final/model.zip"
PHYSICS_FPS = 50   # LunarLander: 1 step = 1/50 s of game time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    p.add_argument("--policy", type=str, default=POLICY)
    p.add_argument("--n_frames", type=int, default=100, help="clip length (a PT segment = 100)")
    p.add_argument("--fps", type=int, default=20, help="playback fps of the clip (PT default 20)")
    p.add_argument("--seed", type=int, default=0, help="env seed; same for every variant")
    p.add_argument("--sticky_p", type=float, default=0.5)
    p.add_argument("--delay_k", type=int, default=3)
    p.add_argument("--sticky_seed", type=int, default=0)
    p.add_argument("--full_episode", action="store_true",
                   help="Roll until the episode terminates (capped at --n_frames) instead of "
                        "forcing exactly --n_frames. A 100-frame PT segment is only 2.0 s of game "
                        "time — too short for a control corruption to visibly change the landing. "
                        "Use this to show the OUTCOME; use the default to show what a real "
                        "preference segment looks like.")
    p.add_argument("--suffix", type=str, default="", help="appended to output filenames")
    return p.parse_args()


def roll(policy, seed, n_frames, sticky_p, sticky_seed, delay_k, full_episode=False):
    """Roll the expert under (sticky_p, delay_k).

    Returns (frames, episode_return, n_overridden, n_steps, ended).
    full_episode=False reproduces a real 100-frame PT segment (resetting if the episode
    happens to end early); True stops at termination so the LANDING is visible.
    """
    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    apply_ctrl, _ = make_control_corruptor(sticky_p, sticky_seed, delay_k, lambda: 0)

    obs, _ = env.reset(seed=seed)
    frames, total, overridden, ended = [env.render()], 0.0, 0, False
    while len(frames) < n_frames:
        act, _ = policy.predict(obs, deterministic=True)
        act = int(act)
        eff = int(apply_ctrl(act))
        overridden += (eff != act)
        obs, rew, term, trunc, _ = env.step(eff)
        total += rew
        frames.append(env.render())
        if term or trunc:
            if full_episode:
                ended = True
                break
            obs, _ = env.reset(seed=seed)   # keep the segment full-length
    env.close()
    return frames[:n_frames], total, overridden, len(frames) - 1, ended


def write_mp4(frames, path, fps):
    import imageio_ffmpeg as iff
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1: without it imageio pads 600x400 up to 608x400 (libx264 likes
    # multiples of 16), and the example clips would not match the real 600x400 PT clips.
    writer = iff.write_frames(str(path), (w, h), fps=fps, codec="libx264",
                              pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                              macro_block_size=1, quality=7)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()


def main():
    args = parse_args()
    from stable_baselines3 import PPO
    policy = PPO.load(args.policy)

    lag_ms = 1000.0 * args.delay_k / args.fps
    sfx = args.suffix
    variants = [
        (f"pt_control_clean_{args.fps}fps{sfx}",                    0.0,           0,
         "clean expert (baseline: sticky_p=0, delay_k=0 -> corruptor is the identity)"),
        (f"pt_sticky_p{int(args.sticky_p*100)}_{args.fps}fps{sfx}", args.sticky_p, 0,
         f"sticky actions, p={args.sticky_p}"),
        (f"pt_delay_k{args.delay_k}_{args.fps}fps{sfx}",            0.0,           args.delay_k,
         f"action delay, k={args.delay_k} steps "
         f"({args.delay_k * 1000 / PHYSICS_FPS:.0f} ms game time, {lag_ms:.0f} ms at {args.fps} fps)"),
    ]

    print(f"policy: {args.policy}")
    print(f"mode: {'full episode (stop at landing/crash)' if args.full_episode else 'fixed segment'}, "
          f"cap {args.n_frames} frames @ {args.fps} fps "
          f"({args.fps/PHYSICS_FPS:.2f}x real time)")
    print("All variants share seed, policy and initial state; they diverge only where the "
          "corruptor changes an action.\n")
    for name, p, k, desc in variants:
        frames, ret, overridden, steps, ended = roll(
            policy, args.seed, args.n_frames, p, args.sticky_seed, k, args.full_episode)
        out = args.out_dir / f"{name}.mp4"
        write_mp4(frames, out, args.fps)
        pct = 100.0 * overridden / max(1, steps)
        print(f"  {out.name:34s} return {ret:+8.1f}  steps {steps:4d}"
              f"{' (terminated)' if ended else ''}  "
              f"actions changed {overridden:3d}/{steps} ({pct:4.1f}%)\n"
              f"      {desc}")


if __name__ == "__main__":
    main()
