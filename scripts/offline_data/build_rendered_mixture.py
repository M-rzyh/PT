"""Build a rendered mixture HDF5 from the 5 rendered source companions.

Recipe mirrors build_mixture_dataset.py (the non-rendered mixture):
    232.5K each from {random, medium, medium-expert, expert}
    +  70K from medium-replay  (= all of it, which is what we have)
    = ~1M total.

medium-expert is constructed inline by sampling 116.25K from rendered
medium + 116.25K from rendered expert (whole episodes, no overlap with
the medium / expert solo slots).

Output:
    seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
    seed_0/render/mixture-v2/episodes/index.pkl   (each entry's mp4_path
        points back to the source rendered companion's per-episode mp4 —
        we don't duplicate mp4 files, just reference them)
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mix_seed", type=int, default=0)
    p.add_argument("--per_variant", type=int, default=232_500,
                   help="Target transitions per non-mr variant.")
    # --- optional overrides so the SAME builder can assemble a delayed mixture (action-delay
    # study) without a second script. Defaults reproduce the original clean 5-source build. ---
    p.add_argument("--no-medium-replay", dest="no_medium_replay", action="store_true",
                   help="Drop the medium-replay slice entirely → a 4-source mixture "
                        "(random+medium+medium-expert+expert). Used by the action-delay study, "
                        "where medium-replay (a training buffer) has no delayed analogue. To "
                        "keep ~1M total, pass --per_variant 250000.")
    p.add_argument("--random-dir", type=Path, default=None,
                   help="Override the random source dir (default render_root/random-v2-mixprep).")
    p.add_argument("--medium-dir", type=Path, default=None,
                   help="Override the medium source dir (e.g. a delayed render).")
    p.add_argument("--expert-dir", type=Path, default=None,
                   help="Override the expert source dir (e.g. a delayed render).")
    p.add_argument("--medium-replay-dir", type=Path, default=None,
                   help="Override the medium-replay source dir.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the output mixture dir (default render_root/mixture-v2).")
    return p.parse_args()


def trajectory_bounds(terminals: np.ndarray, timeouts: np.ndarray) -> list[tuple[int, int]]:
    ends = (terminals | timeouts).astype(bool)
    out, start = [], 0
    for i, e in enumerate(ends):
        if e:
            out.append((start, i)); start = i + 1
    if start <= len(ends) - 1: out.append((start, len(ends) - 1))
    return out


def select_episodes(episodes_index: list[dict], bounds: list[tuple[int, int]],
                    target: int, rng, exclude: set | None = None) -> list[int]:
    """Sample whole-episode indices until cumulative transitions >= target.
    Returns indices into `episodes_index` / `bounds`."""
    n = len(bounds)
    perm = rng.permutation(n)
    if exclude is None: exclude = set()
    chosen = []
    total = 0
    for ti in perm:
        if int(ti) in exclude: continue
        s, e = bounds[ti]
        chosen.append(int(ti))
        total += e - s + 1
        if total >= target: break
    return chosen


def load_source(src_dir: Path):
    """Returns (hdf5_arrays_dict, bounds, episodes_index, src_fps)."""
    # The rendered companion has its HDF5 named after the variant.
    hdf5_glob = list(src_dir.glob("lunarlander-*.hdf5"))
    if not hdf5_glob:
        raise FileNotFoundError(f"no HDF5 in {src_dir}")
    hdf5 = hdf5_glob[0]
    with h5py.File(hdf5, "r") as f:
        arrs = {k: f[k][:] for k in
                ("observations", "actions", "rewards", "next_observations",
                 "terminals", "timeouts")}
        # Carry the rollout's real fps through instead of asserting 20 downstream:
        # extract_segment_videos.py reads this attr to seek into the source mp4s, so a
        # silent disagreement with rollout_with_video.py --fps would cut the wrong frames.
        src_fps = int(f.attrs.get("fps", 20))
    bounds = trajectory_bounds(arrs["terminals"].astype(bool),
                                arrs["timeouts"].astype(bool))
    # index.pkl tells us how each episode maps to an mp4.
    with open(src_dir / "episodes/index.pkl", "rb") as g:
        ep_idx = pickle.load(g)
    # Sanity: episodes_index length should match bounds length (close enough).
    return arrs, bounds, ep_idx, src_fps


def slice_episode(arrs: dict[str, np.ndarray], bound: tuple[int, int]) -> dict[str, np.ndarray]:
    s, e = bound
    return {k: v[s:e + 1] for k, v in arrs.items()}


def main() -> None:
    args = parse_args()
    SCRATCH = Path(os.environ["SCRATCH"])
    render_root = SCRATCH / f"PT/lunarlander/seed_{args.seed}/render"
    out_dir = args.out_dir if args.out_dir is not None else render_root / "mixture-v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source companions (rendered variants). Each is overridable so a delayed build can point
    # medium/expert at delayed renders while reusing the (delay-invariant) random render.
    SRC = {
        "random":        args.random_dir or render_root / "random-v2-mixprep",
        "medium":        args.medium_dir or render_root / "medium-v2-mixprep",
        "medium-replay": args.medium_replay_dir or render_root / "medium-replay-v2",
        "expert":        args.expert_dir or render_root / "expert-v2-mixprep",
    }
    if args.no_medium_replay:
        del SRC["medium-replay"]
        print("[mix] --no-medium-replay: building a 4-source mixture (no medium-replay)")
    for v, d in SRC.items():
        if not d.exists():
            raise SystemExit(f"missing rendered companion: {d}")
        print(f"[src] {v:14s} {d}")

    rng = np.random.default_rng(args.mix_seed)

    # ---- Load all 4 (random, medium, mr, expert) and sample episode lists -----
    loaded: dict[str, tuple[dict, list, list]] = {}
    src_fpss: dict[str, int] = {}
    for v, d in SRC.items():
        arrs, bounds, ep_idx, src_fps = load_source(d)
        loaded[v] = (arrs, bounds, ep_idx)
        src_fpss[v] = src_fps
        print(f"[load] {v:14s} N={arrs['rewards'].shape[0]} bounds={len(bounds)} "
              f"mp4s={len(ep_idx)} fps={src_fps}")

    # The mixture's clips come from these mp4s, so they must all share one frame rate.
    if len(set(src_fpss.values())) != 1:
        raise SystemExit(f"[fatal] source rollouts disagree on fps: {src_fpss} — "
                         f"extract_segment_videos.py would seek to the wrong frames.")
    mix_fps = next(iter(src_fpss.values()))

    chosen_eps: dict[str, list[int]] = {}
    chosen_eps["random"] = select_episodes(loaded["random"][2], loaded["random"][1],
                                            args.per_variant, rng)
    # medium: pick enough for solo (per_variant) + medium-expert half (per_variant//2)
    medium_total = args.per_variant + args.per_variant // 2
    medium_pick = select_episodes(loaded["medium"][2], loaded["medium"][1],
                                   medium_total, rng)
    # expert: same arithmetic
    expert_total = args.per_variant + args.per_variant // 2
    expert_pick = select_episodes(loaded["expert"][2], loaded["expert"][1],
                                   expert_total, rng)
    # Split medium_pick into solo + me-half by cumulative transitions
    def split_at_target(picks, bounds, target):
        cum = 0; cut = 0
        for i, ti in enumerate(picks):
            s, e = bounds[ti]
            cum += e - s + 1
            if cum >= target:
                cut = i + 1; break
        return picks[:cut], picks[cut:]
    chosen_eps["medium"], me_med_half = split_at_target(
        medium_pick, loaded["medium"][1], args.per_variant)
    chosen_eps["expert"], me_exp_half = split_at_target(
        expert_pick, loaded["expert"][1], args.per_variant)
    if not args.no_medium_replay:
        # medium-replay: take ALL episodes (~70K transitions).
        mr_n = len(loaded["medium-replay"][1])
        chosen_eps["medium-replay"] = list(range(mr_n))

    # ---- Assemble in order: random → medium → [medium-replay →] me → expert -----
    mix_obs, mix_act, mix_rew, mix_nxt, mix_term, mix_to = [], [], [], [], [], []
    mix_sources = []
    out_index: list[dict] = []
    cur_pos = 0
    out_episode_idx = 0
    VARIANT_NAMES = (["random", "medium", "medium-expert", "expert"] if args.no_medium_replay
                     else ["random", "medium", "medium-replay", "medium-expert", "expert"])

    def add_episodes(variant_idx: int, src_variant_for_data: str,
                     picks: list[int]):
        nonlocal cur_pos, out_episode_idx
        arrs, bounds, ep_idx = loaded[src_variant_for_data]
        for ti in picks:
            s, e = bounds[ti]
            n = e - s + 1
            mix_obs.append(arrs["observations"][s:e + 1])
            mix_act.append(arrs["actions"][s:e + 1])
            mix_rew.append(arrs["rewards"][s:e + 1])
            mix_nxt.append(arrs["next_observations"][s:e + 1])
            mix_term.append(arrs["terminals"][s:e + 1])
            mix_to.append(arrs["timeouts"][s:e + 1])
            mix_sources.append(np.full(n, variant_idx, dtype=np.int8))
            # Find the source mp4 corresponding to this episode (via ep_idx).
            src_ep = ep_idx[ti] if ti < len(ep_idx) else None
            src_mp4 = src_ep["mp4_path"] if src_ep else None
            out_index.append({
                "episode_idx": out_episode_idx,
                "start_step":  cur_pos,
                "end_step":    cur_pos + n - 1,
                "n_frames":    n,
                "mp4_path":    src_mp4,
                "source":      src_variant_for_data,
                "source_episode_idx": ti,
            })
            cur_pos += n
            out_episode_idx += 1

    # variant_idx must match the position in VARIANT_NAMES for the `sources` tags to be correct.
    idx = {name: i for i, name in enumerate(VARIANT_NAMES)}
    add_episodes(idx["random"], "random", chosen_eps["random"])
    add_episodes(idx["medium"], "medium", chosen_eps["medium"])
    if not args.no_medium_replay:
        add_episodes(idx["medium-replay"], "medium-replay", chosen_eps["medium-replay"])
    # medium-expert: half from medium, half from expert
    add_episodes(idx["medium-expert"], "medium", me_med_half)
    add_episodes(idx["medium-expert"], "expert", me_exp_half)
    add_episodes(idx["expert"], "expert", chosen_eps["expert"])

    # ---- Concat + write HDF5 ----------------------------------------------
    obs = np.concatenate(mix_obs).astype(np.float32)
    act = np.concatenate(mix_act).astype(np.float32)
    rew = np.concatenate(mix_rew).astype(np.float32)
    nxt = np.concatenate(mix_nxt).astype(np.float32)
    term = np.concatenate(mix_term).astype(bool)
    to   = np.concatenate(mix_to).astype(bool)
    src  = np.concatenate(mix_sources)
    N = obs.shape[0]
    print(f"\n[mix] total = {N} transitions, {len(out_index)} episodes")
    counts = {VARIANT_NAMES[i]: int((src == i).sum()) for i in range(len(VARIANT_NAMES))}
    print(f"[mix] counts per variant: {counts}")

    hdf5_out = out_dir / "lunarlander-mixture-v2.hdf5"
    with h5py.File(hdf5_out, "w") as f:
        f.create_dataset("observations",      data=obs,  compression="gzip")
        f.create_dataset("actions",           data=act,  compression="gzip")
        f.create_dataset("rewards",           data=rew,  compression="gzip")
        f.create_dataset("next_observations", data=nxt,  compression="gzip")
        f.create_dataset("terminals",         data=term, compression="gzip")
        f.create_dataset("timeouts",          data=to,   compression="gzip")
        f.create_dataset("sources",           data=src,  compression="gzip")
        f.attrs["n_transitions"] = N
        f.attrs["mix_seed"] = args.mix_seed
        f.attrs["source_seed"] = args.seed
        f.attrs["variant_names"] = "|".join(VARIANT_NAMES)
        f.attrs["fps"] = mix_fps   # carried from the source rollouts, not hardcoded

    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episodes/index.pkl", "wb") as g:
        pickle.dump(out_index, g)
    print(f"[done] wrote {hdf5_out}")
    print(f"[done] wrote {out_dir / 'episodes/index.pkl'}  ({len(out_index)} entries)")


if __name__ == "__main__":
    main()
