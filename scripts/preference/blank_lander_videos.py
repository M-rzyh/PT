#!/usr/bin/env python3
"""Make the LANDER VANISH on blanked frames, keeping the ground visible.

The spatial-selective twin of blank_segment_videos.py. That script blacks out the WHOLE
frame during a blanked block; this one removes only the spacecraft (hull + legs + engine
plume) and fills the hole with the environment behind it, so the terrain, landing pad and
flags stay fully visible but the human cannot tell where the lander is.

WHY INPAINT AND NOT A BLACK BOX
  The sky renders pure black and the ground pure white, so a flat black fill would be
  invisible against sky but leave a tell-tale lander-shaped silhouette against the ground —
  leaking exactly the information we're removing. Instead each removed pixel is filled from
  the nearest surviving pixel in its own column, which reconstructs sky/ground correctly
  even where the lander straddles the horizon.

HOW THE LANDER IS FOUND (all measured on real clips)
  hull/legs  : render as a unique colour (128,102,230); sky is (0,0,0), ground (255,255,255),
               helipad flags (204,204,0) — nothing else uses the hull colour, so a colour
               match + largest-connected-component is reliable (~900-1100 px/frame).
  engine plume: particles fade (255,127,127) -> (51,51,51) via ttl and are emitted at the
               thrusters, so hull-only masking would leave a flame pointing at the lander.
               Picked up as non-background pixels near the hull centroid.
  fringe     : H.264 + anti-aliasing smear the sprite edge, so the mask is dilated a few px.

Blanked-block schedule is IDENTICAL to blank_segment_videos.py (same rng/seed derivation), so
at the same (blank_seed, blank_prob, block_len) this variant hides the lander on exactly the
frames the standard variant blacks out — making the two directly comparable.

  # single clip (proof-of-concept / example)
  python scripts/preference/blank_lander_videos.py --in_mp4 clean.mp4 --out_mp4 vanish.mp4 \
      --blank_prob 0.5 --block_len 5

  # a whole extracted set, in place (same CLI shape as blank_segment_videos.py)
  python scripts/preference/blank_lander_videos.py --videos_dir <dir> --num_query 350 \
      --query_len 100 --blank_prob 0.5 --block_len 5 --blank_seed 0
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import imageio_ffmpeg
from scipy import ndimage

from extract_segment_videos import find_episode   # maps a segment start step -> its episode

HULL = np.array([128, 102, 230])     # lander hull + legs (unique in the scene)
FLAG = np.array([204, 204, 0])       # helipad flags — scenery, must NOT be removed
SKY_MAX = 40                         # sum(rgb) below this == black sky
GROUND_MIN = 720                     # sum(rgb) above this == white ground


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_mp4", type=Path, help="single-clip mode: source clip")
    p.add_argument("--out_mp4", type=Path, help="single-clip mode: destination clip")
    p.add_argument("--videos_dir", type=Path, help="set mode: extract_segment_videos output dir")
    p.add_argument("--num_query", type=int, default=10)
    p.add_argument("--query_len", type=int, default=100)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--blank_prob", type=float, default=0.5)
    p.add_argument("--block_len", type=int, default=5,
                   help="blank contiguous runs of this many frames (~blank_prob of blocks). "
                        "At 20 fps, b=5 -> 0.25 s.")
    p.add_argument("--blank_seed", type=int, default=0)
    p.add_argument("--hull_tol", type=int, default=90, help="L1 colour tolerance for the hull")
    p.add_argument("--plume_radius", type=int, default=0,
                   help="px around the hull searched for engine particles; 0 = no limit (remove the whole plume trail)")
    p.add_argument("--dilate", type=int, default=3, help="px to grow the mask (edge fringe)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--plate_frames", type=int, default=48,
                   help="how many frames, spread across the episode, to median for the "
                        "background plate")
    p.add_argument("--no_plate", action="store_true",
                   help="disable background reconstruction and use the old column fill "
                        "(kept for comparison; it cannot represent the flags)")
    return p.parse_args()


def blanked_frames(L, p, rng, block_len):
    """Same block schedule as blank_segment_videos.py (mode='block')."""
    nblocks = (L + block_len - 1) // block_len
    blank_blk = rng.random(nblocks) < float(p)
    return [i for i in range(L) if blank_blk[i // block_len]]


def lander_mask(fr, hull_tol=90, plume_radius=130, dilate=3, min_px=60):
    """Boolean mask of everything that is NOT the static scene.

    KEEP-LIST, not a find-list. LunarLander's background is exactly three things: black sky,
    white ground, yellow flags. So anything else on screen IS the lander or its engine plume,
    and the mask is simply the complement of the background.

    This replaced a "find the lander" rule (match the hull's purple, then decide which other
    pixels look like flame). That rule has to JUDGE whether a pixel belongs to the lander, and
    the judgement was wrong twice: orange particles near the pad were mistaken for flags and
    protected, and fully-faded particles are pure grey (51,51,51) so a colour-tint test walked
    straight past them. A keep-list cannot make either mistake -- it never asks what a pixel
    is, only whether it is one of the three things known to belong here.

    The cost is that it also removes the terrain's anti-aliased outline, which is grey and so
    fails the keep-list too. That was unacceptable while the fill had to GUESS a colour, and
    is irrelevant now that build_plate() pastes the real background back: measured against the
    old rule, the repainted outline differs by mean 2.2/255 (max 17) and only 2 px in a frame
    flip between ground and sky. Deleting generously is safe once the fill is truthful.

    hull_tol / plume_radius / min_px are accepted and ignored, so existing callers and the
    IRL3 mirror keep the same signature.
    """
    a = fr.astype(int)
    s = a.sum(2)
    black = s <= SKY_MAX
    white = s >= GROUND_MIN
    # Flags are YELLOW (r == g, b low). The L1 tolerance stays generous to catch their
    # anti-aliased edges, but 200 alone also matches ORANGE engine particles -- (204,97,90)
    # is only L1 197 from (204,204,0) -- so require yellowness too. Measured: true flag
    # pixels have |r-g| <= 11, those particles |r-g| = 107.
    yellow = (np.abs(a - FLAG).sum(2) < 200) & (np.abs(a[:, :, 0] - a[:, :, 1]) < 40)
    yellow = ndimage.binary_dilation(yellow, iterations=3)   # keep their soft edges too
    # dilate: the sprite edge is smeared by anti-aliasing and H.264, so grow the mask a little
    return ndimage.binary_dilation(~(black | white | yellow), iterations=dilate)


def moving_mask(fr, dilate=3):
    """Narrow mask: pixels that belong to the LANDER specifically, not merely non-background.

    Two different questions need two different masks, and conflating them is a trap:

      ERASE  — "might this pixel be the lander?"  Answer generously (lander_mask, the
               keep-list): missing a lander pixel leaks its position.
      LEARN  — "is this pixel safe to record as background?"  Answer conservatively HERE:
               the keep-list also rejects the terrain's grey anti-aliased outline, so using
               it to build the plate would mean the outline is never learned, and every
               blanked frame would repaint the horizon with the snapped column fill. The
               horizon would then visibly change shape exactly when the lander is hidden —
               a cue perfectly correlated with the thing we are hiding.

    So this keeps the old discriminator: terrain anti-aliasing is grey AND borders the ground,
    whereas the lander and its plume are either colour-tinted or floating free in the sky.
    A mistake here is cheap — the worst case is that a faint particle is recorded into the
    plate for one frame, and the next frame overwrites it. Erasure is unaffected.
    """
    a = fr.astype(int)
    s = a.sum(2)
    tinted = (a.max(2) - a.min(2)) > 3
    yellow = (np.abs(a - FLAG).sum(2) < 200) & (np.abs(a[:, :, 0] - a[:, :, 1]) < 40)
    near_flag = ndimage.binary_dilation(yellow, iterations=3)
    far_from_ground = ~ndimage.binary_dilation(s >= GROUND_MIN, iterations=3)
    lander = (s > SKY_MAX) & (s < GROUND_MIN) & ~near_flag & (tinted | far_from_ground)
    return ndimage.binary_dilation(lander, iterations=dilate)


def build_plate(frames, sample=48, hull_tol=90, plume_radius=0, dilate=3):
    """Reconstruct the static background of an episode by looking across its frames.

    Within one episode the terrain, landing pad and flags never move — only the lander does.
    So for each pixel, take the frames where the lander is NOT covering it and use the median
    of those values. What's left is the scene as it would look with no lander in it.

    This replaces guessing the fill colour, which is where the two nastiest leaks came from:
    a column fill snapped to black/white turns any covered flag into a WHITE NOTCH (measured:
    up to 38% of the flag area, in 662 frames), and the notch marks the lander as surely as
    the lander would. Copying the real background has no such failure mode — there is nothing
    to infer.

    Why the MEDIAN and not a handful of hand-picked frames: if the lander happens to sit at a
    pixel in one of the sampled frames, a fixed sample copies the lander straight into the
    plate. The median throws it out as the outlier. This is a real risk, not a theoretical
    one — on pair_007 a 6-frame sample was still blocked on 1,265 pixels.

    Built with `moving_mask`, NOT the keep-list `lander_mask`: the keep-list rejects the
    terrain's grey outline, so using it here would leave the outline unlearned and every
    blanked frame would repaint the horizon by guesswork. See moving_mask's docstring.

    Returns (plate, valid) where `valid` marks pixels that were seen unobstructed at least
    once. Pixels outside `valid` are left to the caller's fallback.
    """
    idx = np.unique(np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int))
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    masks = np.stack([moving_mask(frames[i], dilate) for i in idx])
    stack[masks] = np.nan
    valid = ~np.isnan(stack[..., 0]).all(0)
    with np.errstate(all="ignore"):
        plate = np.nanmedian(stack, axis=0)
    plate = np.nan_to_num(plate).astype(np.uint8)
    return plate, valid


def inpaint_from_plate(fr, mask, plate, valid):
    """Paste the reconstructed background wherever the lander was, for pixels the plate saw."""
    out = fr.copy()
    take = mask & valid
    out[take] = plate[take]
    return out, mask & ~valid          # leftover = never seen unobstructed


def inpaint_columns(fr, mask):
    """Fill each masked pixel from the nearest unmasked pixel in the same column, SNAPPED to
    pure sky or pure ground.

    FALLBACK ONLY, now that build_plate() reconstructs the real background: this is used for
    the handful of pixels the plate never saw unobstructed. It cannot represent the flags —
    (204,204,0) has brightness 408 against a 382 threshold, so a covered flag snaps to WHITE.

    The scene is two flat regions (black sky / white ground), so a vertical nearest-neighbour
    fill picks the right side of the horizon. But copying the neighbour's exact value smears
    any contamination (a faint particle or edge fringe sitting just outside the mask) down the
    whole column as a visible vertical ghost streak. Snapping each filled pixel to pure black
    or pure white — whichever the neighbour is closer to — guarantees a clean, flat fill.
    """
    out = fr.copy()
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return out
    for x in np.unique(xs):
        bad = np.where(mask[:, x])[0]
        good = np.where(~mask[:, x])[0]
        if len(good) == 0:
            continue
        idx = np.searchsorted(good, bad)
        lo = np.clip(idx - 1, 0, len(good) - 1)
        hi = np.clip(idx, 0, len(good) - 1)
        pick = np.where(np.abs(good[lo] - bad) <= np.abs(good[hi] - bad), good[lo], good[hi])
        src_is_ground = fr[pick, x].astype(int).sum(1) > 382     # midpoint of 0..765
        out[bad, x] = np.where(src_is_ground[:, None], 255, 0)
    return out


def read_frames(path):
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = rd.__next__()
    w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd], (w, h)


def write_frames(path, frames, fps):
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 keeps 600x400 exactly; the default pads it to 608x400.
    wr = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264",
                                     pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                     macro_block_size=1, quality=7)
    wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(f))
    wr.close()


def vanish_clip(src, dst, idxs, args, plate=None, valid=None):
    """Write `src` to `dst` with the lander removed on frame indices `idxs`.

    With a plate, erased pixels are replaced by the real background behind the lander. The
    column fill is only used for pixels the plate never saw unobstructed, which on a full
    episode is normally none at all.
    """
    frames, _ = read_frames(src)
    out, hidden, fallback = [], 0, 0
    for i, fr in enumerate(frames):
        if i in idxs:
            mk = lander_mask(fr, args.hull_tol, args.plume_radius, args.dilate)
            if mk.any():
                if plate is not None:
                    fr, leftover = inpaint_from_plate(fr, mk, plate, valid)
                    if leftover.any():
                        fr = inpaint_columns(fr, leftover)
                        fallback += int(leftover.sum())
                else:
                    fr = inpaint_columns(fr, mk)
                hidden += 1
        out.append(fr)
    write_frames(dst, out, args.fps)
    return len(frames), hidden, fallback


def get_plate(meta, key, q, episodes, cache, args, seg):
    """Background plate for one segment, taken from its full episode (cached per episode).

    Falls back to a plate built from the clip itself if the episode video can't be read, and
    the caller falls back again to the column fill for any pixel neither could see.
    """
    if episodes is None:
        frames, _ = read_frames(seg)
        return build_plate(frames, args.plate_frames, args.hull_tol, args.plume_radius, args.dilate)

    start = int(meta[key][q])
    ep = find_episode(episodes, start, args.query_len)
    if ep is None:
        frames, _ = read_frames(seg)
        return build_plate(frames, args.plate_frames, args.hull_tol, args.plume_radius, args.dilate)
    eid = ep["episode_idx"]
    if eid not in cache:
        try:
            frames, _ = read_frames(ep["mp4_path"])
        except Exception as e:                      # episode video moved/corrupt
            print(f"  WARN episode {eid} unreadable ({e}); using clip-only plate")
            frames, _ = read_frames(seg)
        cache[eid] = build_plate(frames, args.plate_frames, args.hull_tol,
                                 args.plume_radius, args.dilate)
    return cache[eid]


def main():
    args = parse_args()

    if args.in_mp4:                      # ---- single-clip mode ----
        if not args.out_mp4:
            raise SystemExit("--in_mp4 requires --out_mp4")
        frames, _ = read_frames(args.in_mp4)
        rng = np.random.default_rng(args.blank_seed)
        idxs = set(blanked_frames(len(frames), args.blank_prob, rng, args.block_len))
        # No episode context in single-clip mode, so the plate can only come from this clip.
        plate, valid = build_plate(frames, args.plate_frames, args.hull_tol,
                                   args.plume_radius, args.dilate)
        n, hidden, fb = vanish_clip(args.in_mp4, args.out_mp4, idxs, args, plate, valid)
        print(f"[vanish] {args.out_mp4}")
        print(f"  plate from this clip only: {100*valid.mean():.1f}% of pixels reconstructed, "
              f"{fb} px fell back to the column fill")
        print(f"  {n} frames, {len(idxs)} in blanked blocks ({len(idxs)/n:.1%}), "
              f"lander removed on {hidden} of them "
              f"({len(idxs)-hidden} had no lander on screen)")
        print(f"  blanked frame indices: {sorted(idxs)}")
        return

    if not args.videos_dir:              # ---- set mode ----
        raise SystemExit("pass either --in_mp4/--out_mp4 or --videos_dir")
    batch = args.videos_dir / f"batch_{args.batch_idx:03d}"

    # --- episode context: the background plate is built from the WHOLE episode, not the
    # 100-frame clip. Measured, that matters: on pair_007 the lander had already landed
    # before the clip starts, so 1,152 px are covered in all 100 segment frames and can
    # never be reconstructed from the clip alone. Over the full 911-frame episode: 0. ---
    meta = pickle.load(open(batch / "metadata.pkl", "rb"))
    episodes = plate_cache = None
    if not args.no_plate:
        ep_index = Path(meta["rollout_dir"]) / "episodes" / "index.pkl"
        if ep_index.exists():
            episodes = pickle.load(open(ep_index, "rb"))
            plate_cache = {}     # episode_idx -> (plate, valid); pairs often share an episode
            print(f"[vanish] episode plates enabled ({len(episodes)} episodes indexed)")
        else:
            print(f"[vanish] WARNING: {ep_index} missing — falling back to per-clip plates")

    tot_blank = tot_frames = tot_hidden = tot_fallback = 0
    for q in range(args.num_query):
        for side, tag in ((0, "A"), (1, "B")):
            seg = batch / f"pair_{q:03d}" / f"seg_{tag}.mp4"
            if not seg.exists():
                print(f"  MISSING {seg}"); continue
            # identical seed derivation to blank_segment_videos.py -> identical blocks
            seed = (args.blank_seed * 100003 + args.batch_idx * 1009 + q * 7 + side) % (2 ** 32)
            rng = np.random.default_rng(seed)
            idxs = set(blanked_frames(args.query_len, args.blank_prob, rng, args.block_len))

            plate = valid = None
            if not args.no_plate:
                key = "indices_1" if side == 0 else "indices_2"
                plate, valid = get_plate(meta, key, q, episodes, plate_cache, args, seg)

            tmp = seg.with_suffix(".vanish.mp4")
            n, hidden, fb = vanish_clip(seg, tmp, idxs, args, plate, valid)
            tmp.replace(seg)
            tot_blank += len(idxs); tot_frames += n; tot_hidden += hidden; tot_fallback += fb
        if plate_cache is not None and q % 20 == 0:
            print(f"  ... pair {q}/{args.num_query} ({len(plate_cache)} plates cached)")
    # CRITICAL (same as blank_segment_videos.py): repoint index.pkl at the LOCAL batch, else
    # label_web follows stale absolute paths back to the clean videos and shows no masking.
    local = sorted(str(d.resolve()) for d in args.videos_dir.glob("batch_*") if d.is_dir())
    with open(args.videos_dir / "index.pkl", "wb") as g:
        pickle.dump(local, g)
    print(f"[vanish] {args.num_query} pairs, block b={args.block_len} p={args.blank_prob} "
          f"-> {tot_blank}/{tot_frames} frames in blanked blocks "
          f"({tot_blank/max(1,tot_frames):.1%}), lander removed on {tot_hidden}; "
          f"index.pkl repointed")
    print(f"[vanish] background reconstructed from episode frames; "
          f"{tot_fallback} px needed the column fallback "
          f"({'none — the plate covered everything' if tot_fallback == 0 else 'see above'})")


if __name__ == "__main__":
    main()
