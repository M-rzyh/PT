#!/bin/bash
# Simulation-speed EXAMPLE CLIPS (smoke test / supervisor gallery).
#
# Takes ONE existing clean preference clip and re-times it to a ladder of speeds so
# a human can see where judging gets easy / hard. Nothing existing is touched;
# everything lands in $SCRATCH/PT/lunarlander/speed_demo/ (isolated sibling dir).
#
# KEY FACT (measured, not assumed): the source clip is 600x400, 20 fps, 100 frames,
# 5.000 s. Each frame is ONE physics step (LunarLander FPS=50, dt=20 ms), so those
# 100 frames are 2.0 s of real lander time. The clip we currently label therefore
# already plays at 0.4x REAL TIME -- 2.5x slow motion. All factors below are stated
# vs REAL TIME, where real time == 50 fps playback.
#
# Two mechanisms, because they are not the same thing:
#   RETIME    - keep all 100 frames, change the fps they are shown at.
#               Lossless. Bounded above by the display: >60 fps buys nothing.
#   SUBSAMPLE - keep every Kth frame, play at 50 fps. The only way past ~1.2x real
#               time. Lossy: the human literally never sees the dropped steps.
set -euo pipefail

SRC=${SRC:-$SCRATCH/PT/lunarlander/frame_blanking/pt_human/videos_human_n350_clean/batch_000/pair_000/seg_A.mp4}
OUT=${OUT:-$SCRATCH/PT/lunarlander/speed_demo}
SRC_FPS=20   # measured with ffprobe

[[ -f "$SRC" ]] || { echo "ERROR: source clip missing: $SRC" 1>&2; exit 1; }
mkdir -p "$OUT"
echo "source: $SRC"
echo "out:    $OUT"
echo

# retime <name> <target_fps>  -- keep every frame, restretch timestamps
retime() {
    local name=$1 fps=$2
    ffmpeg -y -loglevel error -i "$SRC" \
        -vf "setpts=PTS*${SRC_FPS}/${fps}" -r "$fps" \
        -c:v libx264 -pix_fmt yuv420p "$OUT/${name}.mp4"
}

# subsample <name> <K>  -- keep every Kth frame, show at 50 fps (= K x real time)
subsample() {
    local name=$1 k=$2
    ffmpeg -y -loglevel error -i "$SRC" \
        -vf "select='not(mod(n\,${k}))',setpts=N/(50*TB)" -r 50 \
        -c:v libx264 -pix_fmt yuv420p "$OUT/${name}.mp4"
}

echo "--- RETIME (all 100 frames kept) ---"
retime slowest_0.02x_1fps        1     # "slowest possible" demo: no lower bound exists
retime slow_0.10x_5fps           5
retime slow_0.20x_10fps         10     # EASIER than today
retime base_0.40x_20fps         20     # <-- what we label today
retime faster_0.60x_30fps       30
retime realtime_1.00x_50fps     50     # true real time
retime maxlossless_1.20x_60fps  60     # last speed that shows every physics step

echo "--- SUBSAMPLE (frames dropped; the only route past 1.2x) ---"
subsample drop2_2.0x_realtime 2
subsample drop3_3.0x_realtime 3
subsample drop5_5.0x_realtime 5

echo
echo "--- verification (fps / frames / duration) ---"
printf "%-30s %8s %8s %10s\n" CLIP FPS FRAMES DURATION
for f in "$OUT"/*.mp4; do
    read -r rate frames dur < <(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=r_frame_rate,nb_read_frames -show_entries format=duration \
        -of csv=p=0:nk=1 "$f" | tr ',' '\n' | paste -sd' ')
    printf "%-30s %8s %8s %10s\n" "$(basename "$f")" "$rate" "$frames" "$dur"
done
echo
echo "Done. Clips in $OUT"
