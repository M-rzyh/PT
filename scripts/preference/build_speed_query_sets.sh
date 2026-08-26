#!/bin/bash
#SBATCH --job-name=pt-build-speed
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/marzii/PT/PreferenceTransformer/logs/%x_%j.out
#SBATCH --error=/home/marzii/PT/PreferenceTransformer/logs/%x_%j.err
#
# Build the PT SPEED-axis preference video sets: the SAME 100 clean pairs, re-timed to a
# faster/slower playback rate. Speed is pure playback (frames are 1-per-step), so this is a
# lossless re-encode via extract_segment_videos.py --out_fps — no re-rolling, no new frames.
#
#   10 fps = 0.2x real time (easy) ; 50 fps = 1.0x real time (hard).
#   20 fps (baseline) is NOT built here — it is the existing clean set, and the SPEED 20fps
#   labels reuse the vanish-0 labels for free.
#
# Uses the SAME query indices as the clean 350 (queries_n350, first 100), so the 10/20/50 sets
# are the identical trajectories at three speeds — exactly the vanish pattern.
#
# Light job (re-encode ~400 clips). Submit with a short walltime:
#   sbatch --time=00:30:00 scripts/preference/build_speed_query_sets.sh
set -euo pipefail
mkdir -p /home/marzii/PT/PreferenceTransformer/logs
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

ROLLOUT=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2          # clean source episodes + hdf5
SRC_Q=$SCRATCH/PT/lunarlander/frame_blanking/pt_human/queries_n350 # the clean 350 indices
DEST_ROOT=$SCRATCH/PT/lunarlander/sim_speed/pt_human
NQ=100; QLEN=100

# extract_segment_videos builds its filename from --num_query, so it wants indices_num100_q100,
# but the clean set only has indices_num350_q100. Write a 100-pair SUBSET (the FIRST 100 -> the
# same pairs vanish 0% used) into a local queries dir so all speed levels share those pairs.
QDIR=$DEST_ROOT/queries_n100_clean
mkdir -p "$QDIR"
python - "$SRC_Q" "$QDIR" "$NQ" "$QLEN" <<'PY'
import pickle, sys
from pathlib import Path
src, dst, n, L = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for name in (f"indices_num350_q{L}", f"indices_2_num350_q{L}"):
    arr = pickle.load(open(src / name, "rb"))[:n]
    out = name.replace(f"num350_q{L}", f"num{n}_q{L}")
    pickle.dump(arr, open(dst / out, "wb"))
    print(f"  wrote {out} ({len(arr)} entries)")
PY

for FPS in 10 50; do
  DEST="$DEST_ROOT/videos_human_n100_speed${FPS}"
  # skip only if actually BUILT (last pair present), not just if the dir was created
  if [[ -f "$DEST/batch_000/pair_$(printf '%03d' $((NQ-1)))/seg_A.mp4" ]]; then
    echo "[speed] $DEST already built — skipping"; continue
  fi
  echo "=============================================================="
  echo "[speed] re-timing the first $NQ clean pairs to ${FPS} fps -> $DEST"
  python scripts/preference/extract_segment_videos.py \
      --rollout_dir "$ROLLOUT" --query_dir "$QDIR" --output_dir "$DEST" \
      --num_query "$NQ" --query_len "$QLEN" --batch_idx 0 --out_fps "$FPS"
done
echo "=============================================================="
echo "[speed] done. 20 fps baseline reuses the existing clean set + vanish-0 labels."
