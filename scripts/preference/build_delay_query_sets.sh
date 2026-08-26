#!/bin/bash
#SBATCH --job-name=pt-build-delay
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/marzii/PT/logs/delay/%x_%j.out
#SBATCH --error=/scratch/marzii/PT/logs/delay/%x_%j.err
#
# NOTE: no #SBATCH --time here on purpose. Walltime is passed at submit so light jobs
# schedule fast and heavy ones get enough runway:
#   K=0 (no delayed rollout, just build+sample+extract):   sbatch --time=00:20:00 ...
#   K=5|10|20 (two 500K-step rollouts + render per level):  sbatch --time=06:00:00 ...
# Slurm favours short jobs, so a light job asking for 6h waits far longer in the queue.
#
# Build ONE PT action-delay preference video set (one delay level K).
#
# Unlike the vanish builder (which copy+masks existing clips), delay changes the TRAJECTORY,
# so this GENERATES fresh data end-to-end:
#   1. roll medium + expert SAC actors under delay K  -> delayed render sources
#      (K=0 reuses the EXISTING clean renders — delay 0 is a proven no-op, so no re-roll)
#   2. build a 4-source mixture (--no-medium-replay), reusing the delay-invariant random render
#   3. sample NUM_QUERY preference pairs from the delayed mixture
#   4. extract the 100-frame seg_A/seg_B clips the labeller consumes
#
# Non-destructive: everything lands under NEW action_delay/ paths; existing renders/mixture are
# only READ. A SUFFIX lets a small validation run write to isolated dirs it can delete.
#
#   K=10 sbatch scripts/preference/build_delay_query_sets.sh                    # production
#   K=10 NUM_STEPS=5000 PER_VARIANT=2000 NUM_QUERY=5 SUFFIX=_val \              # tiny validation
#        sbatch scripts/preference/build_delay_query_sets.sh
set -euo pipefail
mkdir -p /home/marzii/PT/PreferenceTransformer/logs

module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

K=${K:?set K=0|5|10|20 (delay in sim steps)}
NUM_STEPS=${NUM_STEPS:-500000}   # per medium/expert roll (production); need >=1.5*PER_VARIANT
PER_VARIANT=${PER_VARIANT:-250000}   # per source; 4 sources -> ~1M total (no medium-replay)
NUM_QUERY=${NUM_QUERY:-100}
QUERY_LEN=${QUERY_LEN:-100}
SEED=${SEED:-0}
SUFFIX=${SUFFIX:-}               # e.g. _val to isolate a validation run

RENDER=$SCRATCH/PT/lunarlander/seed_0/render
DELAY=$SCRATCH/PT/lunarlander/action_delay
TAG="delay${K}${SUFFIX}"

echo "=============================================================="
echo "[build-delay] K=$K  NUM_STEPS=$NUM_STEPS  PER_VARIANT=$PER_VARIANT  NUM_QUERY=$NUM_QUERY  TAG=$TAG"

# --- 1. delayed render sources (medium + expert). K=0 reuses existing clean renders. ---
if [[ "$K" == "0" ]]; then
  MED_DIR="$RENDER/medium-v2-mixprep"
  EXP_DIR="$RENDER/expert-v2-mixprep"
  echo "[build-delay] K=0 -> reusing existing clean medium/expert renders (delay 0 = no-op)"
else
  MED_DIR="$RENDER/medium-v2-${TAG}-mixprep"
  EXP_DIR="$RENDER/expert-v2-${TAG}-mixprep"
  for pair in "medium:$MED_DIR" "expert:$EXP_DIR"; do
    V="${pair%%:*}"; D="${pair##*:}"
    if [[ -d "$D" ]]; then
      echo "[build-delay] $D EXISTS — skipping roll (delete to redo)"
    else
      echo "[build-delay] rolling $V under delay K=$K -> $D"
      python -m scripts.offline_data.rollout_with_video \
          --actor "$SCRATCH/PT/lunarlander/seed_0/sac_run/${V}/actor.zip" \
          --output_dir "$D" --num_steps "$NUM_STEPS" --fps 20 --seed "$SEED" \
          --variant "$V" --delay-k "$K"
    fi
  done
fi

# --- 2. build the 4-source delayed mixture (random reused; no medium-replay) ---
MIX="$DELAY/mixture-${TAG}"
if [[ -d "$MIX" ]]; then
  echo "[build-delay] $MIX EXISTS — skipping build"
else
  python -m scripts.offline_data.build_rendered_mixture \
      --seed "$SEED" --mix_seed 0 --per_variant "$PER_VARIANT" --no-medium-replay \
      --random-dir "$RENDER/random-v2-mixprep" \
      --medium-dir "$MED_DIR" --expert-dir "$EXP_DIR" \
      --out-dir "$MIX"
fi

# --- 3. sample preference pairs from the delayed mixture ---
QDIR="$DELAY/pt_human/queries_n${NUM_QUERY}_${TAG}"
python scripts/preference/sample_query_indices.py \
    --hdf5 "$MIX/lunarlander-mixture-v2.hdf5" --output_dir "$QDIR" \
    --num_query "$NUM_QUERY" --query_len "$QUERY_LEN" --seed 0

# --- 4. extract the seg_A/seg_B clips for label_web ---
VIDDIR="$DELAY/pt_human/videos_human_n${NUM_QUERY}_${TAG}"
python scripts/preference/extract_segment_videos.py \
    --rollout_dir "$MIX" --query_dir "$QDIR" --output_dir "$VIDDIR" \
    --num_query "$NUM_QUERY" --query_len "$QUERY_LEN" --batch_idx 0

echo "=============================================================="
echo "[build-delay] done. videos -> $VIDDIR"
