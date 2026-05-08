#!/bin/bash
#SBATCH --job-name=pt-rerender-medium
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Re-roll seed_{0,1,2}/medium-v2 at 200K steps with rendering enabled,
# so Run B (human labels) has simulator-faithful videos available for
# the query segments.
#
# 200K steps ≈ ~800 episodes/seed — plenty for 100 query pairs, even
# with the eligibility filter that segments fit within one episode.
# Earlier 1M-step attempt (job 4895570) timed out at 2h because frame
# capture + ffmpeg encode is meaningfully slower than a non-render
# rollout; 200K finishes comfortably in ~30 min.
#
# Output per seed:
#   $SCRATCH/PT/lunarlander/seed_${SEED}/render/medium-v2/
#       lunarlander-medium-v2.hdf5     # 200K transitions, lockstep with mp4s
#       episodes/episode_NNNNN.mp4
#       episodes/index.pkl

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
ACTOR=$SCRATCH/PT/lunarlander/seed_${SEED}/sac_run/medium/actor.zip
OUT=$SCRATCH/PT/lunarlander/seed_${SEED}/render/medium-v2

if [ ! -f "$ACTOR" ]; then
    echo "ERROR: $ACTOR missing — Phase A must have produced this." 1>&2
    exit 1
fi

python -m scripts.offline_data.rollout_with_video \
    --actor "$ACTOR" \
    --output_dir "$OUT" \
    --num_steps 200000 \
    --fps 20 \
    --seed "$SEED" \
    --variant medium

echo "rerender seed=$SEED done → $OUT"
