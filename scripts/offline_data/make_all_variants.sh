#!/bin/bash
#SBATCH --job-name=pt-lunarlander-data
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Phase A orchestrator: produces the 5 LunarLander HDF5 datasets PER SEED.
# Submitted as a 5-seed job array (SLURM_ARRAY_TASK_ID -> SEED in {0..4}).
#
#   $SCRATCH/PT/lunarlander/seed_${SEED}/
#       lunarlander-random-v2.hdf5         random uniform actions, 1M steps
#       lunarlander-medium-v2.hdf5         1M deterministic rollouts of SAC@250K
#       lunarlander-medium-replay-v2.hdf5  the SAC replay buffer at step 250K
#       lunarlander-medium-expert-v2.hdf5  500K medium + 500K expert
#       lunarlander-expert-v2.hdf5         1M deterministic rollouts of SAC@1M
#
# CPU-only: SAC on LunarLander's 8-d state runs ~5K env-steps/sec on a single
# core (smoke test confirmed); GPU would barely help. Wall time per seed is
# ~30 min. The 5 array tasks run in parallel when cluster has capacity.

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
DATA_DIR=${DATA_DIR:-$SCRATCH/PT/lunarlander/seed_${SEED}}
SAC_DIR=${SAC_DIR:-$DATA_DIR/sac_run}
TOTAL_STEPS=${TOTAL_STEPS:-1000000}
MEDIUM_STEP=${MEDIUM_STEP:-250000}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-1000000}

mkdir -p "$DATA_DIR"

echo "=== Phase A · LunarLander dataset gen ==="
echo "DATA_DIR     = $DATA_DIR"
echo "SAC_DIR      = $SAC_DIR"
echo "TOTAL_STEPS  = $TOTAL_STEPS"
echo "MEDIUM_STEP  = $MEDIUM_STEP"
echo "ROLLOUT_STEPS= $ROLLOUT_STEPS"
echo "SEED         = $SEED"
echo ""

echo "[1/5] training SAC ($TOTAL_STEPS steps, snapshot at $MEDIUM_STEP)..."
python -m scripts.offline_data.train_sac_for_offline \
    --save_dir "$SAC_DIR" \
    --total_steps "$TOTAL_STEPS" \
    --medium_step "$MEDIUM_STEP" \
    --seed "$SEED"

echo ""
echo "[2/5] random rollout ($ROLLOUT_STEPS steps)..."
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor random \
    --output "$DATA_DIR/lunarlander-random-v2.hdf5" \
    --num_steps "$ROLLOUT_STEPS" \
    --seed "$SEED"

echo ""
echo "[3/5] medium rollout ($ROLLOUT_STEPS steps, deterministic)..."
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor "$SAC_DIR/medium/actor.zip" \
    --output "$DATA_DIR/lunarlander-medium-v2.hdf5" \
    --num_steps "$ROLLOUT_STEPS" \
    --deterministic \
    --seed "$SEED"

echo ""
echo "[4/5] medium-replay export..."
python -m scripts.offline_data.replay_to_hdf5 \
    --replay "$SAC_DIR/medium/replay.pkl" \
    --actor "$SAC_DIR/medium/actor.zip" \
    --output "$DATA_DIR/lunarlander-medium-replay-v2.hdf5"

echo ""
echo "[5/5] expert rollout ($ROLLOUT_STEPS steps, deterministic)..."
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor "$SAC_DIR/expert/actor.zip" \
    --output "$DATA_DIR/lunarlander-expert-v2.hdf5" \
    --num_steps "$ROLLOUT_STEPS" \
    --deterministic \
    --seed "$SEED"

echo ""
echo "[bonus] medium-expert concat (500K + 500K)..."
HALF=$((ROLLOUT_STEPS / 2))
python -m scripts.offline_data.concat_hdf5 \
    --inputs "$DATA_DIR/lunarlander-medium-v2.hdf5" "$DATA_DIR/lunarlander-expert-v2.hdf5" \
    --num_each "$HALF" \
    --output "$DATA_DIR/lunarlander-medium-expert-v2.hdf5"

echo ""
echo "=== Phase A done ==="
ls -lh "$DATA_DIR"/lunarlander-*-v2.hdf5
