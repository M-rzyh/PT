#!/bin/bash
#SBATCH --job-name=pt-lunarlander-data
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/misc/%x_%A_%a.out
#SBATCH --error=logs/misc/%x_%A_%a.err
#
# Phase A orchestrator: produces the 5 LunarLander HDF5 datasets PER SEED.
# Submitted as a 5-seed job array (SLURM_ARRAY_TASK_ID -> SEED in {0..4}).
#
#   $SCRATCH/PT/lunarlander/seed_${SEED}/
#       lunarlander-random-v2.hdf5         random uniform actions, 1M steps
#       lunarlander-medium-v2.hdf5         1M deterministic rollouts of SAC@70K
#       lunarlander-medium-replay-v2.hdf5  the SAC replay buffer at step 70K
#       lunarlander-medium-expert-v2.hdf5  500K medium + 500K expert
#       lunarlander-expert-v2.hdf5         1M deterministic rollouts of SAC@500K
#
# CPU-only. With gradient updates kicked in (after learning_starts=10K), SAC
# runs ~46 env-steps/sec on a single core. 500K training ≈ 3 h, plus ~10 min
# of rollouts and exports — well under the 4 h SBATCH budget. The 5 array
# tasks run in parallel when cluster has capacity.
#
# MEDIUM_STEP=70000: seed-0's reward curve from the previous (timed-out) run
# crosses +120 at step 70K — squarely in D4RL's "medium" band of [+80, +130].
# (The earlier 250K choice landed at ~+269, essentially expert.)
# TOTAL_STEPS=500000: seed-0 was at +269 by step 250K; 500K gives expert with
# headroom. D4RL paper used 1M end-to-end, but the practical "expert" recipe
# is "the snapshot at the end of training", which is what we keep.

set -euo pipefail

mkdir -p logs/misc

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
TOTAL_STEPS=${TOTAL_STEPS:-500000}
MEDIUM_STEP=${MEDIUM_STEP:-70000}
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

if [ -f "$SAC_DIR/medium/actor.zip" ] && [ -f "$SAC_DIR/expert/actor.zip" ]; then
    echo "[1/5] skipping SAC training — $SAC_DIR/{medium,expert}/actor.zip already exist."
else
    echo "[1/5] training SAC ($TOTAL_STEPS steps, snapshot at $MEDIUM_STEP)..."
    python -m scripts.offline_data.train_sac_for_offline \
        --save_dir "$SAC_DIR" \
        --total_steps "$TOTAL_STEPS" \
        --medium_step "$MEDIUM_STEP" \
        --seed "$SEED"
fi

echo ""
echo "[2/5] random rollout ($ROLLOUT_STEPS steps)..."
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor random \
    --output "$DATA_DIR/lunarlander-random-v2.hdf5" \
    --num_steps "$ROLLOUT_STEPS" \
    --seed "$SEED"

echo ""
echo "[3/5] medium rollout ($ROLLOUT_STEPS steps, stochastic)..."
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor "$SAC_DIR/medium/actor.zip" \
    --output "$DATA_DIR/lunarlander-medium-v2.hdf5" \
    --num_steps "$ROLLOUT_STEPS" \
    --seed "$SEED"

echo ""
echo "[4/5] medium-replay export..."
python -m scripts.offline_data.replay_to_hdf5 \
    --replay "$SAC_DIR/medium/replay.pkl" \
    --actor "$SAC_DIR/medium/actor.zip" \
    --output "$DATA_DIR/lunarlander-medium-replay-v2.hdf5"

echo ""
echo "[5/5] expert rollout ($ROLLOUT_STEPS steps, stochastic)..."
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor "$SAC_DIR/expert/actor.zip" \
    --output "$DATA_DIR/lunarlander-expert-v2.hdf5" \
    --num_steps "$ROLLOUT_STEPS" \
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
