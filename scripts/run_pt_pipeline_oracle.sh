#!/bin/bash
#SBATCH --job-name=pt-pipeline-oracle
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run A: PT-with-oracle on LunarLander-medium-v2.
# 3-seed array (SLURM_ARRAY_TASK_ID -> seed in {0,1,2}). Each task does:
#
#   stage 1: train PrefTransformer reward model
#            (auto-sample 100 query pairs, scripted teacher,
#             paper defaults: n_epochs=2000, query_len=100, batch_size=64,
#             transformer.embd_dim=256 / n_layer=1 / n_head=4)
#
#   stage 2: train_offline.py with --use_reward_model=True
#            (relabel the 1M-transition dataset with the PT reward,
#             run IQL for 1M steps, eval 10 episodes every 5K steps)
#
# Wall: ~15 min stage 1 + ~1.5 min relabel + ~40-50 min IQL ≈ ~60-70 min per seed.

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
ENV_TAG=lunarlander-medium-v2-oracle-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_${SEED}/lunarlander-medium-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/oracle100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/oracle100/seed_${SEED}

if [ ! -f "$DATASET" ]; then
    echo "ERROR: $DATASET missing — Phase A must have produced this." 1>&2
    exit 1
fi

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG ==="
echo "DATASET   = $DATASET"
echo "CKPT_DIR  = $CKPT_DIR"
echo "IQL_LOGS  = $IQL_LOG_DIR"
echo ""

echo "[stage 1/2] PrefTransformer reward training (2000 epochs, oracle, 100 queries)"
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=False \
    --num_query=100 \
    --query_len=100 \
    --n_epochs=2000 \
    --eval_period=10 \
    --batch_size=64 \
    --seed="$SEED" \
    --data_seed="$SEED" \
    --comment=oracle100 \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo ""
echo "[stage 2/2] IQL on PT-relabelled dataset (1M steps)"
python train_offline.py \
    --env_name="$ENV_TAG" \
    --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=True \
    --model_type=PrefTransformer \
    --ckpt_dir="$CKPT_DIR" \
    --max_steps=1000000 \
    --eval_interval=5000 \
    --eval_episodes=10 \
    --log_interval=1000 \
    --tqdm=False \
    --save_dir="$IQL_LOG_DIR" \
    --seed="$SEED" \
    --comment=oracle100

echo ""
echo "Run A seed=$SEED done"
