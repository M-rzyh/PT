#!/bin/bash
#SBATCH --job-name=pt-pipeline-human
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Run B: PT-with-human-labels on LunarLander-medium-v2, seed 0.
#
# Mirrors run_pt_pipeline_oracle.sh but with --use_human_label=True. Reads
# the 100 human labels from human_label/lunarlander-medium-v2-human-s0/
# (produced by labels_web_to_pt_format.py).
#
# Stage 1: PrefTransformer reward training (2000 epochs, 100 human queries)
# Stage 2: train_offline.py with --use_reward_model=True (1M IQL steps)
#
# Wall: ~15 min stage 1 + ~1.5 min relabel + ~30-50 min IQL ≈ ~50-70 min.

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

SEED=${SEED:-0}
ENV_TAG=lunarlander-medium-v2-human-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_${SEED}/render/medium-v2/lunarlander-medium-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/human100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/human100/seed_${SEED}

if [ ! -f "$DATASET" ]; then
    echo "ERROR: $DATASET missing — run rerender_medium_3seeds.sh first." 1>&2
    exit 1
fi

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG ==="
echo "DATASET   = $DATASET"
echo "CKPT_DIR  = $CKPT_DIR"
echo ""

echo "[stage 1/2] PrefTransformer reward training (2000 epochs, 100 HUMAN queries)"
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=True \
    --num_query=100 \
    --query_len=100 \
    --n_epochs=2000 \
    --eval_period=10 \
    --batch_size=64 \
    --seed="$SEED" \
    --data_seed="$SEED" \
    --comment=human100 \
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
    --comment=human100

echo ""
echo "Run B seed=$SEED done"
