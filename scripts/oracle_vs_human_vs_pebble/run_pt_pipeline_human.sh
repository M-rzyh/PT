#!/bin/bash
#SBATCH --job-name=pt-pipeline-human
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pipeline/%x_%j.out
#SBATCH --error=logs/pipeline/%x_%j.err
#
# Run B: PT-with-human-labels on LunarLander-medium-v2, single seed (seed 0).
#
#   stage 1: train PrefTransformer reward model on the 100 web-collected
#            human labels (paper defaults: n_epochs=2000, query_len=100,
#            batch_size=64, transformer.embd_dim=256/n_layer=1/n_head=4).
#            --use_human_label=True so the rational/scripted labels are
#            ignored in favour of label_human.
#
#   stage 2: train_offline.py with --use_reward_model=True
#            (PT relabels the 200K-transition rendered dataset, IQL runs
#             1M steps, 10 eval episodes every 5K).
#
# Wall (typical): ~15 min stage 1 + ~30s relabel + ~30-40 min IQL ≈ ~50
# min. 4h SBATCH ceiling = padding for cluster slowness so morning
# results are essentially guaranteed.

set -euo pipefail
mkdir -p logs/pipeline

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
    echo "ERROR: $DATASET missing." 1>&2
    exit 1
fi
if [ ! -f /home/marzii/PT/PreferenceTransformer/human_label/${ENV_TAG}/label_human ]; then
    echo "ERROR: human_label/${ENV_TAG}/label_human missing." 1>&2
    exit 1
fi

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG ==="
echo "DATASET   = $DATASET"
echo "CKPT_DIR  = $CKPT_DIR"
echo "IQL_LOGS  = $IQL_LOG_DIR"
echo ""

echo "[stage 1/2] PrefTransformer reward training (2000 epochs, human labels, 100 queries)"
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
echo "[stage 2/2] IQL on PT-relabelled rendered dataset (1M steps)"
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
