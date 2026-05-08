#!/bin/bash
#SBATCH --job-name=pt-iql-relabel-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Phase D smoke: train_offline.py with --use_reward_model=True. Loads the
# already-trained oracle100 PT reward model, relabels every transition in
# seed_0/medium-v2.hdf5, runs IQL for 200 steps. Validates the relabel
# path before launching the 3-seed Run A array.

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

DATASET=$SCRATCH/PT/lunarlander/seed_0/lunarlander-medium-v2.hdf5
PT_CKPT=./reward_model/lunarlander-medium-v2-s0-oracle100/PrefTransformer/oracle100/s42
WORK=${TMPDIR:-/tmp}/pt-iql-relabel-smoke-${SLURM_JOB_ID:-local}
mkdir -p "$WORK"

python train_offline.py \
    --env_name=lunarlander-medium-v2-s0 \
    --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=True \
    --model_type=PrefTransformer \
    --ckpt_dir="$PT_CKPT" \
    --max_steps=200 \
    --eval_interval=100 \
    --eval_episodes=2 \
    --log_interval=50 \
    --tqdm=False \
    --save_dir="$WORK" \
    --seed=42

echo "phase D iql-relabel smoke done"
