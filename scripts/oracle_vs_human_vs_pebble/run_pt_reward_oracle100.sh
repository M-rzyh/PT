#!/bin/bash
#SBATCH --job-name=pt-reward-oracle100
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/misc/%x_%j.out
#SBATCH --error=logs/misc/%x_%j.err
#
# Phase D first real run: PrefTransformer reward training, oracle labels,
# 100 query pairs, 200 epochs (sanity scope).
#
# Uses the full 1M-step seed_0/medium-v2 HDF5 (Phase A output). The script
# auto-samples 100 random pairs and labels them by ground-truth reward sum
# (--use_human_label=False).
#
# Output: ./reward_model/lunarlander-medium-v2-s0-oracle100/PrefTransformer/oracle100/s42/{best_model,model}.pkl

set -euo pipefail
mkdir -p logs/misc

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

# Use a distinct env-suffix so sampled indices land in their own dir and
# do NOT clobber the existing human-labelled human_label/lunarlander-medium-v2-s0/.
ENV_TAG=lunarlander-medium-v2-s0-oracle100
DATASET=$SCRATCH/PT/lunarlander/seed_0/lunarlander-medium-v2.hdf5

python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=False \
    --num_query=100 \
    --query_len=100 \
    --n_epochs=200 \
    --eval_period=10 \
    --batch_size=64 \
    --comment=oracle100 \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo "phase D oracle100 done"
