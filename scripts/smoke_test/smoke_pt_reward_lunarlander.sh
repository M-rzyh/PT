#!/bin/bash
#SBATCH --job-name=pt-reward-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke/%x_%j.out
#SBATCH --error=logs/smoke/%x_%j.err
#
# Phase D smoke: run JaxPref/new_preference_reward_main.py on the 10
# human-labelled pairs for seed_0/medium-v2. Uses PrefTransformer with a
# small config; n_epochs=10. Verifies the loader / label convention /
# upstream code path all wire together before scaling.

set -euo pipefail
mkdir -p logs/smoke

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

DATASET=$SCRATCH/PT/lunarlander/labels/lunarlander-medium-v2/seed_0/rollout/lunarlander-medium-v2.hdf5

python -m JaxPref.new_preference_reward_main \
    --env=lunarlander-medium-v2-s0 \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=True \
    --num_query=10 \
    --query_len=100 \
    --n_epochs=10 \
    --eval_period=2 \
    --batch_size=8 \
    --comment=smoke \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo "phase D smoke done"
