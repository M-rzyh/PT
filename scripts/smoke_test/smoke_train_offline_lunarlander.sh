#!/bin/bash
#SBATCH --job-name=pt-iql-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Phase B smoke: produces a 10K-step random LunarLander HDF5 in $TMPDIR, then
# runs `train_offline.py --env_name=lunarlander-random-v2` for 200 IQL steps
# to verify the loader, dispatch, and IQL forward path all wire together.

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

WORK=${TMPDIR:-/tmp}/pt-iql-smoke-${SLURM_JOB_ID:-local}
mkdir -p "$WORK"

echo "=== [1/2] generate 10K-step random HDF5 ==="
python -m scripts.offline_data.rollout_to_hdf5 \
    --actor random \
    --output "$WORK/lunarlander-random-v2.hdf5" \
    --num_steps 10000 \
    --seed 0

echo ""
echo "=== [2/2] run train_offline.py for 200 IQL steps ==="
python train_offline.py \
    --env_name=lunarlander-random-v2 \
    --dataset_path="$WORK/lunarlander-random-v2.hdf5" \
    --config=configs/lunarlander_config.py \
    --max_steps=200 \
    --eval_interval=100 \
    --eval_episodes=2 \
    --log_interval=50 \
    --tqdm=False \
    --save_dir="$WORK/iql_run" \
    --seed=0

echo ""
echo "phase B smoke OK"
ls -lh "$WORK/iql_run" || true
