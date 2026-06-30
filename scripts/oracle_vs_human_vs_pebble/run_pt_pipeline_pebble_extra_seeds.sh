#!/bin/bash
#SBATCH --job-name=pt-pipeline-pebble-extra
#SBATCH --account=aip-mtaylor3
#SBATCH --array=1-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run C extra seeds (1..4). Each task reuses the same 100 PEBBLE labels
# and the synthetic reward-training HDF5 from seed_0; only the PT/IQL
# RNG varies. IQL dataset = seed_0/lunarlander-medium-v2.hdf5 (same 1M
# offline data as seed 0's run, matching PT-paper "labels fixed, algo
# RNG varies" protocol).

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-1}}
ENV_TAG=lunarlander-pebble100-s${SEED}
SYNTHETIC=$SCRATCH/PT/lunarlander/pebble_labels/lunarlander-pebble100-s0.hdf5
IQL_DATASET=$SCRATCH/PT/lunarlander/seed_0/lunarlander-medium-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/pebble100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/pebble100/seed_${SEED}

for f in "$SYNTHETIC" "$IQL_DATASET"; do
    [[ -f "$f" ]] || { echo "ERROR: $f missing." 1>&2; exit 1; }
done
[[ -f /home/marzii/PT/PreferenceTransformer/human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing (need symlink to s0 dir)." 1>&2; exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG (labels = seed_0's PEBBLE 100, IQL on seed_0/medium-v2) ==="

echo "[stage 1/2] PrefTransformer reward training (2000 epochs, PEBBLE labels, 100 queries, query_len=50)"
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" \
    --dataset_path="$SYNTHETIC" \
    --model_type=PrefTransformer \
    --use_human_label=True \
    --num_query=100 \
    --query_len=50 \
    --n_epochs=2000 \
    --eval_period=10 \
    --batch_size=64 \
    --seed="$SEED" \
    --data_seed="$SEED" \
    --comment=pebble100 \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo ""
echo "[stage 2/2] IQL on PT-relabelled seed_0/medium-v2 1M dataset (1M steps)"
python train_offline.py \
    --env_name="$ENV_TAG" \
    --dataset_path="$IQL_DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=True \
    --model_type=PrefTransformer \
    --ckpt_dir="$CKPT_DIR" \
    --seq_len=50 \
    --max_steps=1000000 \
    --eval_interval=5000 \
    --eval_episodes=10 \
    --log_interval=1000 \
    --tqdm=False \
    --save_dir="$IQL_LOG_DIR" \
    --seed="$SEED" \
    --comment=pebble100

echo ""
echo "Run C seed=$SEED done"
