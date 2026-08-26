#!/bin/bash
#SBATCH --job-name=pt-pipeline-pebble100
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pipeline/%x_%j.out
#SBATCH --error=logs/pipeline/%x_%j.err
#
# Run C: PT-with-PEBBLE-labels on LunarLander.
#
# Reward stage uses the 100 (sa_t_1, sa_t_2, label) tuples from PEBBLE job
# 4895573's pref_buffer:
#   - first 25 = oracle teacher  (scripted in PEBBLE)
#   - next  75 = human teacher   (web labels from BPref3 label_web.py)
# Built by scripts/preference/build_pebble_pt_dataset.py into:
#   $SCRATCH/PT/lunarlander/pebble_labels/lunarlander-pebble100-s0.hdf5
#   human_label/lunarlander-pebble100-s0/{indices_num100_q50,
#                                          indices_2_num100_q50,
#                                          label_human, provenance.json}
#
# query_len=50 matches PEBBLE; everything else stays at paper defaults.
# IQL stage runs on the same lunar-lander medium-v2-s0 1M HDF5 used by Runs
# A/B so the three runs sit in the same eval space for direct comparison.

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
ENV_TAG=lunarlander-pebble100-s${SEED}
SYNTHETIC=$SCRATCH/PT/lunarlander/pebble_labels/${ENV_TAG}.hdf5
IQL_DATASET=$SCRATCH/PT/lunarlander/seed_${SEED}/lunarlander-medium-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG%%-*}/${ENV_TAG}/PrefTransformer/pebble100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/pebble100/seed_${SEED}

for f in "$SYNTHETIC" "$IQL_DATASET"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f missing." 1>&2; exit 1
    fi
done

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG ==="
echo "SYNTHETIC = $SYNTHETIC   (used for PT reward training)"
echo "IQL_DATASET = $IQL_DATASET  (used for IQL relabel + training)"
echo "CKPT_DIR  = $CKPT_DIR"
echo "IQL_LOGS  = $IQL_LOG_DIR"
echo ""

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
echo "[stage 2/2] IQL on PT-relabelled medium-v2 1M dataset (1M steps)"
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
