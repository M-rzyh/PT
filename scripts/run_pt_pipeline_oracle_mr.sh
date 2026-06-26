#!/bin/bash
#SBATCH --job-name=pt-pipeline-oracle-mr
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run A on the medium-REPLAY variant. Mirrors run_pt_pipeline_oracle.sh
# but swaps the dataset to seed_${SEED}/lunarlander-medium-replay-v2.hdf5.
# PT reward + IQL both use the same medium-replay dataset.

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
ENV_TAG=lunarlander-medium-replay-v2-oracle-s${SEED}
# Single rendered medium-replay HDF5, shared across all 5 PT training seeds
# (mirrors how Run B on medium used seed_0/render/medium-v2 across its 5 seeds).
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/medium-replay-v2/lunarlander-medium-replay-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/oracle100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/oracle100_mr/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG  (medium-replay) ==="

echo "[stage 1/2] PrefTransformer reward training (2000 ep, oracle, 100 queries)"
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=False \
    --num_query=100 \
    --query_len=50 \
    --n_epochs=2000 \
    --eval_period=10 \
    --batch_size=64 \
    --seed="$SEED" \
    --data_seed=42 \
    --comment=oracle100 \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo ""
echo "[stage 2/2] IQL on PT-relabelled medium-replay (1M steps)"
python train_offline.py \
    --env_name="$ENV_TAG" \
    --dataset_path="$DATASET" \
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
    --comment=oracle100
