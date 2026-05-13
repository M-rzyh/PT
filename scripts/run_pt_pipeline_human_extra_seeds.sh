#!/bin/bash
#SBATCH --job-name=pt-pipeline-human-extra
#SBATCH --account=aip-mtaylor3
#SBATCH --array=1-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Run B extra seeds (1..4). Each task reuses the same 100 human labels
# (collected for seed_0) and the same rendered 200K HDF5; only the PT/IQL
# training RNG (--seed, --data_seed) varies. This matches PT-paper
# convention of "labels fixed across seeds, algorithm RNG varies".

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
ENV_TAG=lunarlander-medium-v2-human-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/medium-v2/lunarlander-medium-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/human100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/human100/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f /home/marzii/PT/PreferenceTransformer/human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing (need symlink to s0 dir)." 1>&2; exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG (labels = seed_0's, dataset = seed_0/render/medium-v2) ==="

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
