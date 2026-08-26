#!/bin/bash
#SBATCH --job-name=pt-pipeline-human-mixture
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pipeline/%x_%A_%a.out
#SBATCH --error=logs/pipeline/%x_%A_%a.err

set -euo pipefail
mkdir -p logs/pipeline
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
ENV_TAG=lunarlander-mixture-v2-human-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG%%-*}/${ENV_TAG}/PrefTransformer/human100/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/iql_runs/human100_mixture/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f /home/marzii/PT/PreferenceTransformer/human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing." 1>&2; exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG  (mixture, human 100, data_seed=42) ==="

python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" --dataset_path="$DATASET" --model_type=PrefTransformer \
    --use_human_label=True --num_query=100 --query_len=100 \
    --n_epochs=2000 --eval_period=10 --batch_size=64 \
    --seed="$SEED" --data_seed=42 \
    --comment=human100 --logging.online=False \
    --transformer.embd_dim=256 --transformer.n_layer=1 --transformer.n_head=4

python train_offline.py \
    --env_name="$ENV_TAG" --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=True --model_type=PrefTransformer --ckpt_dir="$CKPT_DIR" \
    --max_steps=1000000 --eval_interval=5000 --eval_episodes=10 \
    --log_interval=1000 --tqdm=False --save_dir="$IQL_LOG_DIR" \
    --seed="$SEED" --comment=human100

echo "Run B-mixture seed=$SEED done"
