#!/bin/bash
#SBATCH --job-name=pt-humanblock5p50
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/humanblock/%x_%A_%a.out
#SBATCH --error=logs/humanblock/%x_%A_%a.err
#
# PT BLIND-HUMAN 50% at BLOCK b=5 (0.25 s @ 20 fps) -- the b=5 redo of the
# frame-blanking human arm. Reward model + IQL on the rendered mixture using the
# human's preferences over 50%-blanked videos, N=350 pairs. Labels are identical
# across the 10 array tasks (the human labelled once); the array = 10 TRAINING
# seeds -> the mean+-std band. Outputs isolated in frame_blanking/humanblock5p50_mixture.
set -euo pipefail
mkdir -p logs/humanblock
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
ENV_TAG=lunarlander-mixture-v2-humanblock5p50-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG%%-*}/${ENV_TAG}/PrefTransformer/humanblock5p50/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/frame_blanking/humanblock5p50_mixture/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing." 1>&2; exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG  (PT blind-human 50% block b=5, N=350) ==="
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" --dataset_path="$DATASET" --model_type=PrefTransformer \
    --use_human_label=True --num_query=350 --query_len=100 \
    --n_epochs=2000 --eval_period=10 --batch_size=64 \
    --seed="$SEED" --data_seed=42 \
    --comment=humanblock5p50 --logging.online=False \
    --transformer.embd_dim=256 --transformer.n_layer=1 --transformer.n_head=4

python train_offline.py \
    --env_name="$ENV_TAG" --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=True --model_type=PrefTransformer --ckpt_dir="$CKPT_DIR" \
    --max_steps=1000000 --eval_interval=5000 --eval_episodes=10 \
    --log_interval=1000 --tqdm=False --save_dir="$IQL_LOG_DIR" \
    --seed="$SEED" --comment=humanblock5p50
echo "humanblock5p50 seed=$SEED done"
