#!/bin/bash
#SBATCH --job-name=pt-vanish
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# PT HUMAN arm of the LANDER-VANISHING study. Reward model + IQL on the rendered mixture,
# using the human's preferences over videos in which the lander itself was removed (terrain
# left visible) during b=5 blocks.
#
# ONE script for all four levels -- pass the level in:
#     PCT=0  sbatch scripts/lander_vanish/run_pt_pipeline_vanish.sh
#     PCT=25 sbatch scripts/lander_vanish/run_pt_pipeline_vanish.sh
#     PCT=50 sbatch scripts/lander_vanish/run_pt_pipeline_vanish.sh
#     PCT=75 sbatch scripts/lander_vanish/run_pt_pipeline_vanish.sh
#
# Budget is N=100 preferences (the low-budget pairing validated on the noise axis against
# GAIL N=15). Labels are identical across the 10 array tasks -- the human labelled once, so
# the array is 10 TRAINING seeds giving the mean+-std band, matching the convention on both
# arms. Outputs are isolated under lander_vanish/, leaving frame_blanking/ untouched.
#
# PCT=0 is the CLEAN baseline and costs no human time: its labels are the first 100 of the
# existing clean 350 (see scripts/preference/make_vanish0_labels.py), which is why the
# 25/50/75 video sets were built from exactly those same pairs.
set -euo pipefail
mkdir -p logs
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

PCT=${PCT:?set PCT=0|25|50|75}
case "$PCT" in 0|25|50|75) ;; *) echo "ERROR: PCT must be 0, 25, 50 or 75 (got '$PCT')" 1>&2; exit 1 ;; esac

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NQ=100
TAG=humanvanish${PCT}
ENV_TAG=lunarlander-mixture-v2-${TAG}-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/${TAG}/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/lander_vanish/${TAG}_mixture/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
# Fail before burning a GPU hour: a missing label dir means the human hasn't labelled this
# level yet, and the run would otherwise train on nothing.
[[ -f human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing — label level ${PCT}% first." 1>&2
    exit 1; }
[[ -f human_label/${ENV_TAG}/indices_num${NQ}_q100 ]] || {
    echo "ERROR: human_label/${ENV_TAG}/indices_num${NQ}_q100 missing — labels are not at N=${NQ}." 1>&2
    exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG  (PT lander-vanish ${PCT}%, b=5, N=${NQ}) ==="
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" --dataset_path="$DATASET" --model_type=PrefTransformer \
    --use_human_label=True --num_query=${NQ} --query_len=100 \
    --n_epochs=2000 --eval_period=10 --batch_size=64 \
    --seed="$SEED" --data_seed=42 \
    --comment=${TAG} --logging.online=False \
    --transformer.embd_dim=256 --transformer.n_layer=1 --transformer.n_head=4

python train_offline.py \
    --env_name="$ENV_TAG" --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=True --model_type=PrefTransformer --ckpt_dir="$CKPT_DIR" \
    --max_steps=1000000 --eval_interval=5000 --eval_episodes=10 \
    --log_interval=1000 --tqdm=False --save_dir="$IQL_LOG_DIR" \
    --seed="$SEED" --comment=${TAG}
echo "${TAG} seed=$SEED done"
