#!/bin/bash
#SBATCH --job-name=pt-delay
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/delay/%x_%A_%a.out
#SBATCH --error=logs/delay/%x_%A_%a.err
#
# PT HUMAN arm of the ACTION-DELAY study. Reward model + IQL.
#
# CRUCIAL DIFFERENCE from vanish/speed: those manipulate the human's VIEW of clean trajectories,
# so their IQL dataset is the clean mixture. DELAY manipulates the TRAJECTORIES themselves, so
# each level's IQL dataset is that level's DELAYED 4-source mixture (mixture-delay{K}), and the
# 100 labelled pairs come from that same delayed mixture. Every delay level therefore has its
# OWN pairs (unlike vanish/speed, which share one set of 100).
#
#   K=0  sbatch scripts/action_delay/run_pt_pipeline_delay.sh
#   K=5  sbatch scripts/action_delay/run_pt_pipeline_delay.sh
#   K=10 sbatch scripts/action_delay/run_pt_pipeline_delay.sh
#   K=20 sbatch scripts/action_delay/run_pt_pipeline_delay.sh
set -euo pipefail
mkdir -p logs/delay
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

K=${K:?set K=0|5|10|20 (delay in sim steps)}
case "$K" in 0|5|10|20) ;; *) echo "ERROR: K must be 0, 5, 10 or 20 (got '$K')" 1>&2; exit 1 ;; esac

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NQ=100
TAG=humandelay${K}
ENV_TAG=lunarlander-mixture-v2-${TAG}-s${SEED}
# The DELAYED mixture for this level — NOT the clean one.
DATASET=$SCRATCH/PT/lunarlander/action_delay/mixture-delay${K}/lunarlander-mixture-v2.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/${TAG}/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/action_delay/${TAG}_mixture/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: delayed mixture $DATASET missing (build it first)." 1>&2; exit 1; }
[[ -f human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing — label delay K=${K} first." 1>&2; exit 1; }
[[ -f human_label/${ENV_TAG}/indices_num${NQ}_q100 ]] || {
    echo "ERROR: human_label/${ENV_TAG}/indices_num${NQ}_q100 missing — labels not at N=${NQ}." 1>&2; exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG  (PT action-delay K=${K}, N=${NQ}, dataset=delayed mixture) ==="
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
