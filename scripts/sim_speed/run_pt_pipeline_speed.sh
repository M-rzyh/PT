#!/bin/bash
#SBATCH --job-name=pt-speed
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/speed/%x_%A_%a.out
#SBATCH --error=logs/speed/%x_%A_%a.err
#
# PT HUMAN arm of the SIMULATION-SPEED study. Reward model + IQL on the CLEAN rendered mixture,
# using the human's preferences over the SAME 100 clips played at a faster/slower rate.
#
# Like vanish (and UNLIKE delay), speed only changes the human's VIEW — the trajectories, and
# therefore the IQL DATASET, are the clean mixture, unchanged. Only the labels differ per level.
#
#   FPS=10 sbatch scripts/sim_speed/run_pt_pipeline_speed.sh   # 0.2x real time (easy)
#   FPS=20 sbatch scripts/sim_speed/run_pt_pipeline_speed.sh   # 0.4x baseline (labels reuse vanish-0)
#   FPS=50 sbatch scripts/sim_speed/run_pt_pipeline_speed.sh   # 1.0x real time (hard)
set -euo pipefail
mkdir -p logs/speed
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH; export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

FPS=${FPS:?set FPS=10|20|50}
case "$FPS" in 10|20|50) ;; *) echo "ERROR: FPS must be 10, 20 or 50 (got '$FPS')" 1>&2; exit 1 ;; esac

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
NQ=100
TAG=humanspeed${FPS}
ENV_TAG=lunarlander-mixture-v2-${TAG}-s${SEED}
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5   # CLEAN
CKPT_DIR=./reward_model/${ENV_TAG%%-*}/${ENV_TAG}/PrefTransformer/${TAG}/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/sim_speed/${TAG}_mixture/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing — label ${FPS} fps first "
         "(20 fps reuses the vanish-0 labels)." 1>&2; exit 1; }
[[ -f human_label/${ENV_TAG}/indices_num${NQ}_q100 ]] || {
    echo "ERROR: human_label/${ENV_TAG}/indices_num${NQ}_q100 missing — labels not at N=${NQ}." 1>&2
    exit 1; }

echo "=== seed=$SEED  ENV_TAG=$ENV_TAG  (PT sim-speed ${FPS} fps, N=${NQ}) ==="
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
