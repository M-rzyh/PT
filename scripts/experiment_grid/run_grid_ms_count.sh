#!/bin/bash
#SBATCH --job-name=pt-grid-ms-count
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grid_ms/%x_%A_%a.out
#SBATCH --error=logs/grid_ms/%x_%A_%a.err
#
# Allocated 2 hours to each of the runs, but it can be done in 30 minutes. 
# WHAT: Step 3 of 3. The actual work for ONE (count N, seed): PT reward model
#       -> IQL (1M steps) -> eval_summary.json. Launched by submit_ms_count.sh.
# Multi-seed count-axis run. Each array task (seed) sees a DIFFERENT random
# sample of N preference pairs from the dataset (via per-seed data_seed in
# the label generation step).
#
# Submit per N:
#   sbatch --export=ALL,N_COUNT=<n> \
#       scripts/experiment_grid/run_grid_ms_count.sh

set -euo pipefail
mkdir -p logs/grid_ms

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

: "${N_COUNT:?must export N_COUNT=<integer>}"
SEED=${SLURM_ARRAY_TASK_ID:-0}

# LABEL_PREFIX + DATASET are overridable (defaults = the original oracle count, so existing runs
# are unchanged). For the HUMAN count axis: LABEL_PREFIX=lunarlander-human-count and
# DATASET=<seed_0/render/mixture-v2/...> (the hdf5 the human labels were sampled against).
LABEL_PREFIX=${LABEL_PREFIX:-lunarlander-grid-ms-count}
LABEL_TAG="${LABEL_PREFIX}-N${N_COUNT}-s${SEED}"
COND_ID="${LABEL_PREFIX}-N${N_COUNT}"

DATASET=${DATASET:-$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5}
CKPT_DIR=./reward_model/${LABEL_TAG}/PrefTransformer/grid_ms_count/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/grid_mixture_ms/${COND_ID}/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f "human_label/${LABEL_TAG}/label_human" ]] || {
    echo "ERROR: human_label/${LABEL_TAG}/label_human missing." 1>&2; exit 1; }

echo "=== N_COUNT=$N_COUNT  SEED=$SEED  LABEL_TAG=$LABEL_TAG ==="

echo "[stage 1/3] PT reward training"
python -m JaxPref.new_preference_reward_main \
    --env="$LABEL_TAG" \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=True \
    --num_query="$N_COUNT" \
    --query_len=100 \
    --n_epochs=2000 \
    --eval_period=10 \
    --batch_size=64 \
    --seed="$SEED" \
    --data_seed=42 \
    --comment=grid_ms_count \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo ""
echo "[stage 2/3] IQL on PT-relabelled mixture dataset (1M steps)"
python train_offline.py \
    --env_name="$COND_ID" \
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
    --comment=grid_ms_count

echo ""
echo "[stage 3/3] writing eval_summary.json"
LABEL_TAG="$LABEL_TAG" COND_ID="$COND_ID" SEED="$SEED" \
N_COUNT="$N_COUNT" IQL_LOG_DIR="$IQL_LOG_DIR" \
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np

label_tag   = os.environ["LABEL_TAG"]
cond_id     = os.environ["COND_ID"]
seed        = int(os.environ["SEED"])
n_count     = int(os.environ["N_COUNT"])
iql_log_dir = Path(os.environ["IQL_LOG_DIR"])

meta_path = Path("human_label/_grid_metadata") / f"{label_tag}.label_alignment.json"

prog_files = sorted(iql_log_dir.glob("**/progress.txt"))
if not prog_files:
    print(f"WARN: no progress.txt under {iql_log_dir}"); raise SystemExit(0)
data = np.loadtxt(prog_files[-1])
if data.ndim == 1: data = data.reshape(1, -1)
last10 = float(data[-10:, 1].mean()) if len(data) >= 10 else float(data[:, 1].mean())
final  = float(data[-1, 1])

with open(meta_path) as f:
    meta = json.load(f)

summary = dict(
    condition_id=cond_id,
    label_tag=label_tag,
    num_query=n_count,
    noise_pct=0,
    rm_seed=seed,
    iql_seed=seed,
    actual_data_seed=int(meta.get("actual_data_seed", -1)),
    label_alignment=1.0,
    last10_eval_reward=last10,
    final_eval_reward=final,
    n_evals=int(len(data)),
)
out = iql_log_dir / "eval_summary.json"
with open(out, "w") as g:
    json.dump(summary, g, indent=2)
print(f"wrote {out}")
PY

echo "Run $COND_ID seed=$SEED done"
