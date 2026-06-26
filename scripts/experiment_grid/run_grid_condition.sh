#!/bin/bash
#SBATCH --job-name=pt-grid
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Parameterised PT-grid pipeline run. Submit per condition via --export:
#   sbatch --export=ALL,CONDITION=<env_tag>,NUM_QUERY=<N> \
#       scripts/experiment_grid/run_grid_condition.sh
#
# Each of the 10 array tasks does:
#   stage 1: PT reward model on the condition's pre-saved label file
#            (data_seed=42 fixed, seed varies 0..9).
#   stage 2: IQL on PT-relabelled seed_0/lunarlander-medium-v2.hdf5
#            (1M steps).
#   stage 3: write eval_summary.json with condition metadata + final metrics.

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

: "${CONDITION:?must export CONDITION=<env_tag>}"
: "${NUM_QUERY:?must export NUM_QUERY=<N>}"
SEED=${SLURM_ARRAY_TASK_ID:-0}
ENV_TAG=$CONDITION
# Queries and labels were sampled against the mixture-v2 dataset (1M
# transitions, 232.5K each from random/medium/medium-expert/expert + 70K
# medium-replay). PT-reward AND IQL both use the same dataset.
DATASET=$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5
CKPT_DIR=./reward_model/${ENV_TAG}/PrefTransformer/grid_mixture/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/grid_mixture/${ENV_TAG}/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f /home/marzii/PT/PreferenceTransformer/human_label/${ENV_TAG}/label_human ]] || {
    echo "ERROR: human_label/${ENV_TAG}/label_human missing." 1>&2; exit 1; }

echo "=== CONDITION=$ENV_TAG  NUM_QUERY=$NUM_QUERY  SEED=$SEED  data_seed=42 ==="

echo "[stage 1/3] PT reward training"
python -m JaxPref.new_preference_reward_main \
    --env="$ENV_TAG" \
    --dataset_path="$DATASET" \
    --model_type=PrefTransformer \
    --use_human_label=True \
    --num_query="$NUM_QUERY" \
    --query_len=100 \
    --n_epochs=2000 \
    --eval_period=10 \
    --batch_size=64 \
    --seed="$SEED" \
    --data_seed=42 \
    --comment=grid_mixture \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4

echo ""
echo "[stage 2/3] IQL on PT-relabelled medium-v2 (1M steps)"
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
    --comment=grid_mixture

echo ""
echo "[stage 3/3] writing eval_summary.json"
ENV_TAG="$ENV_TAG" SEED="$SEED" NUM_QUERY="$NUM_QUERY" IQL_LOG_DIR="$IQL_LOG_DIR" \
python - <<'PY'
import json, os, glob
from pathlib import Path
import numpy as np

env_tag = os.environ["ENV_TAG"]
seed = int(os.environ["SEED"])
num_query = int(os.environ["NUM_QUERY"])
iql_log_dir = Path(os.environ["IQL_LOG_DIR"])
meta_path = Path("/home/marzii/PT/PreferenceTransformer/human_label/_grid_metadata") / f"{env_tag}.label_alignment.json"

# Latest progress.txt for this run.
prog_files = sorted(iql_log_dir.glob("**/progress.txt"))
if not prog_files:
    print(f"WARN: no progress.txt under {iql_log_dir}; eval_summary skipped"); raise SystemExit(0)
data = np.loadtxt(prog_files[-1])
if data.ndim == 1: data = data.reshape(1, -1)
last10 = float(data[-10:, 1].mean()) if len(data) >= 10 else float(data[:, 1].mean())
final = float(data[-1, 1])

# Label metadata from setup_grid_labels.py
with open(meta_path) as f:
    meta = json.load(f)

summary = dict(
    condition_id=env_tag,
    num_query=num_query,
    noise_pct=int(meta.get("noise_pct", 0)),
    rm_seed=seed,
    iql_seed=seed,
    label_alignment=float(meta.get("label_alignment", 1.0)),
    last10_eval_reward=last10,
    final_eval_reward=final,
    n_evals=int(len(data)),
)
out = iql_log_dir / "eval_summary.json"
with open(out, "w") as g:
    json.dump(summary, g, indent=2)
print(f"wrote {out}\n  {json.dumps(summary, indent=2)}")
PY

echo "Run $ENV_TAG seed=$SEED done"
