#!/bin/bash
#SBATCH --job-name=iql-truereward
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-9
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grid_ms/%x_%A_%a.out
#SBATCH --error=logs/grid_ms/%x_%A_%a.err
#
# GROUND-TRUTH-REWARD IQL BASELINE for the PT count axis.
#
# Same offline dataset, same IQL config, same training length, same eval
# protocol and same seeds as the PT runs -- but trained on the ORIGINAL
# environment rewards. No preference labels, no reward model, no subsampling.
#
# Two arms (SHIFT controls the only difference):
#   A: SHIFT=0.0 (default) -- standard oracle IQL: normalize, no offset.
#   B: SHIFT=0.5           -- same normalization plus +0.5/step, reproducing the
#                             offset the learned-reward path applies.
#
# Submit:
#   sbatch --array=0-9 scripts/experiment_grid/run_iql_truereward.sh
#   sbatch --array=0-9 --export=ALL,SHIFT=0.5,RUN_TAG=shift05 \
#       scripts/experiment_grid/run_iql_truereward.sh

set -euo pipefail
mkdir -p logs/grid_ms

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

SEED=${SLURM_ARRAY_TASK_ID:-0}
SHIFT=${SHIFT:-0.0}
RUN_TAG=${RUN_TAG:-}
COND_ID="lunarlander-truereward-iql${RUN_TAG:+-$RUN_TAG}"
# The rendered mixture -- the exact dataset the human and scripted count-axis
# runs train on (and the file their preference segments were cut from).
DATASET=${DATASET:-$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/grid_mixture_ms/${COND_ID}/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }

echo "=== $COND_ID  SEED=$SEED  SHIFT=$SHIFT ==="
echo "dataset: $DATASET"
echo "out:     $IQL_LOG_DIR"

python train_offline.py \
    --env_name="$COND_ID" \
    --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py \
    --use_reward_model=False \
    --true_reward_shift="$SHIFT" \
    --max_steps=1000000 \
    --eval_interval=5000 \
    --eval_episodes=10 \
    --log_interval=1000 \
    --tqdm=False \
    --save_dir="$IQL_LOG_DIR" \
    --save_policy=True \
    --seed="$SEED" \
    --comment=truereward

echo ""
echo "writing eval_summary.json"
COND_ID="$COND_ID" SEED="$SEED" IQL_LOG_DIR="$IQL_LOG_DIR" SHIFT="$SHIFT" python - <<'PY'
import json, os
from pathlib import Path
import numpy as np

iql_log_dir = Path(os.environ["IQL_LOG_DIR"])
prog = sorted(iql_log_dir.glob("**/progress.txt"))
if not prog:
    print(f"WARN: no progress.txt under {iql_log_dir}")
    raise SystemExit(0)
data = np.loadtxt(prog[-1])
if data.ndim == 1:
    data = data.reshape(1, -1)
last10 = float(data[-10:, 1].mean()) if len(data) >= 10 else float(data[:, 1].mean())

summary = dict(
    condition_id=os.environ["COND_ID"],
    label_tag="none-true-reward",
    num_query=0,
    noise_pct=0,
    rm_seed=int(os.environ["SEED"]),
    iql_seed=int(os.environ["SEED"]),
    reward_source="ground_truth_env_reward",
    true_reward_shift=float(os.environ["SHIFT"]),
    last10_eval_reward=last10,
    final_eval_reward=float(data[-1, 1]),
    n_evals=int(len(data)),
)
out = iql_log_dir / "eval_summary.json"
with open(out, "w") as g:
    json.dump(summary, g, indent=2)
print(json.dumps(summary, indent=2))
print(f"wrote {out}")
PY

echo "Run $COND_ID seed=$SEED shift=$SHIFT done"
