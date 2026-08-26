#!/bin/bash
#SBATCH --job-name=pt-grid-ms
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
# WHAT: Step 3 of 3. The actual work for ONE (noise level, seed): PT reward
#       model -> IQL (1M steps) -> eval_summary.json. Launched by submit_ms_noise.sh.
# Multi-seed noise grid run. Each array task (seed) sees a DIFFERENT set of
# corrupted preference pairs — unlike run_grid_condition.sh where all seeds
# share the same label file.
#
# Submit per noise level:
#   sbatch --export=ALL,NOISE_PCT=<pct> \
#       scripts/experiment_grid/run_grid_ms.sh
#
# NOISE_PCT: integer 0-100 (0 = clean / no corruption).

set -euo pipefail
mkdir -p logs/grid_ms

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

: "${NOISE_PCT:?must export NOISE_PCT=<integer 0-100>}"
# NOISE_MODE selects the corruption model. Must match the mode the labels were
# generated with in setup_grid_labels_ms.py:
#   random_replace   (default): tags 'noise{P}' / 'clean'
#   deterministic_flip        : tags 'flipnoise{P}' / 'flipclean'
NOISE_MODE=${NOISE_MODE:-random_replace}
SEED=${SLURM_ARRAY_TASK_ID:-0}
# Label budget. Default 1000 (existing noise grid). Export NUM_QUERY=50 for the
# low-budget / matched-human-time sweep. Labels must exist for this N.
NUM_QUERY=${NUM_QUERY:-1000}

NOISE_SELECTION=${NOISE_SELECTION:-exact}   # exact (exnoise*) or coin (noise*)
if [ "$NOISE_MODE" = "deterministic_flip" ]; then
    if [ "$NOISE_SELECTION" = "coin" ]; then NOISE_WORD="flipnoise"; CLEAN_WORD="flipclean"
    else NOISE_WORD="exflipnoise"; CLEAN_WORD="exflipclean"; fi
else
    if [ "$NOISE_SELECTION" = "coin" ]; then NOISE_WORD="noise"; CLEAN_WORD="clean"
    else NOISE_WORD="exnoise"; CLEAN_WORD="exclean"; fi
fi

# Seed-specific label dir (unique noise draw per seed).
if [ "$NOISE_PCT" -eq 0 ]; then
    LABEL_TAG="lunarlander-grid-ms-N${NUM_QUERY}-${CLEAN_WORD}-s${SEED}"
else
    LABEL_TAG="lunarlander-grid-ms-N${NUM_QUERY}-${NOISE_WORD}${NOISE_PCT}-s${SEED}"
fi

# Condition ID used for IQL output dir and eval_summary (no seed suffix so
# all seeds for the same noise level land under the same condition directory).
if [ "$NOISE_PCT" -eq 0 ]; then
    COND_ID="lunarlander-grid-ms-N${NUM_QUERY}-${CLEAN_WORD}"
else
    COND_ID="lunarlander-grid-ms-N${NUM_QUERY}-${NOISE_WORD}${NOISE_PCT}"
fi

DATASET=$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
COND_SUFFIX=${COND_SUFFIX:-}
SKIP_REWARD=${SKIP_REWARD:-0}
COND_ID="${COND_ID}${COND_SUFFIX}"
CKPT_DIR=./reward_model/${LABEL_TAG%%-*}/${LABEL_TAG}/PrefTransformer/grid_ms/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/grid_mixture_ms/${COND_ID}/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f "human_label/${LABEL_TAG}/label_human" ]] || {
    echo "ERROR: human_label/${LABEL_TAG}/label_human missing." 1>&2; exit 1; }

echo "=== NOISE_PCT=$NOISE_PCT  SEED=$SEED  LABEL_TAG=$LABEL_TAG ==="
# this here "--use_human_label=True" means loading the labels we have done before, not that the reward model is trained with human labels (it is trained with the noisy labels, but the --use_human_label flag just controls loading the label file for generating the training pairs, not whether those labels are noisy or clean).
echo "[stage 1/3] PT reward training"

if [[ "$SKIP_REWARD" == "1" ]]; then
  echo "  SKIP_REWARD=1 -> reusing existing reward model at $CKPT_DIR"
else
python -m JaxPref.new_preference_reward_main \
    --env="$LABEL_TAG" \
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
    --comment=grid_ms \
    --logging.online=False \
    --transformer.embd_dim=256 \
    --transformer.n_layer=1 \
    --transformer.n_head=4
fi

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
    --eval_interval="$EVAL_INTERVAL" \
    --eval_episodes=10 \
    --log_interval=1000 \
    --tqdm=False \
    --save_dir="$IQL_LOG_DIR" \
    --seed="$SEED" \
    --comment=grid_ms

echo ""
echo "[stage 3/3] writing eval_summary.json"
LABEL_TAG="$LABEL_TAG" COND_ID="$COND_ID" SEED="$SEED" \
NUM_QUERY="$NUM_QUERY" IQL_LOG_DIR="$IQL_LOG_DIR" NOISE_PCT="$NOISE_PCT" \
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np

label_tag  = os.environ["LABEL_TAG"]
cond_id    = os.environ["COND_ID"]
seed       = int(os.environ["SEED"])
num_query  = int(os.environ["NUM_QUERY"])
noise_pct  = int(os.environ["NOISE_PCT"])
iql_log_dir = Path(os.environ["IQL_LOG_DIR"])

meta_path = Path("human_label/_grid_metadata") / f"{label_tag}.label_alignment.json"

prog_files = sorted(iql_log_dir.glob("**/progress.txt"))
if not prog_files:
    print(f"WARN: no progress.txt under {iql_log_dir}; eval_summary skipped")
    raise SystemExit(0)
data = np.loadtxt(prog_files[-1])
if data.ndim == 1: data = data.reshape(1, -1)
last10 = float(data[-10:, 1].mean()) if len(data) >= 10 else float(data[:, 1].mean())
final  = float(data[-1, 1])

with open(meta_path) as f:
    meta = json.load(f)

summary = dict(
    condition_id=cond_id,
    label_tag=label_tag,
    num_query=num_query,
    noise_pct=noise_pct,
    rm_seed=seed,
    iql_seed=seed,
    noise_rng_seed=int(meta.get("noise_rng_seed", -1)),
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

echo "Run $COND_ID seed=$SEED done"
