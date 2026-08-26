#!/bin/bash
#SBATCH --job-name=pt-blindoracle
#SBATCH --account=aip-mtaylor3
#SBATCH --array=0-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/blindoracle/%x_%A_%a.out
#SBATCH --error=logs/blindoracle/%x_%A_%a.err
#
# WHAT: PT blind-oracle training (frame-blanking, Exp 1). Reward model + IQL on the
#       TRUE 8-D mixture using BLIND-ORACLE labels (degraded by frame-blanking). The
#       agent solves the NORMAL task; only the labels are degraded. Outputs isolated
#       in frame_blanking/. Eval = last10_eval_reward (matches the noise plots).
#   sbatch --array=0-4 --export=ALL,BLANK_PCT=<pct> scripts/experiment_grid/run_grid_ms_blind_oracle.sh
set -euo pipefail
mkdir -p logs/blindoracle
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy
cd /home/marzii/PT/PreferenceTransformer

: "${BLANK_PCT:?must export BLANK_PCT=<integer 0-100>}"
SEED=${SLURM_ARRAY_TASK_ID:-0}
NUM_QUERY=${NUM_QUERY:-350}

if [ "$BLANK_PCT" -eq 0 ]; then
    LABEL_TAG="lunarlander-grid-ms-N${NUM_QUERY}-blindoracleclean-s${SEED}"
    COND_ID="lunarlander-grid-ms-N${NUM_QUERY}-blindoracleclean"
else
    LABEL_TAG="lunarlander-grid-ms-N${NUM_QUERY}-blindoracle${BLANK_PCT}-s${SEED}"
    COND_ID="lunarlander-grid-ms-N${NUM_QUERY}-blindoracle${BLANK_PCT}"
fi

DATASET=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5   # RENDERED 8-D mixture (same data the human labels; normal task)
CKPT_DIR=./reward_model/${LABEL_TAG}/PrefTransformer/frame_blank/s${SEED}
IQL_LOG_DIR=$SCRATCH/PT/lunarlander/frame_blanking/${COND_ID}/seed_${SEED}

[[ -f "$DATASET" ]] || { echo "ERROR: $DATASET missing." 1>&2; exit 1; }
[[ -f "human_label/${LABEL_TAG}/label_human" ]] || {
    echo "ERROR: human_label/${LABEL_TAG}/label_human missing (run gen_ms_blind_oracle.sh)." 1>&2; exit 1; }

echo "=== BLANK_PCT=$BLANK_PCT  SEED=$SEED  LABEL_TAG=$LABEL_TAG ==="
echo "[stage 1/3] PT reward training (blind-oracle labels)"
python -m JaxPref.new_preference_reward_main \
    --env="$LABEL_TAG" --dataset_path="$DATASET" --model_type=PrefTransformer \
    --use_human_label=True --num_query="$NUM_QUERY" --query_len=100 \
    --n_epochs=2000 --eval_period=10 --batch_size=64 \
    --seed="$SEED" --data_seed=42 --comment=frame_blank \
    --logging.online=False --transformer.embd_dim=256 --transformer.n_layer=1 --transformer.n_head=4

echo ""
echo "[stage 2/3] IQL on the TRUE 8-D mixture (1M steps)"
python train_offline.py \
    --env_name="$COND_ID" --dataset_path="$DATASET" \
    --config=configs/lunarlander_config.py --use_reward_model=True \
    --model_type=PrefTransformer --ckpt_dir="$CKPT_DIR" \
    --max_steps=1000000 --eval_interval=5000 --eval_episodes=10 --log_interval=1000 \
    --tqdm=False --save_dir="$IQL_LOG_DIR" --seed="$SEED" --comment=frame_blank

echo ""
echo "[stage 3/3] writing eval_summary.json (last10)"
LABEL_TAG="$LABEL_TAG" COND_ID="$COND_ID" SEED="$SEED" \
NUM_QUERY="$NUM_QUERY" IQL_LOG_DIR="$IQL_LOG_DIR" BLANK_PCT="$BLANK_PCT" \
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np
label_tag=os.environ["LABEL_TAG"]; cond_id=os.environ["COND_ID"]; seed=int(os.environ["SEED"])
num_query=int(os.environ["NUM_QUERY"]); blank_pct=int(os.environ["BLANK_PCT"])
iql_log_dir=Path(os.environ["IQL_LOG_DIR"])
meta_path=Path("human_label/_grid_metadata")/f"{label_tag}.label_alignment.json"
prog=sorted(iql_log_dir.glob("**/progress.txt"))
if not prog:
    print(f"WARN: no progress.txt under {iql_log_dir}"); raise SystemExit(0)
data=np.loadtxt(prog[-1])
if data.ndim==1: data=data.reshape(1,-1)
last10=float(data[-10:,1].mean()) if len(data)>=10 else float(data[:,1].mean())
final=float(data[-1,1])
meta=json.load(open(meta_path)) if meta_path.exists() else {}
summary=dict(condition_id=cond_id, label_tag=label_tag, num_query=num_query,
    blank_pct=blank_pct, label_mode="blind_oracle", rm_seed=seed, iql_seed=seed,
    label_alignment=float(meta.get("label_alignment", 1.0)),
    last10_eval_reward=last10, final_eval_reward=final, n_evals=int(len(data)))
out=iql_log_dir/"eval_summary.json"
json.dump(summary, open(out, "w"), indent=2)
print(f"wrote {out}\n  {json.dumps(summary, indent=2)}")
PY
echo "Run $COND_ID seed=$SEED done"
