#!/bin/bash
#SBATCH --job-name=pt-gen-blind-oracle
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/label_gen/gen_blind_oracle_%j.out
#SBATCH --error=logs/label_gen/gen_blind_oracle_%j.err
#
# Generate BLIND-ORACLE preference labels (frame-blanking difficulty, Exp 1).
# The oracle scores each segment using reward on VISIBLE (non-blanked) frames only
# -> labels degrade with blank %. Labels land in human_label/ under distinct
# 'blindoracle{P}'/'blindoracleclean' tags (non-destructive). Segments come from
# the true 8-D mixture; the agent still learns the NORMAL task.
#   sbatch scripts/gen_ms_blind_oracle.sh [NSEEDS]   (default 5)
set -euo pipefail
mkdir -p logs/label_gen
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH

cd /home/marzii/PT/PreferenceTransformer
# RENDERED mixture (same data the human labels from real videos) — keeps the
# blind-oracle (#3) and blind-human (#4) PT arms on identical segments.
DATASET=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
NSEEDS=${1:-5}
LEVELS="0 25 50 75"

for S in $(seq 0 $((NSEEDS - 1))); do
    python scripts/experiment_grid/setup_grid_labels_ms.py \
        --hdf5 "$DATASET" \
        --label_root human_label \
        --training_seed "$S" \
        --num_total 350 \
        --blind_oracle_pcts $LEVELS \
        --blind_blank_mode block \
        --blind_block_len 10
done
echo "Blind-oracle labels generated (blank {$LEVELS}%, seeds 0-$((NSEEDS - 1)))."
