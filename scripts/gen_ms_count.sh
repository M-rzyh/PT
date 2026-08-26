#!/bin/bash
#SBATCH --job-name=pt-gen-ms-count
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/label_gen/gen_ms_count_%j.out
#SBATCH --error=logs/label_gen/gen_ms_count_%j.err
#
# WHAT: Step 1 of 3 (gen -> submit -> run). Makes the COUNT labels: samples N
#       pairs per seed and oracle-labels them (clean, no noise).
#
# Generate multi-seed COUNT-axis preference labels (clean; x-axis = #labels N).
#   sbatch scripts/gen_ms_count.sh [NSEEDS]
#
# NSEEDS (default 30): seeds 0..NSEEDS-1.
# The N values are the COUNTS list below — edit to change the count axis.
#
# Mirrors submit_ms_count.sh. Calls setup_grid_labels_ms.py once per seed.
set -euo pipefail
mkdir -p logs/label_gen
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
cd /home/marzii/PT/PreferenceTransformer

COUNTS=(50 100 250 500 750 1000)   # <-- edit to change which counts are generated

NSEEDS=${1:-30}; shift || true
LAST=$((NSEEDS - 1))

DATASET=$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5

echo "Gen COUNT labels: counts=${COUNTS[*]}, seeds 0-${LAST}"
for S in $(seq 0 "${LAST}"); do
  python scripts/experiment_grid/setup_grid_labels_ms.py \
      --hdf5 "$DATASET" \
      --label_root human_label \
      --training_seed "$S" \
      --n_counts "${COUNTS[@]}"
done
echo "Done."
