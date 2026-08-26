#!/bin/bash
#SBATCH --job-name=pt-gen-ms-noise
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/label_gen/gen_ms_noise_%j.out
#SBATCH --error=logs/label_gen/gen_ms_noise_%j.err
#
# WHAT: Step 1 of 3 (gen -> submit -> run). Makes the NOISE labels: samples N
#       pairs per seed, oracle-labels them, then corrupts a fraction at each
#       level (mode = random_replace or deterministic_flip).
#
# Generate multi-seed NOISE-axis preference labels (one corruption mode).
#   sbatch scripts/gen_ms_noise.sh [MODE] [N] [NSEEDS] [pct ...]
#
# MODE   (default random_replace): random_replace|random|rr  or  deterministic_flip|flip
# N      (default 1000): label budget (50 for low-budget plot, 1000 for full).
# NSEEDS (default 30):   seeds 0..NSEEDS-1.
# pct    (default 0 10 20 30 40 50 60 70 80 90 100): noise levels.
#
# Mirrors submit_ms_noise.sh. Calls setup_grid_labels_ms.py once per seed.
set -euo pipefail
mkdir -p logs/label_gen
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
cd /home/marzii/PT/PreferenceTransformer

MODE=${1:-random_replace}; shift || true
case "$MODE" in
  random_replace|random|rr) MODE=random_replace ;;
  deterministic_flip|flip)  MODE=deterministic_flip ;;
  *) echo "ERROR: unknown MODE '$MODE' (use random_replace|deterministic_flip)" 1>&2; exit 1 ;;
esac
N=${1:-1000};    shift || true
NSEEDS=${1:-30}; shift || true
LEVELS=("$@")
[ ${#LEVELS[@]} -eq 0 ] && LEVELS=(0 10 20 30 40 50 60 70 80 90 100)
LAST=$((NSEEDS - 1))

DATASET=$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5

echo "Gen NOISE labels: mode=$MODE, N=$N, seeds 0-${LAST}, levels: ${LEVELS[*]}"
for S in $(seq 0 "${LAST}"); do
  python scripts/experiment_grid/setup_grid_labels_ms.py \
      --hdf5 "$DATASET" \
      --label_root human_label \
      --training_seed "$S" \
      --num_total "$N" \
      --noise_mode "$MODE" \
      --noise_pcts "${LEVELS[@]}"
done
echo "Done."
