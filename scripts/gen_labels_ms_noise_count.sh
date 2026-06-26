#!/bin/bash
#SBATCH --job-name=pt-gen-labels-s0-29
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gen_labels_s0_29_%j.out
#SBATCH --error=logs/gen_labels_s0_29_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH

cd /home/marzii/PT/PreferenceTransformer

DATASET=$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5

echo "=== Noise axis labels (seeds 0-29) ==="
for S in $(seq 0 29); do
    python scripts/experiment_grid/setup_grid_labels_ms.py \
        --hdf5 "$DATASET" \
        --label_root human_label \
        --training_seed "$S" \
        --noise_pcts 0 10 20 25 30 40 50 60 70 75 80 90 100
done

echo ""
echo "=== Count axis labels (seeds 0-29) ==="
for S in $(seq 0 29); do
    python scripts/experiment_grid/setup_grid_labels_ms.py \
        --hdf5 "$DATASET" \
        --label_root human_label \
        --training_seed "$S" \
        --n_counts 50 100 250 500 750 1000
done

echo "All labels for seeds 0-29 generated."
