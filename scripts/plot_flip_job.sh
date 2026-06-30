#!/bin/bash
#SBATCH --job-name=pt-flip-plot
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/flip_plot_%j.out
#SBATCH --error=logs/flip_plot_%j.err
set -euo pipefail
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
cd /home/marzii/PT/PreferenceTransformer
python scripts/plots/plot_pt_flip_vs_random.py
