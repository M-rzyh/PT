#!/bin/bash
#SBATCH --job-name=replot-mixture
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/replot_mixture_%j.out
#SBATCH --error=logs/replot_mixture_%j.err

mkdir -p logs
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

python scripts/plots/plot_runs_mixture_no_pebblelabels.py
python scripts/plots/plot_runs_mixture.py
echo "done"
