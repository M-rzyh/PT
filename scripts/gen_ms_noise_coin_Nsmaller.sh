#!/bin/bash
#SBATCH --job-name=pt-gen-coin-n50
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=logs/gen_coin_n50_%j.out
#SBATCH --error=logs/gen_coin_n50_%j.err
#
# COIN random_replace labels at N=50 (11 levels, 30 seeds). Same grid mixture +
# same per-seed pairs as the existing EXACT N=50 (exnoise) set, so coin-vs-exact
# differ only in the corruption *selection* -> shows the small-N count wobble.
# Tags: noise{P}/clean (coin), distinct from exnoise/exclean (exact).

# Just a smaller version of gen_ms_noise.sh, but with NOISE_SELECTION=coin and NUM=50/350/100.

set -euo pipefail
mkdir -p logs
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
cd /home/marzii/PT/PreferenceTransformer

DATASET=$SCRATCH/PT/lunarlander/mixture/lunarlander-mixture-v2-s0.hdf5
LEVELS="0 10 20 30 40 50 60 70 80 90 100"
NSEEDS=${1:-30}
MODE=${2:-random_replace}    # random_replace (noise*) or deterministic_flip (flipnoise*)
NUM=${3:-100}                 # num_total preferences (50 or 350 or 100)

for S in $(seq 0 $((NSEEDS - 1))); do
    python scripts/experiment_grid/setup_grid_labels_ms.py \
        --hdf5 "$DATASET" --label_root human_label --training_seed "$S" \
        --num_total "$NUM" --noise_pcts $LEVELS \
        --noise_mode "$MODE" --noise_selection coin
done
echo "COIN $MODE N=$NUM labels done ($NSEEDS seeds, 11 levels)."
