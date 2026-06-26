#!/bin/bash
#SBATCH --job-name=pt-render-mix-prep
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Render one of {random, medium, expert} from seed_0 with simulator frames,
# in preparation for the rendered mixture HDF5 used by mixture-Run-B
# labelling. Picked via env var VARIANT.
#
# Output: $SCRATCH/PT/lunarlander/seed_0/render/<variant>-v2/

set -euo pipefail
mkdir -p logs

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

: "${VARIANT:?must set VARIANT=random|medium|expert}"
: "${NUM_STEPS:?must set NUM_STEPS=...}"

case "$VARIANT" in
    random) ACTOR=random ;;
    medium) ACTOR=$SCRATCH/PT/lunarlander/seed_0/sac_run/medium/actor.zip ;;
    expert) ACTOR=$SCRATCH/PT/lunarlander/seed_0/sac_run/expert/actor.zip ;;
    *) echo "ERROR: VARIANT must be random|medium|expert (got '$VARIANT')" 1>&2; exit 1 ;;
esac

OUT=$SCRATCH/PT/lunarlander/seed_0/render/${VARIANT}-v2-mixprep

python -m scripts.offline_data.rollout_with_video \
    --actor "$ACTOR" \
    --output_dir "$OUT" \
    --num_steps "$NUM_STEPS" \
    --fps 20 \
    --seed 0 \
    --variant "$VARIANT"

echo "render ${VARIANT} (${NUM_STEPS} steps) → $OUT"
