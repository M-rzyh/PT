#!/bin/bash
#SBATCH --job-name=pt-rerender-mr-s0
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/render/%x_%j.out
#SBATCH --error=logs/render/%x_%j.err
#
# Re-train sb3 SAC seed=0 with rendering enabled, up to the medium-replay
# checkpoint (~70 K steps). Produces a new HDF5 + per-episode mp4s in
# lockstep with the replay-buffer transitions, for Run B (human) labelling
# on medium-replay.
#
# Output: $SCRATCH/PT/lunarlander/seed_0/render/medium-replay-v2/
#   actor.zip, replay.pkl
#   lunarlander-medium-replay-v2.hdf5
#   episodes/episode_NNNNN.mp4
#   episodes/index.pkl

set -euo pipefail
mkdir -p logs/render

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

cd /home/marzii/PT/PreferenceTransformer

OUT=$SCRATCH/PT/lunarlander/seed_0/render/medium-replay-v2

python -m scripts.offline_data.train_sac_with_video \
    --save_dir "$OUT" \
    --total_steps 70000 \
    --seed 0 \
    --fps 20

echo "rerender medium-replay seed=0 done → $OUT"
