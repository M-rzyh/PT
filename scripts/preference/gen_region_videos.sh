#!/bin/bash
#SBATCH --job-name=pt-region-vid
#SBATCH --account=aip-mtaylor3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/region_vid_%j.out
#SBATCH --error=logs/region_vid_%j.err
#
# Region masking (Sweep 1) — PT human video sets. Copies the CLEAN N=350 segment
# set and paints one fixed black rectangle over every frame of each segment (area =
# frac*frame, location uniform-random per segment, region_seed=0 so identical across
# training seeds). Levels 25/50/75%; 0% reuses videos_human_n350_clean / humanblock0.
# Isolated in region_mask/ (sibling of frame_blanking/); nothing existing is modified.
set -euo pipefail
mkdir -p logs
module --force purge; module load StdEnv/2023
eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
unset PYTHONPATH
cd /home/marzii/PT/PreferenceTransformer

SRC=$SCRATCH/PT/lunarlander/frame_blanking/pt_human/videos_human_n350_clean
DEST=$SCRATCH/PT/lunarlander/region_mask/pt_human
NUM=350
mkdir -p "$DEST"

for PF in "25 0.25" "50 0.5" "75 0.75"; do
    set -- $PF; P=$1; FRAC=$2
    OUT=$DEST/videos_human_n350_region${P}
    echo "=== region ${P}% (frac=$FRAC) -> $OUT ==="
    rm -rf "$OUT"
    cp -r "$SRC" "$OUT"
    python scripts/preference/mask_region_videos.py \
        --videos_dir "$OUT" --num_query "$NUM" --region_frac "$FRAC" --region_seed 0
done
echo "Region video sets done: region25/50/75 (N=$NUM pairs each)."
