#!/bin/bash
# Submit PT blind-oracle training (frame-blanking, Exp 1): reward model + IQL on the
# true 8-D mixture with blind-oracle labels. One Slurm array (one task per seed) per
# blank level. Labels must exist (run gen_ms_blind_oracle.sh first).
#   bash scripts/experiment_grid/submit_ms_blind_oracle.sh [NSEEDS] [pct ...]
set -euo pipefail
cd /home/marzii/PT/PreferenceTransformer

NSEEDS=${1:-5}; shift || true
PCTS=("$@"); [ ${#PCTS[@]} -eq 0 ] && PCTS=(0 25 50 75)
LAST=$((NSEEDS - 1))

echo "PT blind-oracle sweep: ${NSEEDS} seeds (0-${LAST}), levels: ${PCTS[*]}"
for P in "${PCTS[@]}"; do
  echo "[submit] blind_oracle blank=${P}%  array=0-${LAST}"
  sbatch ${DEP:+--dependency=afterok:$DEP} --array=0-${LAST} --export=ALL,BLANK_PCT=${P} \
    --job-name=pt-blindoracle-p${P} \
    scripts/experiment_grid/run_grid_ms_blind_oracle.sh
done
echo "Done. Watch: squeue -u $USER"
