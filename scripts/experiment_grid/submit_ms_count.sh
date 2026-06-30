#!/bin/bash
# WHAT: Step 2 of 3. Launcher ONLY — submits one Slurm array (one task per seed)
#       per count N by calling run_grid_ms_count.sh. Trains nothing itself.
# Submit the multi-seed COUNT-axis sweep (clean labels; x-axis = number of labels N).
#   bash scripts/experiment_grid/submit_ms_count.sh [NSEEDS] [DEP]
#
# NSEEDS (default 30):  array runs seeds 0..NSEEDS-1.
# DEP    (default none): Slurm dependency, e.g. afterany:JOBID[:JOBID...]. "-" or omit = none.
#
# The N values swept are the COUNTS list below — edit it to change the count axis.
# Drives run_grid_ms_count.sh (one array job per N). Labels for each N must exist.
set -euo pipefail
cd /home/marzii/PT/PreferenceTransformer

COUNTS=(50 100 250 500 750 1000)   # <-- edit to change which counts are swept

NSEEDS=${1:-30}; shift || true
DEP=${1:-};      shift || true
LAST=$((NSEEDS - 1))

DEPARG=()
if [ -n "$DEP" ] && [ "$DEP" != "-" ]; then DEPARG=(--dependency="$DEP"); echo "dependency: $DEP"; fi

echo "Count sweep: counts=${COUNTS[*]}, ${NSEEDS} seeds (0-${LAST})"
for N in "${COUNTS[@]}"; do
  echo "[submit] count N=$N  array=0-${LAST}"
  sbatch --array=0-${LAST} "${DEPARG[@]}" \
    --export=ALL,N_COUNT=${N} \
    --job-name=pt-ms-count-N${N} \
    scripts/experiment_grid/run_grid_ms_count.sh
done
echo "Done. Watch: squeue -u $USER"
