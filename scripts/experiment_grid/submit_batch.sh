#!/bin/bash
# Submit a small batch of grid conditions. Usage:
#   bash scripts/experiment_grid/submit_batch.sh <condition_name> <num_query> [more pairs ...]
# Example:
#   bash submit_batch.sh \
#       lunarlander-grid-N50-clean 50 \
#       lunarlander-grid-N100-clean 100 \
#       lunarlander-grid-N500-clean 500

set -euo pipefail
cd /home/marzii/PT/PreferenceTransformer

if [ $(( $# % 2 )) -ne 0 ]; then
    echo "usage: $0 <condition> <num_query> [<condition> <num_query> ...]" 1>&2
    exit 1
fi

while [ $# -gt 0 ]; do
    COND=$1; N=$2; shift 2
    echo "[submit] $COND  N=$N"
    sbatch --export=ALL,CONDITION=$COND,NUM_QUERY=$N \
        --job-name=pt-grid-${COND} \
        scripts/experiment_grid/run_grid_condition.sh
done
