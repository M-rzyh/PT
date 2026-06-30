#!/bin/bash
# WHAT: Step 2 of 3. Launcher ONLY — submits one Slurm array (one task per seed)
#       per noise level by calling run_grid_ms_noise.sh. Trains nothing itself.
# Submit a multi-seed PT NOISE sweep (one corruption mode).
#   bash scripts/experiment_grid/submit_ms_noise.sh [MODE] [N] [NSEEDS] [DEP] [pct ...]
#
# MODE   (default random_replace): corruption model. Accepts:
#          random_replace | random | rr   -> uniform {-1,0,1}
#          deterministic_flip | flip       -> invert 0<->1
# N      (default 1000): label budget (50 for low-budget plot, 1000 for full).
# NSEEDS (default 30):   array runs seeds 0..NSEEDS-1.
# DEP    (default none): Slurm dependency, e.g. afterany:JOBID[:JOBID...]. "-"/omit = none.
# pct    (default 0 10 20 30 40 50 60 70 80 90 100): noise levels to sweep.
#
# Drives run_grid_ms_noise.sh. Labels for this MODE + N must already exist.
set -euo pipefail
cd /home/marzii/PT/PreferenceTransformer

MODE=${1:-random_replace}; shift || true
case "$MODE" in
  random_replace|random|rr) MODE=random_replace;     TAG=rr ;;
  deterministic_flip|flip)  MODE=deterministic_flip; TAG=flip ;;
  *) echo "ERROR: unknown MODE '$MODE' (use random_replace|deterministic_flip)" 1>&2; exit 1 ;;
esac

N=${1:-1000};    shift || true
NSEEDS=${1:-30}; shift || true
DEP=${1:-};      shift || true
PCTS=("$@")
[ ${#PCTS[@]} -eq 0 ] && PCTS=(0 10 20 30 40 50 60 70 80 90 100)
LAST=$((NSEEDS - 1))

DEPARG=()
if [ -n "$DEP" ] && [ "$DEP" != "-" ]; then DEPARG=(--dependency="$DEP"); echo "dependency: $DEP"; fi

echo "Noise sweep: mode=$MODE, N=$N, ${NSEEDS} seeds (0-${LAST}), levels: ${PCTS[*]}"
for P in "${PCTS[@]}"; do
  echo "[submit] $TAG N=$N noise=${P}%  array=0-${LAST}"
  sbatch --array=0-${LAST} "${DEPARG[@]}" \
    --export=ALL,NOISE_PCT=${P},NOISE_MODE=${MODE},NUM_QUERY=${N} \
    --job-name=pt-ms-${TAG}-N${N}-p${P} \
    scripts/experiment_grid/run_grid_ms_noise.sh
done
echo "Done. Watch: squeue -u $USER"
