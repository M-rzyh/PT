#!/bin/bash
#SBATCH --job-name=pt-build-vanish
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/scratch/marzii/PT/logs/vanish/%x_%j.out
#SBATCH --error=/scratch/marzii/PT/logs/vanish/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# (time/mem raised for the background-plate pass: each segment's FULL episode is decoded —
#  up to 911 frames — and a per-episode plate is cached.)
#
# Build the three N=100 LANDER-VANISH preference video sets (25 / 50 / 75 %) for the human
# PT arm of the lander-vanishing study.
#
# WHY THE SETS ARE SUBSETS OF THE CLEAN 350
#   The 0% condition is taken for free from the first 100 of the existing clean human labels,
#   so the vanish sets MUST be built from exactly those same pairs (pair_000..pair_099).
#   Any other sampling would make the 0% point incomparable to 25/50/75.
#
# Never touches the clean source: each level is copied to its own directory first, and the
# masker then rewrites the copies in place.
#
#   sbatch scripts/preference/build_vanish_query_sets.sh
set -euo pipefail

PY=/scratch/marzii/envs/pt/bin/python
REPO=/home/marzii/PT/PreferenceTransformer
SRC=/scratch/marzii/PT/lunarlander/frame_blanking/pt_human/videos_human_n350_clean
DEST_ROOT=/scratch/marzii/PT/lunarlander/lander_vanish/pt_human
N=100          # pairs per set (the low budget)
QLEN=100       # frames per clip
BLOCK=5        # b=5 -> 0.25 s at 20 fps
SEED=0

mkdir -p /scratch/marzii/PT/logs/vanish "$DEST_ROOT"

for PCT in 25 50 75; do
  DEST="$DEST_ROOT/videos_human_n100_vanish${PCT}"
  echo "=============================================================="
  echo "[build] level ${PCT}%  ->  $DEST"

  if [[ -d "$DEST" ]]; then
    echo "  EXISTS, skipping copy+mask (delete it by hand to rebuild)"
    continue
  fi

  # --- copy pair_000..pair_099 out of the clean set ---
  mkdir -p "$DEST/batch_000"
  for q in $(seq 0 $((N - 1))); do
    P=$(printf 'pair_%03d' "$q")
    cp -r "$SRC/batch_000/$P" "$DEST/batch_000/$P"
  done

  # --- metadata: truncate n_pairs / indices to the 100-pair subset ---
  # label_web loops `range(meta["n_pairs"])`, so a stale 350 would advertise pairs that
  # aren't there; the indices must also line up 1:1 with the pairs we copied.
  "$PY" - "$SRC/batch_000/metadata.pkl" "$DEST/batch_000/metadata.pkl" "$N" <<'EOF'
import pickle, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = pickle.load(open(src, 'rb'))
m['n_pairs'] = n
for k in ('indices_1', 'indices_2'):
    if k in m:
        m[k] = m[k][:n]
m['subset_of'] = src
m['subset_note'] = f'first {n} pairs of the clean 350 (0% baseline reuses these labels)'
pickle.dump(m, open(dst, 'wb'))
print(f"  metadata: n_pairs={m['n_pairs']} indices_1[:3]={list(m['indices_1'][:3])}")
EOF

  # --- vanish the lander on the blanked blocks (rewrites the copies in place) ---
  "$PY" "$REPO/scripts/preference/blank_lander_videos.py" \
      --videos_dir "$DEST" --num_query "$N" --query_len "$QLEN" \
      --blank_prob "0.${PCT}" --block_len "$BLOCK" --blank_seed "$SEED"
done

echo "=============================================================="
echo "[build] done. Verify with: $PY $REPO/scripts/preference/verify_vanish_sets.py"
