# PT noise/count grid — workflow

Goal: train PT reward + IQL agents on the LunarLander **mixture** dataset under
varying preference-label corruption (noise axis) or label budget (count axis),
30 seeds each, then plot mean eval reward.

Two corruption modes (noise axis):
- `random_replace` — corrupted label → uniform {-1,0,1}  (tags `noise{P}`/`clean`)
- `deterministic_flip` — corrupted label → inverted 0↔1  (tags `flipnoise{P}`/`flipclean`)

---

## Step 1 — make the labels  (CPU, seconds)

Samples 1000 segment pairs per seed, oracle-labels them, writes corrupted copies
to `human_label/`. One dir per (seed, level).

    sbatch scripts/gen_labels_ms_noise_count.sh   # random_replace (noise + count)
    sbatch scripts/gen_labels_ms_flip.sh          # deterministic_flip

Calls: `scripts/experiment_grid/setup_grid_labels_ms.py`

## Step 2 — train the agents  (GPU, ~hrs each, 30 per level)

Each task: PT reward model → IQL 1M steps → `eval_summary.json`.
One array job per noise level (array index = seed).

    # random_replace, e.g. noise 40%:
    sbatch --array=0-29 --export=ALL,NOISE_PCT=40 \
        scripts/experiment_grid/run_grid_ms_noise.sh

    # deterministic_flip sweep (all 6 levels at once):
    bash scripts/experiment_grid/submit_ms_noise_flip.sh 30

    # count axis, e.g. N=250:
    sbatch --array=0-29 --export=ALL,N_COUNT=250 \
        scripts/experiment_grid/run_grid_ms_count.sh

Results land in: `$SCRATCH/PT/lunarlander/grid_mixture_ms/<COND_ID>/seed_*/`

## Step 3 — plot  (CPU, seconds)

    python scripts/plot_grid_pt_vs_gail.py      # PT vs GAIL, count + noise
    python scripts/plot_pt_flip_vs_random.py    # PT flip vs random, noise

Figures → `figures/`

---

## Naming (dirs in human_label/)

    lunarlander-grid-ms-N1000-flipnoise60-s7
                      │    │       │       └ seed
                      │    │       └ corruption (clean/noise{P}/flipclean/flipnoise{P}/count-N{N})
                      │    └ # labels
                      └ ms = per-seed label draw (each seed = different pairs)

Implicit for all `grid` names: oracle labels, mixture dataset.
Start rows change with seed; only labels change with noise level.
