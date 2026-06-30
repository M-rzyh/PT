# PT count grid — workflow

Count axis: how PT scales with the **number of labels** N (clean labels, no noise),
on the LunarLander mixture dataset, 30 seeds. x-axis = N; y = mean eval reward.
Default N values: 50 100 250 500 750 1000.

Three steps, each its own file (all paths from repo root):

## 1. Make labels — `gen_ms_count.sh`  (CPU, seconds)
Samples one 1000-pair pool per seed, oracle-labels it; each N = first N pairs.
    sbatch scripts/gen_ms_count.sh [NSEEDS]
N values come from the COUNTS list inside the file (edit to change the axis).

## 2. Launch training — `submit_ms_count.sh`  (launcher)
Fires one Slurm array (seeds 0..NSEEDS-1) per N.
    bash scripts/experiment_grid/submit_ms_count.sh [NSEEDS] [DEP]

## 3. The work per (N, seed) — `run_grid_ms_count.sh`  (GPU, ~30 min each)
One array task = PT reward model → IQL (1M steps) → eval_summary.json.
Not run directly; submit_ms_count.sh calls it. Manual single N:
    sbatch --array=0-29 --export=ALL,N_COUNT=250 scripts/experiment_grid/run_grid_ms_count.sh

Results: $SCRATCH/PT/lunarlander/grid_mixture_ms/lunarlander-grid-ms-count-N{N}/seed_*/

## Plot
    python scripts/plot_grid_pt_vs_gail.py    # → figures/grid_count_pt_vs_gail.png

---
Naming: `lunarlander-grid-ms-count-N{N}-s{S}` — ms = per-seed label draw; N = #labels; S = seed.
Implicit for all `grid` names: oracle labels, mixture dataset.
Noise axis is the sibling trio: gen_ms_noise.sh → submit_ms_noise.sh → run_grid_ms_noise.sh.
