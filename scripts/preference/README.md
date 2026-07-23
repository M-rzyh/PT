# PT human preference labeling — with / without frame-blanking

How to produce human-labeled PT runs on the **rendered** LunarLander mixture, either
plain (0% = no blanking) or with **frame-blanking** difficulty (the labeler can't see
the lander during blackouts → degraded "blind" preferences). The agent always solves
the **normal** task; blanking only degrades the *labels*.

Mixture (real videos + training both use this):
```
$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
```
Env: `/scratch/marzii/envs/pt/bin/python`.  Videos/labels live under
`$SCRATCH/PT/lunarlander/frame_blanking/pt_human/`.

---

## The pipeline (order of operations)
```
1. sample_query_indices.py     pick N segment pairs            -> query dir (indices_*)
2. extract_segment_videos.py   slice REAL gym frames           -> batch_/pair_/seg_{A,B}.mp4  (CLEAN)
3. blank_segment_videos.py     black out frames  [BLANKING ONLY, skip for 0%]
4. label_web.py                YOU label in the browser        -> labels_*.pkl
5. labels_web_to_pt_format.py  convert to PT labels            -> human_label/<TAG>/
6. train (reward model + IQL)  on the rendered mixture + <TAG> -> iql_runs/...
```
**0% = just skip step 3.**  Blanking is a separate post-processing step, not a flag on
the trainer.

---

## What the blanking values do (`blank_segment_videos.py`)
| flag | meaning |
|---|---|
| `--blank_mode block` | black out **contiguous runs** of frames (recommended — clearly visible). `stochastic` = independent per-frame (blends into flicker at 20 fps → avoid); `deterministic` = every k-th. |
| `--blank_prob 0.5` | fraction of **blocks** blanked → ~50% of frames black. `0` = nothing blanked. |
| `--block_len 10` | blackout length in frames. 20 fps ⇒ **B=10 → 0.5 s**, B=20 → 1 s. |
| `--blank_seed 0` | reproducible mask. |

`blank_segment_videos.py` also repoints `index.pkl` at the local blanked clips (else
`label_web` follows stale absolute paths back to the clean originals).

---

## Recipe A — 0% baseline (regular PT, no blanking)
**You already have this**: `human100_mixture` (100 labels, 5 seeds, rendered mixture,
trained). Reuse it — no need to relabel.

To make a *fresh* 0% run instead:
```bash
cd /home/marzii/PT/PreferenceTransformer
PY=/scratch/marzii/envs/pt/bin/python
RMIX=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2/lunarlander-mixture-v2.hdf5
RDIR=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2
B=$SCRATCH/PT/lunarlander/frame_blanking/pt_human

$PY scripts/preference/sample_query_indices.py --hdf5 $RMIX --output_dir $B/q_0pct --num_query 100 --query_len 100 --seed 42
$PY scripts/preference/extract_segment_videos.py --rollout_dir $RDIR --query_dir $B/q_0pct --output_dir $B/vid_0pct --num_query 100 --query_len 100
# NO blank step
$PY scripts/preference/label_web.py --query_dir $B/vid_0pct --output $B/labels_0pct.pkl --port 8711
$PY scripts/preference/labels_web_to_pt_format.py --labels $B/labels_0pct.pkl --output_dir human_label/lunarlander-mixture-v2-humanblock0
```

## Recipe B — 50% blanking (B=10 blocks)
Renders on the **same 100 pairs** as `human100_mixture` so 0% vs 50% is a clean
within-subject comparison (reuse `human100_mixture` as the 0% point).
```bash
cd /home/marzii/PT/PreferenceTransformer
PY=/scratch/marzii/envs/pt/bin/python
RDIR=$SCRATCH/PT/lunarlander/seed_0/render/mixture-v2
B=$SCRATCH/PT/lunarlander/frame_blanking/pt_human
QIDX=human_label/lunarlander-mixture-v2-human-s0        # existing human100 pairs

$PY scripts/preference/extract_segment_videos.py --rollout_dir $RDIR --query_dir $QIDX --output_dir $B/vid_50pct_b10 --num_query 100 --query_len 100
$PY scripts/preference/blank_segment_videos.py --videos_dir $B/vid_50pct_b10 --num_query 100 --query_len 100 --blank_mode block --block_len 10 --blank_prob 0.5 --blank_seed 0
$PY scripts/preference/label_web.py --query_dir $B/vid_50pct_b10 --output $B/labels_50pct_b10.pkl --port 8712
$PY scripts/preference/labels_web_to_pt_format.py --labels $B/labels_50pct_b10.pkl --output_dir human_label/lunarlander-mixture-v2-humanblock50
```
For a **different blank level**, change `--blank_prob` (e.g. 0.25 / 0.75) and the output
tag. For a different blackout length, change `--block_len` (10 → 0.5 s, 20 → 1 s).

## Step 6 — train (both recipes)
Reward model + IQL on the rendered mixture + your label tag, e.g. adapt
`scripts/oracle_vs_human_vs_pebble/run_pt_pipeline_human_mixture.sh` with
`--env=<TAG>` (and `DATASET` = the rendered mixture). Eval = `last10_eval_reward`
(matches the noise plots).

---

## What a "seed" means for human runs
The human labels a query set **once**; those labels are **identical across all seeds**
(e.g. `human100_mixture` s0–s4 share the exact same `label_human` file — verified by
hash). A seed here is a **training seed, not a labeling seed**: it re-runs the
reward-model + IQL training with different network initialization, minibatch order,
and environment-eval seeds. So the 5 seeds measure **training / RL variance** given the
*same* human data — that's what produces the mean±std band on the plots — **not**
variance in the human's labeling. (Same convention as the noise/oracle sweeps: one
label set, N training seeds for the band.)

---

## label_web browser controls
`http://localhost:<port>` (SSH-forward the port). Per pair: **A better / B better /
Equal / Skip**. Auto-saves after each click. If a port is busy, pick another (8711,
8712, 8734, …). Use a fresh/incognito tab to avoid stale-cache surprises.
