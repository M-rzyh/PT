# PT (Preference Transformer) workspace handoff

This document covers everything done since the user asked Claude to explain
the Preference Transformer paper's algorithm. It exists so a fresh
Claude/agent session can pick up the work without re-reading the previous
transcript.

> **Status (2026-05-06):** Phase 0 (env setup) is **done and verified on
> the L40S** (smoke #2 = SLURM job 4890090). Phase A (LunarLander dataset
> generation) is the next step — see §6.

---

## 1. Paper context

**Preference Transformer** — Kim et al., ICLR 2023.
[Paper link](https://openreview.net/forum?id=Peot1SFDX0).

Algorithm summary (more detail in the original paper, but for orientation):

- **Setting:** *offline* preference-based RL. Start from a fixed offline
  trajectory dataset (D4RL / Robomimic). Sample segment pairs uniformly,
  show to a human, get a preference label `y ∈ {0, 1, 0.5}`.
- **Model:** two stacked sub-networks
  1. **Causal transformer** (GPT-style, masked) — produces hidden states `x_t`
     for each `(s_t, a_t)`.
  2. **Preference attention layer** (bidirectional) — projects each `x_t` into
     `(k_t, q_t, r̂_t)`. The non-Markovian rewards `r̂_t` are scalar values; the
     bidirectional self-attention produces importance weights `w_t`.
- **Preference predictor** (Eq. 3):
  `P[σ¹ ≻ σ⁰] = exp(Σ w_t¹·r̂_t¹) / [exp(Σ w_t⁰·r̂_t⁰) + exp(Σ w_t¹·r̂_t¹)]`
- **Loss:** cross-entropy against human label (Eq. 2).
- **Inference at RL time:** for each `(s_t, a_t)`, feed PT the previous `H`
  transitions and use the `t`-th `r̂_t` as the relabeled reward.
- **Downstream RL:** **IQL** (offline). They use the standard IQL hyperparams,
  only the reward source changes.
- **Datasets in the paper:** D4RL AntMaze (medium/large play+diverse), D4RL
  MuJoCo (Hopper / Walker2d medium-replay + medium-expert), Robosuite (lift,
  can, square — proficient-human PH and multi-human MH variants).

Difference vs prior work:
- **MR** (Christiano '17, PEBBLE) = MLP `r̂(s,a)`, equal weights.
- **NMR** (Early '22) = LSTM `r̂(σ_{≤t})`, equal weights.
- **PT** (this paper) = causal-transformer `r̂` + bidirectional importance
  weights `w_t`.

---

## 2. The user's plan

User wants to apply PT to LunarLander to compare with their PEBBLE/GAIL work.
Neither D4RL nor Robomimic includes LunarLander, so:

1. Build a custom LunarLander offline dataset matching D4RL's MuJoCo schema
   (the user explicitly wants the SAME generation methodology and SAME
   performance levels as D4RL: random / medium / medium-replay / medium-expert
   / expert).
2. Collect human preferences over segments sampled from that dataset.
3. Train PT on those preferences.
4. Train IQL on the dataset relabeled with PT rewards.
5. Compare against the user's PEBBLE oracle and PEBBLE-with-human runs.

Schema for the custom dataset (D4RL convention, HDF5):
```python
{
    "observations":      np.float32, shape=(N, 8),
    "actions":           np.float32, shape=(N, 2),
    "rewards":           np.float32, shape=(N,),
    "next_observations": np.float32, shape=(N, 8),
    "terminals":         np.bool_,   shape=(N,),
    "timeouts":          np.bool_,   shape=(N,),
}
```
With `N ≈ 1M` per variant. The five variants and their generation recipe match
D4RL Hopper/Walker2d's scheme: random uniform actions / SAC-medium /
SAC-medium replay buffer / 50:50 medium+expert / SAC-expert. For LunarLander,
"expert" target ~+260 reward (matches user's PEBBLE-oracle final), "medium"
~+80–130 (a partially-trained SAC checkpoint).

The user's existing oracle PEBBLE runs (jobs 4883370–4883374) already produce
SAC checkpoints at step 1M that qualify as "expert", and earlier checkpoints
(~step 250K) qualify as "medium". This means much of the data generation can
reuse existing trained models — we just need to add periodic checkpoint+replay
saves to one fresh SAC training run, then write rollout scripts.

---

## 3. Workspace boundary (IMPORTANT)

The user explicitly drew a line:
**All PT-related code and data lives in `/home/marzii/PT/PreferenceTransformer/`.**
Do **NOT** mix PT code into `BPref3` (PEBBLE) or `IRL3` (GAIL).

- PT-related Python scripts → `/home/marzii/PT/PreferenceTransformer/scripts/` (or sibling dirs).
- PT-related data (the 5 LunarLander HDF5 files) → `$SCRATCH/PT/lunarlander/` (separate from `$SCRATCH/compare_runs/pebble/...`).
- Comparison plots may live under `$SCRATCH/compare_runs/` since they cross-cut, but their source data must come from the PT workspace.

This is also saved as a memory:
`/home/marzii/.claude/projects/-home-marzii-BPref3/memory/project_pt_workspace.md`

---

## 4. What's on disk

Repo: `/home/marzii/PT/PreferenceTransformer/` (cloned upstream JAX/Flax code).
Layout:
- `train_offline.py`, `robosuite_train_offline.py`, `train_finetune.py` — entry points (D4RL/Robosuite).
- `JaxPref/new_preference_reward_main.py` — main PT/NMR/MR reward training.
- `JaxPref/model.py`, `flaxmodels/flaxmodels/gpt2/trajectory_gpt2.py` — actual transformer model.
- `JaxPref/sampler.py`, `JaxPref/reward_transform.py`, `JaxPref/jax_utils.py`.
- `human_label/` — pre-collected human prefs for D4RL/Robomimic tasks.
- `d4rl/` — vendored D4RL package (needs MuJoCo to install).
- `flaxmodels/`, `wrappers/`, `viskit/`, `configs/`.
- `requirements.txt`, `README.md`.
- `run_smoke_jax.sh` — JAX-only smoke-test SLURM script (passed: job 4889583).
- `run_smoke_jax2.sh` — full minimal-stack smoke test (passed: job 4890090).
- `HANDOFF_PT_SETUP.md` — this file.

> ⚠️ This file and `run_smoke_jax.sh` were both untracked when this session
> began and were observed to disappear from disk between session messages
> on 2026-05-06 (not in any commit, not stashed, not in user-site). Both
> were recreated from session context. Consider `git add` + a small commit
> on a feature branch to keep them durable.

---

## 5. Conda env (DONE)

**Env name:** `pt`
**Path:** `/scratch/marzii/envs/pt`
**Python:** 3.10.20
**Status:** smoke test #1 (job 4889583) and smoke test #2 (job 4890090) both PASSED on L40S.

User chose **minimal install path** (LunarLander only) on 2026-05-06 round 2 —
**not** "option 1 = install everything". MuJoCo / d4rl / robosuite / robomimic
are intentionally **skipped**.

### Final installed stack
- `jax==0.4.28+computecanada`, `jaxlib==0.4.28+cuda12.cudnn89.computecanada`
  (CC wheels — vanilla PyPI does not host the `+cuda12.cudnn89` build).
- `flax==0.8.5`  ← **NOT 0.7.5**. flax 0.7.5 calls
  `jax.config.define_bool_state(...)` as a `Config` instance method, but
  jax 0.4.27+ moved that to a module-level function. flax 0.8.5 uses the
  new API and imports cleanly under jax 0.4.28; the upstream PT code uses
  only stable `flax.linen` / `flax.training.train_state.TrainState` /
  `flax.training.early_stopping.EarlyStopping` APIs.
- `optax==0.1.7`, `distrax==0.1.5`, `chex==0.1.86` (May-2024 era,
  jax-0.4.28-compatible).
- `orbax-checkpoint==0.5.20`, `tensorflow-probability==0.24.0`,
  `nest_asyncio==1.6.0` (required by orbax-checkpoint).
- `transformers==5.8.0`, `ml_collections==1.1.0`, `absl-py==2.4.0`,
  `tensorboardX==2.6.5`, `pandas==2.3.3`.
- `gym==0.23.1` (intentionally pinned to old gym, matches upstream PT).
- `box2d-py==2.3.8`, `pygame==2.6.1` (gym 0.23.1 imports pygame eagerly
  even when not rendering), `h5py==3.16.0`.
- `Pillow==12.2.0`, `ujson==5.12.1` (transitive — `JaxPref/reward_transform.py`).
- `swig==4.4.1` (conda-forge — needed at install time to compile
  box2d-py from sdist; kept in env for future rebuilds).
- CC wheels already present from initial install: `imageio`,
  `imageio_ffmpeg`, `tqdm`, `numpy==1.26.4` (vanilla, see hack #1 below),
  `scipy==1.15.2` (vanilla).

### Skipped / deferred
- **`wandb`** — latest wandb builds a Go binary at install time, which
  OOMs the login node (`runtime: failed to create new OS thread`). Re-add
  with `pip install "wandb<0.17"` (pure-Python era) when needed.
- **MuJoCo / `d4rl` / robosuite / robomimic** — out of minimal scope. Add
  later only if running upstream D4RL/Robosuite tasks for paper comparison.
  Recipe: see §7 below.

### Critical environment hacks (DO NOT TRIP OVER THESE)

1. **`/scratch/marzii/envs/pt/lib/python3.10/site-packages/_manylinux.py`** —
   our shim with `manylinux*_compatible = True`. Compute Canada's Python
   ships `/cvmfs/.../site-packages/_manylinux.py` that returns `False`,
   forcing pip to ignore every PyPI manylinux wheel and rebuild from sdist
   (which trips bazel/Go OOM on login nodes). Our shim wins **only when
   `PYTHONPATH` is unset**, since CC's `PYTHONPATH` puts their
   site-packages at `sys.path[1]`.

2. **Two install regimes — pick one based on what wheel you want:**
   - **CC wheels** (`+computecanada`, `+cuda12.cudnn89.computecanada`):
     keep CC's `PYTHONPATH` and use
     `PIP_CONFIG_FILE=/cvmfs/soft.computecanada.ca/config/python/pip-x86-64-v4-gentoo2023.conf`.
     CC's wheels carry `cp310-cp310-linux_x86_64` tags (no manylinux) and
     resolve only when CC's `_manylinux.py` is winning.
   - **PyPI wheels** (everything else):
     `env -u PYTHONPATH PIP_CONFIG_FILE=/dev/null /scratch/marzii/envs/pt/bin/pip install ...`

3. **Pin `jax`/`jaxlib` whenever installing other packages** — pip's resolver
   will happily upgrade jax to 0.6.x to satisfy newer chex/orbax/TFP
   requirements, which silently breaks flax 0.8.5 and removes CC's CUDA
   jaxlib. We hit this once on 2026-05-06 and had to roll it back.

4. **In SLURM run scripts always `unset PYTHONPATH`** after
   `module load StdEnv/2023`, otherwise CC site-packages may shadow our
   env at runtime. Reference: `run_smoke_jax2.sh`.

5. **`SDL_VIDEODRIVER=dummy`** must be set before importing gym
   `LunarLanderContinuous-v2` (gym imports pygame eagerly).

### Verification artefacts
- `logs/pt-smoke_4889583.out` — smoke #1 (JAX-only) pass.
- `logs/pt-smoke2_4890090.out` — smoke #2 (full minimal stack) pass:
  GPU JAX, Flax MLP forward+grad on `cuda(id=0)`, LunarLander rollout,
  every `JaxPref/*.py` module imports cleanly.

---

## 6. After the env: the LunarLander work (NEXT)

### Phase A: Generate the offline dataset
Goal: 5 HDF5 files matching D4RL Hopper/Walker2d's schema, with the same
training methodology and performance buckets.

Recommended files to create (all under `/home/marzii/PT/PreferenceTransformer/`):
- `scripts/offline_data/train_sac_for_offline.py` — trains SAC on
  LunarLanderContinuous-v2, saves the full replay buffer at the "medium"
  checkpoint AND at the "expert" checkpoint, plus the actor weights at both
  points. Reuses the user's existing PEBBLE oracle SAC implementation if
  practical; otherwise standalone via `stable_baselines3`.
- `scripts/offline_data/rollout_to_hdf5.py` — loads an actor checkpoint,
  rolls out N steps deterministic, writes a D4RL-format HDF5.
- `scripts/offline_data/replay_to_hdf5.py` — converts a saved replay buffer
  to D4RL-format HDF5.
- `scripts/offline_data/concat_hdf5.py` — concatenates two HDF5 files (for
  medium-expert variant).
- `scripts/offline_data/make_all_variants.sh` — orchestrator that produces
  all 5 files into `$SCRATCH/PT/lunarlander/`.

Output:
```
$SCRATCH/PT/lunarlander/
    lunarlander-random-v2.hdf5        (random uniform actions)
    lunarlander-medium-v2.hdf5        (1M steps from SAC-medium)
    lunarlander-medium-replay-v2.hdf5 (replay buffer at medium)
    lunarlander-medium-expert-v2.hdf5 (500K medium + 500K expert)
    lunarlander-expert-v2.hdf5        (1M steps from SAC-expert)
```

### Phase B: Patch PT for LunarLander
The PT codebase imports `D4RLDataset` (MuJoCo) and `robosuite/robomimic`
(robotics) at the top of `dataset_utils.py` and
`JaxPref/new_preference_reward_main.py`. Since we skipped those heavy deps,
those imports will fail. Recommended approach:

- Add a new dataset class `LunarLanderHDF5Dataset` to `dataset_utils.py`
  (or sibling file) that loads our HDF5 in the same shape as `D4RLDataset`.
- Guard the `D4RLDataset` / `robosuite` imports in `dataset_utils.py` with
  `try/except ImportError` so LunarLander runs do not require MuJoCo.
- Add a switch in `train_offline.py` keyed off `--env_name lunarlander*` so
  the right dataset class is selected.
- Add a `wrappers/lunarlander.py` wrapper if needed for action-space
  normalization (probably not — LunarLanderContinuous already in [-1,1]).
- Add `configs/lunarlander_config.py` (start from `configs/mujoco_config.py`,
  same IQL hyperparameters).

### Phase C: Collect preferences
The `human_label/` dir has pre-collected preferences for D4RL/Robomimic
tasks but obviously nothing for LunarLander. The user already built a web
labeling tool in `BPref3/label_web.py` for PEBBLE; it could be adapted, OR
the user can collect labels through the PT repo's own human-labeling
pipeline (haven't yet read which one is more convenient).

Goal: the same scale as D4RL — about 100 preference queries for
LunarLander-medium, 500 for LunarLander-medium-expert, etc.

### Phase D: Train and evaluate
```bash
# Reward model
python -m JaxPref.new_preference_reward_main \
    --env lunarlander-medium-replay-v2 \
    --num_query 500 --query_len 100 \
    --model_type PrefTransformer \
    --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4

# IQL with PT reward
python train_offline.py \
    --env_name lunarlander-medium-replay-v2 \
    --config configs/lunarlander_config.py \
    --use_reward_model True --model_type PrefTransformer \
    --ckpt_dir <reward_model_path>
```

---

## 7. Open decisions for the user (NOT yet made)

- **Whether MuJoCo / robosuite / robomimic / d4rl are eventually needed.**
  Currently skipped (minimal install path). They become relevant only if
  the user wants to also run the upstream D4RL/Robosuite tasks for direct
  paper reproduction. Recipe (when ready):
  - MuJoCo: `module load mujoco/2.3.6` on Compute Canada (check
    `module spider mujoco`). Set `MUJOCO_GL=osmesa`.
  - `pip install "mujoco-py>=2.1"`.
  - `cd /home/marzii/PT/PreferenceTransformer/d4rl && pip install -e .`.
  - `pip install git+https://github.com/ARISE-Initiative/robosuite.git@v1.3`.
  - `pip install git+https://github.com/ARISE-Initiative/robomimic.git`.
- **Whether to reuse one of the existing oracle PEBBLE runs to source the
  expert checkpoint**, or train a fresh SAC. Reusing is faster but needs a
  PyTorch-checkpoint-to-Flax conversion (PEBBLE is PyTorch). Easier to do
  rollouts in PyTorch, save tuples to HDF5, then read them in JAX-PT.
- **Preference collection tool:** PT's own pipeline vs adapt
  `BPref3/label_web.py`.

---

## 8. Pre-existing memories that constrain this work

In `/home/marzii/.claude/projects/-home-marzii-BPref3/memory/`:
- `feedback_fragile_env.md` — never modify `bpref39`, only `bpref39_clone`.
- `feedback_fragile_irl_env.md` — never modify IRL/IRL2/imitation-GAIL, only IRL3.
- `feedback_output_subdirs.md` — env-specific subfolders (e.g. LunarLander/) in output dirs.
- `feedback_check_after_each_task.md` — ask user to review after each task.
- `project_pt_workspace.md` — PT code/data confined to `/home/marzii/PT/PreferenceTransformer/`.

In `/home/marzii/.claude/projects/-home-marzii-PT-PreferenceTransformer/memory/`
(written 2026-05-06):
- `project_pt_env_setup.md` — exact env state, manylinux/PYTHONPATH hacks,
  pin lessons learned (the flax 0.7.5 trap especially).
- `project_pt_lunarlander_plan.md` — research goal and 5-dataset roadmap.

---

## 9. Resuming this work in a future session

Two ways to pick up:

1. **`claude --resume`** in this directory and pick this conversation from
   the list. Transcript is at
   `/home/marzii/.claude/projects/-home-marzii-PT-PreferenceTransformer/<session-id>.jsonl`.
2. Start a fresh `claude` session and the PT memory directory above will
   load `project_pt_env_setup.md` and `project_pt_lunarlander_plan.md`
   automatically via `MEMORY.md`. Then read this file and you have full
   context.

If `HANDOFF_PT_SETUP.md` or `run_smoke_jax.sh` go missing again, the memory
files contain enough information to recreate them.
