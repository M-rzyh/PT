#!/bin/bash
#SBATCH --job-name=pt-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke/%x_%j.out
#SBATCH --error=logs/smoke/%x_%j.err

# Smoke test #1: verify JAX detects the GPU and runs a basic computation.
# If this passes, we know the L40S (sm_89) is compatible with our JAX install.
# A more comprehensive smoke test (Flax MLP, LunarLander rollout, full JaxPref
# imports) lives in run_smoke_jax2.sh.

mkdir -p logs/smoke

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
echo "Activated pt env at $(which python)"
python --version

# Drop CC's site-packages from PYTHONPATH so our env's packages take precedence.
unset PYTHONPATH

echo ""
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1 | head -3

echo ""
echo "=== JAX smoke test ==="
python -u <<'PY'
import jax
import jax.numpy as jnp
print("jax.__version__:", jax.__version__)
print("jaxlib.__version__:", jax.lib.__version__)
print("jax.devices():", jax.devices())
print("jax.default_backend():", jax.default_backend())

import time
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4096, 4096))
t0 = time.time()
y = jnp.dot(x, x.T).block_until_ready()
dt = time.time() - t0
print(f"4096x4096 matmul OK in {dt*1000:.1f} ms, sum={float(y.sum()):.2f}")
print(f"y.shape: {y.shape}, y.dtype: {y.dtype}")
PY

echo ""
echo "smoke test done"
