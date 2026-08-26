#!/bin/bash
#SBATCH --job-name=pt-smoke2
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke/%x_%j.out
#SBATCH --error=logs/smoke/%x_%j.err

# Smoke test #2: verifies the full PT minimal stack on a GPU node.
# Imports flax/optax/distrax/transformers/gym, runs a small Flax MLP on the
# L40S, exercises a LunarLanderContinuous-v2 rollout, and imports every
# JaxPref module. If this passes, we are ready to start §6 (LunarLander
# dataset generation).

mkdir -p logs/smoke

module --force purge
module load StdEnv/2023

eval "$(/scratch/marzii/miniforge3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/pt
echo "Activated pt env at $(which python)"
python --version

# Drop CC's site-packages from PYTHONPATH so our env's packages take
# precedence (mirrors the install-time configuration that produced this env).
unset PYTHONPATH
export SDL_VIDEODRIVER=dummy

echo ""
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1 | head -3

echo ""
echo "=== Smoke test #2 ==="
cd /home/marzii/PT/PreferenceTransformer
python -u <<'PY'
import sys
sys.path.insert(0, '/home/marzii/PT/PreferenceTransformer')

import jax, jaxlib, flax, optax, distrax, chex
import orbax.checkpoint as ocp
import tensorflow_probability as tfp
import transformers, ml_collections, tensorboardX
import gym, h5py, pygame, Box2D
import numpy as np

print("---- versions ----")
for name, mod in [("jax", jax), ("jaxlib", jaxlib), ("flax", flax),
                  ("optax", optax), ("distrax", distrax), ("chex", chex),
                  ("orbax-checkpoint", ocp), ("tfp", tfp),
                  ("transformers", transformers), ("ml_collections", ml_collections),
                  ("tensorboardX", tensorboardX), ("gym", gym),
                  ("h5py", h5py), ("pygame", pygame), ("Box2D", Box2D),
                  ("numpy", np)]:
    print(f"  {name:20s} {mod.__version__}")

print("\n---- jax devices ----")
print("  default backend:", jax.default_backend())
print("  devices        :", jax.devices())
assert jax.default_backend() == "gpu", "expected gpu backend"

print("\n---- LunarLanderContinuous-v2 rollout ----")
env = gym.make("LunarLanderContinuous-v2")
print("  obs space:", env.observation_space, "act space:", env.action_space)
obs = env.reset()
total_r, steps = 0.0, 0
for _ in range(200):
    a = env.action_space.sample()
    nxt, r, done, _ = env.step(a)
    total_r += float(r); steps += 1
    if done:
        env.reset()
env.close()
print(f"  {steps} random steps, total reward {total_r:.1f}")

print("\n---- Flax MLP forward pass on GPU ----")
import jax.numpy as jnp
import flax.linen as nn

class TinyMLP(nn.Module):
    hidden: int = 64
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)

key = jax.random.PRNGKey(0)
mlp = TinyMLP()
x = jax.random.normal(key, (32, 8))
params = mlp.init(key, x)
y = mlp.apply(params, x)
y.block_until_ready()
print("  forward shape:", y.shape, "device:", y.devices())

print("\n---- Tiny optax + grad step (sanity for the IQL/PT loss path) ----")
import optax as ox
def loss_fn(params, x):
    return jnp.mean(mlp.apply(params, x) ** 2)
grads = jax.grad(loss_fn)(params, x)
opt = ox.adam(1e-3)
opt_state = opt.init(params)
updates, opt_state = opt.update(grads, opt_state)
new_params = optax.apply_updates(params, updates)
print("  grad+optax step OK")

print("\n---- JaxPref module imports ----")
from JaxPref import (
    model, PrefTransformer, NMR, MR,
    jax_utils, sampler, reward_transform,
)
print("  imported model, PrefTransformer, NMR, MR, jax_utils, sampler, reward_transform")

print("\nsmoke test #2 PASSED")
PY

echo ""
echo "smoke test #2 done"
