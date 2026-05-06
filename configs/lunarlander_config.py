import ml_collections


def get_config():
    """IQL hyperparameters for LunarLanderContinuous offline data.

    Mirrors `mujoco_config.py` (the IQL paper's locomotion setting). The
    8-d state and 2-d action space are simpler than MuJoCo Hopper/Walker2d,
    but the 256x256 actor/critic and standard discount/expectile/temperature
    work fine.
    """
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 3.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    return config
