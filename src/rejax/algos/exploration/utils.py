"""Utilities for exploration bonuses in reinforcement learning."""
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp

from rejax.algos.exploration.defs import Trajectory
from rejax.algos.exploration.rnd import update_rnd, compute_rnd_bonus
from rejax.algos.exploration.rnk import update_rnk, compute_rnk_bonus
from rejax.algos.exploration.drnd import update_drnd, compute_drnd_bonus
from rejax.algos.exploration.hash import update_hash, compute_hash_bonus
from rejax.algos.exploration.vime import update_vime, compute_vime_bonus

# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

def create_exploration_bonus(
    bonus_type: str,
    key: jnp.ndarray,
    obs_size: int,
    action_size: int,
    bonus_params: Any
) -> Optional[Any]:
    """Create an exploration bonus instance of the specified type."""
    if bonus_type == "none" or not bonus_type:
        return None

    if bonus_type == "rnd":
        from rejax.algos.exploration.rnd import init_rnd
        return init_rnd(key, obs_size, bonus_params)
    elif bonus_type == "rnk":
        from rejax.algos.exploration.rnk import init_rnk
        return init_rnk(key, obs_size, bonus_params)
    elif bonus_type == "drnd":
        from rejax.algos.exploration.drnd import init_drnd
        return init_drnd(key, obs_size, bonus_params)
    elif bonus_type == "hash":
        from rejax.algos.exploration.hash import init_hash
        return init_hash(key, obs_size, bonus_params)
    elif bonus_type == "vime":
        from rejax.algos.exploration.vime import init_vime
        return init_vime(key, obs_size, action_size, bonus_params)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")

@partial(jax.jit, static_argnums=0)
def update_bonus(
    bonus_type: str,
    bonus: Any,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
):
    if bonus_type == "rnd":
        return update_rnd(bonus, batch, key)
    elif bonus_type == "rnk":
        return update_rnk(bonus, batch, key)
    elif bonus_type == "drnd":
        return update_drnd(bonus, batch, key)
    elif bonus_type == "hash":
        return update_hash(bonus, batch, key)
    elif bonus_type == "vime":
        return update_vime(bonus, batch, key)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")

@partial(jax.jit, static_argnums=0)
def compute_bonus(
    bonus_type: str,
    bonus: Any,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: Optional[jnp.ndarray] = None,
    key: Optional[jnp.ndarray] = None,
):
    if bonus_type == "rnd":
        return jax.vmap(compute_rnd_bonus, in_axes=(None, 0, 0))(bonus, observations, actions)
    elif bonus_type == "rnk":
        return jax.vmap(compute_rnk_bonus, in_axes=(None, 0, 0))(bonus, observations, actions)
    elif bonus_type == "drnd":
        return jax.vmap(compute_drnd_bonus, in_axes=(None, 0, 0))(bonus, observations, actions)
    elif bonus_type == "hash":
        return jax.vmap(compute_hash_bonus, in_axes=(None, 0, 0))(bonus, observations, actions)
    elif bonus_type == "vime":
        if next_observations is None:
            raise ValueError("VIME requires next_observations")
        if key is None:
            raise ValueError("VIME requires a random key")
        return compute_vime_bonus(bonus, observations, actions, next_observations, key)
    else:
        raise ValueError(f"Unknown exploration bonus type: {bonus_type}")