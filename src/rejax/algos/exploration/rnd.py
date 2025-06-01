"""Random Network Distillation (RND) exploration bonus implementation."""

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax import linen as nn

from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch


# -----------------------------------------------------------------------------
# RND Parameters
# -----------------------------------------------------------------------------

@struct.dataclass
class RNDParams(ExplorationBonusParams):
    """Parameters for Random Network Distillation."""
    embedding_size: int = 256
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)
    bonus_learning_rate: float = 1e-4
    layer_norm: bool = False

# -----------------------------------------------------------------------------
# RND State - Make it a PyTreeNode
# -----------------------------------------------------------------------------

@struct.dataclass
class RNDState:
    """State for Random Network Distillation."""
    rnd_params: Any
    optimizer_state: optax.OptState
    apply_fn: Callable

# -----------------------------------------------------------------------------
# RND Bonus - Make it a PyTreeNode
# -----------------------------------------------------------------------------

@struct.dataclass
class RNDBonus:
    """RND exploration bonus state."""
    state: RNDState
    params: RNDParams

# -----------------------------------------------------------------------------
# RND Network
# -----------------------------------------------------------------------------

class RNDModule(nn.Module):
    """Module for RND with target and predictor networks."""
    embedding_size: int
    hidden_layer_sizes: Tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        # Target network
        target = x
        for i, size in enumerate(self.hidden_layer_sizes):
            target = nn.Dense(size, name=f"target_{i}")(target)
            if self.layer_norm:
                target = nn.LayerNorm(name=f"target_ln_{i}")(target)
            target = nn.relu(target)
        target = nn.Dense(self.embedding_size, name="target_out")(target)
        target = jax.lax.stop_gradient(target)

        # Predictor network
        predictor = x
        for i, size in enumerate(self.hidden_layer_sizes):
            predictor = nn.Dense(size, name=f"predictor_{i}")(predictor)
            if self.layer_norm:
                predictor = nn.LayerNorm(name=f"predictor_ln_{i}")(predictor)
            predictor = nn.relu(predictor)
        predictor = nn.Dense(self.embedding_size, name="predictor_out")(predictor)

        # Compute squared error
        return jnp.square(target - predictor).mean(axis=-1, keepdims=True)

# -----------------------------------------------------------------------------
# RND Implementation - Convert to pure functions
# -----------------------------------------------------------------------------

def init_rnd(key: jnp.ndarray, obs_size: int, params: RNDParams) -> RNDBonus:
    """Initialize RND state."""
    rnd_encoder = RNDModule(
        embedding_size=params.embedding_size,
        hidden_layer_sizes=params.hidden_layer_sizes,
        layer_norm=params.layer_norm
    )

    dummy_obs = jnp.zeros((1, obs_size))
    rnd_params = rnd_encoder.init(key, dummy_obs)

    optimizer = optax.adam(learning_rate=params.bonus_learning_rate)
    optimizer_state = optimizer.init(rnd_params)
    apply_fn = jax.tree_util.Partial(rnd_encoder.apply)

    state = RNDState(
        rnd_params=rnd_params,
        optimizer_state=optimizer_state,
        apply_fn=apply_fn
    )

    return RNDBonus(state=state, params=params)


@jax.jit
def compute_rnd_bonus(bonus: RNDBonus, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Compute RND bonus."""
    bonus = bonus.state.apply_fn(bonus.state.rnd_params, observations)
    return jnp.squeeze(bonus, axis=-1)


def update_rnd(
    bonus: RNDBonus,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[RNDBonus, Dict[str, Any]]:
    """Update RND predictor network."""
    observations = flatten_batch(batch.obs)

    def loss_fn(rnd_params):
        return jnp.mean(bonus.state.apply_fn(rnd_params, observations))

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = value_and_grad_fn(bonus.state.rnd_params)


    optimizer = optax.adam(learning_rate=bonus.params.bonus_learning_rate)
    updates, new_optimizer_state = optimizer.update(
        grads, bonus.state.optimizer_state
    )
    new_rnd_params = optax.apply_updates(bonus.state.rnd_params, updates)

    new_state = RNDState(
        rnd_params=new_rnd_params,
        optimizer_state=new_optimizer_state,
        apply_fn=bonus.state.apply_fn
    )

    return RNDBonus(state=new_state, params=bonus.params), {"bonus/loss": loss}