"""Distributional Random Network Distillation (DRND) exploration bonus implementation."""

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax import linen as nn

from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch


# -----------------------------------------------------------------------------
# DRND Parameters
# -----------------------------------------------------------------------------

@struct.dataclass
class DRNDParams(ExplorationBonusParams):
    """Parameters for Distributional Random Network Distillation."""
    embedding_size: int = 256
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)
    bonus_learning_rate: float = 1e-4
    layer_norm: bool = False
    n_target_networks: int = 5  # Number of random target networks
    alpha: float = 0.5  # Mixing coefficient between b1 and b2


# -----------------------------------------------------------------------------
# DRND State
# -----------------------------------------------------------------------------

@struct.dataclass
class DRNDState:
    """State for Distributional Random Network Distillation."""
    drnd_params: Any
    optimizer_state: optax.OptState
    apply_fn: Callable


# -----------------------------------------------------------------------------
# DRND Bonus
# -----------------------------------------------------------------------------

@struct.dataclass
class DRNDBonus:
    """DRND exploration bonus state."""
    state: DRNDState
    params: DRNDParams


# -----------------------------------------------------------------------------
# DRND Network
# -----------------------------------------------------------------------------

class DRNDModule(nn.Module):
    """Module for DRND with multiple target networks and one predictor network."""
    embedding_size: int
    hidden_layer_sizes: Tuple[int, ...]
    n_target_networks: int = 5
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))

        # Create multiple target networks (frozen)
        target_outputs = []
        for i in range(self.n_target_networks):
            target = x
            for j, size in enumerate(self.hidden_layer_sizes):
                target = nn.Dense(size, name=f"target_{i}_{j}")(target)
                if self.layer_norm:
                    target = nn.LayerNorm(name=f"target_{i}_ln_{j}")(target)
                target = nn.relu(target)
            target = nn.Dense(self.embedding_size, name=f"target_{i}_out")(target)
            target = jax.lax.stop_gradient(target)  # Freeze target networks
            target_outputs.append(target)

        target_outputs = jnp.stack(target_outputs, axis=0)  # Shape: (n_targets, batch, embedding)

        # Compute target statistics
        target_mean = jnp.mean(target_outputs, axis=0)  # μ(x)
        target_second_moment = jnp.mean(target_outputs ** 2, axis=0)  # B₂(x)

        # Create predictor network (trainable)
        predictor = x
        for j, size in enumerate(self.hidden_layer_sizes):
            predictor = nn.Dense(size, name=f"predictor_{j}")(predictor)
            if self.layer_norm:
                predictor = nn.LayerNorm(name=f"predictor_ln_{j}")(predictor)
            predictor = nn.relu(predictor)
        predictor = nn.Dense(self.embedding_size, name="predictor_out")(predictor)

        return {
            'predictor_output': predictor,
            'target_mean': target_mean,
            'target_second_moment': target_second_moment,
            'target_outputs': target_outputs
        }


# -----------------------------------------------------------------------------
# DRND Implementation
# -----------------------------------------------------------------------------

def init_drnd(key: jnp.ndarray, obs_size: int, params: DRNDParams) -> DRNDBonus:
    """Initialize DRND state."""
    drnd_module = DRNDModule(
        embedding_size=params.embedding_size,
        hidden_layer_sizes=params.hidden_layer_sizes,
        n_target_networks=params.n_target_networks,
        layer_norm=params.layer_norm
    )

    dummy_obs = jnp.zeros((1, obs_size))
    drnd_params = drnd_module.init(key, dummy_obs)

    # Initialize optimizer for all parameters (but gradients will be stopped for targets)
    optimizer = optax.adam(learning_rate=params.bonus_learning_rate)
    optimizer_state = optimizer.init(drnd_params)

    # Create apply function
    apply_fn = jax.tree_util.Partial(drnd_module.apply)

    state = DRNDState(
        drnd_params=drnd_params,
        optimizer_state=optimizer_state,
        apply_fn=apply_fn
    )

    return DRNDBonus(state=state, params=params)


@jax.jit
def compute_drnd_bonus(bonus: DRNDBonus, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Compute DRND bonus for given observations."""
    # Get network outputs
    outputs = bonus.state.apply_fn(bonus.state.drnd_params, observations)

    predictor_output = outputs['predictor_output']  # fθ(x)
    target_mean = outputs['target_mean']  # μ(x)
    target_second_moment = outputs['target_second_moment']  # B₂(x)

    # First bonus: b₁(x) = ||fθ(x) - μ(x)||²
    b1 = jnp.sum((predictor_output - target_mean) ** 2, axis=-1)

    # Second bonus: b₂(x) = √([fθ(x)]² - [μ(x)]²) / (B₂(x) - [μ(x)]²)
    predictor_squared = jnp.sum(predictor_output ** 2, axis=-1, keepdims=True)
    target_mean_squared = jnp.sum(target_mean ** 2, axis=-1, keepdims=True)
    target_second_moment_sum = jnp.sum(target_second_moment, axis=-1, keepdims=True)

    numerator = jnp.maximum(predictor_squared - target_mean_squared, 1e-8)
    denominator = jnp.maximum(target_second_moment_sum - target_mean_squared, 1e-8)

    b2 = jnp.sqrt(numerator / denominator).squeeze(-1)

    # Combined bonus: b(x) = α * b₁(x) + (1-α) * b₂(x)
    combined_bonus = bonus.params.alpha * b1 + (1 - bonus.params.alpha) * b2

    return combined_bonus


def update_drnd(
    bonus: DRNDBonus,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[DRNDBonus, Dict[str, Any]]:
    """Update DRND predictor network."""
    observations = flatten_batch(batch.obs)

    def loss_fn(drnd_params):
        # Get network outputs
        outputs = bonus.state.apply_fn(drnd_params, observations)

        predictor_output = outputs['predictor_output']
        target_outputs = outputs['target_outputs']  # Shape: (n_targets, batch, embedding)

        # Sample from target distribution for each observation
        batch_size = observations.shape[0]

        # Create sampling key if not provided
        if key is None:
            sample_key = jax.random.PRNGKey(0)
        else:
            sample_key = key

        target_indices = jax.random.randint(
            sample_key, (batch_size,), 0, bonus.params.n_target_networks
        )

        # Select random target output for each observation
        sampled_targets = target_outputs[target_indices, jnp.arange(batch_size)]

        # MSE loss between predictor and sampled target
        loss = jnp.mean(jnp.sum((predictor_output - sampled_targets) ** 2, axis=-1))

        return loss

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(bonus.state.drnd_params)

    # Update parameters (gradients are automatically stopped for target networks)
    optimizer = optax.adam(learning_rate=bonus.params.bonus_learning_rate)
    updates, new_optimizer_state = optimizer.update(
        grads, bonus.state.optimizer_state
    )
    new_drnd_params = optax.apply_updates(bonus.state.drnd_params, updates)

    # Create new state
    new_state = DRNDState(
        drnd_params=new_drnd_params,
        optimizer_state=new_optimizer_state,
        apply_fn=bonus.state.apply_fn
    )

    return DRNDBonus(state=new_state, params=bonus.params), {"bonus/loss": loss}