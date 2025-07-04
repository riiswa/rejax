"""VIME (Variational Information Maximizing Exploration) implementation - FIXED."""

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax import linen as nn

from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch


# -----------------------------------------------------------------------------
# VIME Parameters
# -----------------------------------------------------------------------------

@struct.dataclass
class VIMEParams(ExplorationBonusParams):
    """Parameters for VIME exploration bonus.

    Args:
        hidden_layer_sizes: Hidden layer sizes for the dynamics BNN
        learning_rate: Learning rate for BNN updates
        prior_scale: Standard deviation of the weight prior
    """
    hidden_layer_sizes: Tuple[int, ...] = struct.field(pytree_node=False, default=(64, 64))
    learning_rate: float = 1e-3
    prior_scale: float = 1.0


# -----------------------------------------------------------------------------
# Bayesian Neural Network for Dynamics
# -----------------------------------------------------------------------------

class BayesianLinear(nn.Module):
    """Bayesian linear layer with fully factorized Gaussian weights."""
    features: int
    prior_scale: float = 1.0

    @nn.compact
    def __call__(self, x, sample_weights=True):
        # Let Flax infer input size automatically
        input_size = x.shape[-1]

        # Weight parameters: mean and rho (where σ = log(1 + exp(rho)))
        w_mean = self.param('w_mean', nn.initializers.normal(0.1), (input_size, self.features))
        w_rho = self.param('w_rho', nn.initializers.constant(-3.0), (input_size, self.features))

        # Bias parameters
        b_mean = self.param('b_mean', nn.initializers.zeros, (self.features,))
        b_rho = self.param('b_rho', nn.initializers.constant(-3.0), (self.features,))

        if sample_weights:
            # Sample weights from the variational posterior
            w_std = jnp.log(1.0 + jnp.exp(w_rho))
            b_std = jnp.log(1.0 + jnp.exp(b_rho))

            # Use the module's RNG state
            w_eps = jax.random.normal(self.make_rng('weights'), w_mean.shape)
            b_eps = jax.random.normal(self.make_rng('biases'), b_mean.shape)

            w = w_mean + w_std * w_eps
            b = b_mean + b_std * b_eps
        else:
            # Use mean weights (for deterministic forward pass)
            w = w_mean
            b = b_mean

        return x @ w + b


class DynamicsBNN(nn.Module):
    """Bayesian Neural Network for modeling environment dynamics."""
    hidden_layer_sizes: Tuple[int, ...]
    prior_scale: float = 1.0

    @nn.compact
    def __call__(self, state, action, sample_weights=True):
        # Flatten observations and actions
        state = state.reshape((state.shape[0], -1))
        action = action.reshape((action.shape[0], -1))

        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)

        # Store sizes for output layer
        state_size = state.shape[-1]

        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            x = BayesianLinear(hidden_size, self.prior_scale)(x, sample_weights)
            x = jnp.tanh(x)

        # Output layer - predict next state (same size as input state)
        x = BayesianLinear(state_size, self.prior_scale)(x, sample_weights)
        return x


def compute_kl_divergence(params_new, params_old, prior_scale):
    """Compute KL divergence between two BNN parameter distributions."""

    def kl_for_layer(new_params, old_params):
        kl = 0.0

        for param_name in ['w_mean', 'w_rho', 'b_mean', 'b_rho']:
            if param_name in new_params and param_name in old_params:
                if 'mean' in param_name:
                    # For mean parameters
                    mu_new = new_params[param_name]
                    mu_old = old_params[param_name]

                    # Get corresponding rho parameters
                    rho_name = param_name.replace('mean', 'rho')
                    rho_new = new_params[rho_name]
                    rho_old = old_params[rho_name]

                    # Compute standard deviations
                    sigma_new = jnp.log(1.0 + jnp.exp(rho_new))
                    sigma_old = jnp.log(1.0 + jnp.exp(rho_old))

                    # KL divergence between two Gaussians
                    kl_param = (
                        jnp.log(sigma_old / sigma_new) +
                        (sigma_new**2 + (mu_new - mu_old)**2) / (2 * sigma_old**2) -
                        0.5
                    )
                    kl += jnp.sum(kl_param)

        return kl

    # Compute KL for each layer
    total_kl = 0.0

    # Handle nested structure (layers)
    if isinstance(params_new, dict) and isinstance(params_old, dict):
        for key in params_new:
            if key in params_old:
                if isinstance(params_new[key], dict):
                    total_kl += kl_for_layer(params_new[key], params_old[key])

    return total_kl


# -----------------------------------------------------------------------------
# VIME State and Bonus
# -----------------------------------------------------------------------------

@struct.dataclass
class VIMEState:
    """State for VIME exploration bonus."""
    bnn_params: Any
    optimizer_state: optax.OptState
    step_count: int


@struct.dataclass
class VIMEBonus:
    """VIME exploration bonus container."""
    state: VIMEState
    params: VIMEParams
    obs_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)


# -----------------------------------------------------------------------------
# VIME Implementation
# -----------------------------------------------------------------------------

def init_vime(key: jnp.ndarray, obs_size: int, action_size: int, params: VIMEParams) -> VIMEBonus:
    """Initialize VIME exploration bonus."""

    # Create the dynamics BNN
    dynamics_net = DynamicsBNN(
        hidden_layer_sizes=params.hidden_layer_sizes,
        prior_scale=params.prior_scale
    )

    # Initialize with flattened dummy inputs
    # obs_size represents the total flattened observation size
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))

    bnn_params = dynamics_net.init(
        {'params': key, 'weights': key, 'biases': key},
        dummy_obs, dummy_action, sample_weights=True
    )

    # Initialize optimizer
    optimizer = optax.adam(learning_rate=params.learning_rate)
    optimizer_state = optimizer.init(bnn_params)

    state = VIMEState(
        bnn_params=bnn_params,
        optimizer_state=optimizer_state,
        step_count=0
    )

    return VIMEBonus(
        state=state,
        params=params,
        obs_dim=obs_size,
        action_dim=action_size
    )


def compute_vime_bonus(
    bonus: VIMEBonus,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    key: jnp.ndarray
) -> jnp.ndarray:
    """Compute VIME bonus (information gain)."""

    # Store original shape for reshaping output
    original_shape = observations.shape[:2]  # (num_steps, num_envs)

    # Store old parameters
    old_params = bonus.state.bnn_params

    # Create dynamics network (same as initialization)
    dynamics_net = DynamicsBNN(
        hidden_layer_sizes=bonus.params.hidden_layer_sizes,
        prior_scale=bonus.params.prior_scale
    )

    # Use flatten_batch to properly handle batch dimensions
    # This converts (num_steps, num_envs, *obs_shape) -> (batch_size, *obs_shape)
    obs_batch = flatten_batch(observations)  # (4096, 10, 10, 4) for example
    actions_batch = flatten_batch(actions)   # (4096, action_dim)
    next_obs_batch = flatten_batch(next_observations)  # (4096, 10, 10, 4)

    # Now flatten observation dimensions for neural network input
    # (batch_size, *obs_shape) -> (batch_size, obs_flat_dim)
    batch_size = obs_batch.shape[0]
    obs_flat = obs_batch.reshape(batch_size, -1)  # (4096, 400) for (10,10,4) obs
    next_obs_flat = next_obs_batch.reshape(batch_size, -1)
    actions_flat = actions_batch.reshape(batch_size, -1)

    # Define loss function for single step update
    def loss_fn(params):
        # Sample multiple predictions and compute negative log likelihood
        num_samples = 5  # Number of samples for Monte Carlo estimate

        def single_prediction(rng_key):
            pred = dynamics_net.apply(
                params, obs_flat, actions_flat,
                sample_weights=True,
                rngs={'weights': rng_key, 'biases': rng_key}
            )
            return pred

        # Sample predictions
        keys = jax.random.split(key, num_samples)
        predictions = jax.vmap(single_prediction)(keys)

        # Mean prediction
        mean_pred = jnp.mean(predictions, axis=0)

        # Negative log likelihood (reconstruction loss)
        mse_loss = jnp.mean((mean_pred - next_obs_flat) ** 2)

        # KL divergence to prior (regularization)
        kl_loss = 0.0
        for layer_params in jax.tree_leaves(params['params']):
            if isinstance(layer_params, dict):
                for param_name, param_value in layer_params.items():
                    if 'mean' in param_name:
                        # KL to zero-mean prior
                        kl_loss += jnp.sum(param_value ** 2) / (2 * bonus.params.prior_scale ** 2)
                    elif 'rho' in param_name:
                        # KL for variance parameters
                        sigma = jnp.log(1.0 + jnp.exp(param_value))
                        kl_loss += jnp.sum(
                            -param_value + jnp.log(bonus.params.prior_scale) +
                            sigma ** 2 / (2 * bonus.params.prior_scale ** 2)
                        )

        return mse_loss + 0.01 * kl_loss  # Small weight on KL to prior

    # Compute gradients and perform one optimization step
    loss, grads = jax.value_and_grad(loss_fn)(old_params)

    # Apply gradients to get new parameters
    optimizer = optax.adam(learning_rate=bonus.params.learning_rate)
    updates, _ = optimizer.update(grads, bonus.state.optimizer_state)
    new_params = optax.apply_updates(old_params, updates)

    # Compute KL divergence between old and new parameters
    kl_div = compute_kl_divergence(
        new_params['params'],
        old_params['params'],
        bonus.params.prior_scale
    )

    # Return KL divergence as intrinsic reward, reshaped to original batch shape
    bonuses = jnp.full((batch_size,), kl_div)
    return bonuses.reshape(original_shape)


def update_vime(
    bonus: VIMEBonus,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[VIMEBonus, Dict[str, Any]]:
    """Update VIME dynamics model."""

    # Use flatten_batch to properly handle batch dimensions
    observations = flatten_batch(batch.obs)  # (batch_size, *obs_shape)
    actions = flatten_batch(batch.action)    # (batch_size, *action_shape)
    next_observations = jnp.roll(observations, -1, axis=0)  # Next observations

    # Flatten observation dimensions for neural network
    batch_size = observations.shape[0]
    obs_flat = observations.reshape(batch_size, -1)
    actions_flat = actions.reshape(batch_size, -1)
    next_obs_flat = next_observations.reshape(batch_size, -1)

    # Create dynamics network (same as initialization)
    dynamics_net = DynamicsBNN(
        hidden_layer_sizes=bonus.params.hidden_layer_sizes,
        prior_scale=bonus.params.prior_scale
    )

    # Define loss function on full batch
    def loss_fn(params):
        # Sample multiple predictions
        num_samples = 3

        def single_prediction(rng_key):
            pred = dynamics_net.apply(
                params, obs_flat, actions_flat,
                sample_weights=True,
                rngs={'weights': rng_key, 'biases': rng_key}
            )
            return pred

        keys = jax.random.split(key, num_samples)
        predictions = jax.vmap(single_prediction)(keys)
        mean_pred = jnp.mean(predictions, axis=0)

        # Reconstruction loss
        mse_loss = jnp.mean((mean_pred - next_obs_flat) ** 2)
        return mse_loss

    # Compute gradients and update
    loss, grads = jax.value_and_grad(loss_fn)(bonus.state.bnn_params)

    optimizer = optax.adam(learning_rate=bonus.params.learning_rate)
    updates, new_optimizer_state = optimizer.update(
        grads, bonus.state.optimizer_state
    )
    new_bnn_params = optax.apply_updates(bonus.state.bnn_params, updates)

    # Update state
    new_state = VIMEState(
        bnn_params=new_bnn_params,
        optimizer_state=new_optimizer_state,
        step_count=bonus.state.step_count + 1
    )

    new_bonus = VIMEBonus(
        state=new_state,
        params=bonus.params,
        obs_dim=bonus.obs_dim,
        action_dim=bonus.action_dim
    )

    return new_bonus, {"vime/loss": loss}