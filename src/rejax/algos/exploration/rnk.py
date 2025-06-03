"""Random Network for Knowledge (RNK) exploration bonus implementation."""
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct

from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch


# -----------------------------------------------------------------------------
# RNK Parameters
# -----------------------------------------------------------------------------

@struct.dataclass
class RNKParams(ExplorationBonusParams):
    """Parameters for Random Network for Knowledge exploration bonus.

    Args:
        n_features: Number of random Fourier features to use
        length_scale: Length scale for RBF kernel (smaller = more variation)
        reg: Regularization parameter for covariance matrix
        n_iterations: Number of iterations for iterative matrix inversion
    """
    n_features: int = 1000
    length_scale: float = None
    reg: float = 1e-3
    n_iterations: int = struct.field(pytree_node=False, default=20)
    n_samples: int = struct.field(pytree_node=False, default=128)

# -----------------------------------------------------------------------------
# RNK State
# -----------------------------------------------------------------------------

@struct.dataclass
class RNKState:
    """State for Random Network for Knowledge exploration bonus.

    Args:
        cov_matrix: Covariance matrix Φᵀ Φ + regularization
        precision_matrix: Inverse of regularized covariance matrix
        feature_fn: Function to compute random Fourier features
        count: Number of observations seen
    """
    cov_matrix: jnp.ndarray
    precision_matrix: jnp.ndarray
    feature_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    count: jnp.ndarray
    length_scale: jnp.ndarray

# -----------------------------------------------------------------------------
# RNK Bonus
# -----------------------------------------------------------------------------

@struct.dataclass
class RNKBonus:
    """RNK exploration bonus container.

    Args:
        state: Current RNK state
        params: RNK hyperparameters
    """
    state: RNKState
    params: RNKParams


def _create_random_fourier_features(
    key: jnp.ndarray,
    input_dim: int,
    n_features: int,
    length_scale: float
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Create random Fourier feature transformation function.

    Args:
        key: Random key for sampling frequencies and phases
        input_dim: Dimensionality of input observations
        n_features: Number of random features to generate
        length_scale: Length scale parameter for RBF kernel

    Returns:
        Function that transforms input to random Fourier features
    """
    key_freq, key_phase = jax.random.split(key)

    # Sample frequencies from N(0, (1/length_scale²)I)
    frequencies = jax.random.normal(
        key_freq, (n_features, input_dim)
    )

    # Sample phases uniformly from [0, 2π]
    phases = jax.random.uniform(
        key_phase, (n_features,), maxval=2 * jnp.pi
    )

    def feature_fn(length_scale, x: jnp.ndarray) -> jnp.ndarray:
        """Transform input to random Fourier features.

        Args:
            x: Input observations of shape (..., input_dim)

        Returns:
            Random features of shape (..., n_features)
        """
        x = x.reshape((x.shape[0], -1))
        projections = jnp.dot(x, frequencies.T / length_scale) + phases
        normalization = jnp.sqrt(2.0 / n_features)
        return normalization * jnp.cos(projections)

    return feature_fn


def init_rnk(key: jnp.ndarray, obs_size: int, params: RNKParams) -> RNKBonus:
    """Initialize RNK exploration bonus.

    Args:
        key: Random key for initialization
        obs_size: Dimensionality of observations
        params: RNK hyperparameters

    Returns:
        Initialized RNK bonus object
    """
    feature_fn = _create_random_fourier_features(
        key, obs_size, params.n_features, params.length_scale
    )

    # Initialize covariance as identity matrix
    cov_matrix = jnp.eye(params.n_features)

    # Initialize precision matrix as regularized inverse
    precision_matrix = jnp.zeros((params.n_features, params.n_features)) / params.reg

    state = RNKState(
        cov_matrix=cov_matrix,
        precision_matrix=precision_matrix,
        feature_fn=jax.tree_util.Partial(feature_fn),
        count=jnp.array(0.0),
        length_scale=params.length_scale if params.length_scale is not None else jnp.sqrt(obs_size),
    )

    return RNKBonus(state=state, params=params)


@jax.jit
def quad(x, A):
    return x.T @ A @ x


@jax.jit
def compute_rnk_bonus(bonus: RNKBonus, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Compute RNK exploration bonus for given observations.

    Args:
        bonus: Current RNK bonus state
        observations: Batch of observations

    Returns:
        Exploration bonus values for each observation
    """
    features = bonus.state.feature_fn(bonus.state.length_scale, observations)

    #quadratic_form = jax.vmap(quad, in_axes=(0, None))(features, bonus.state.precision_matrix)

    # Compute uncertainty: 0.5 * log(1 + φᵀ Σ⁻¹ φ)
    quadratic_form = jnp.einsum('bi,ij,bj->b',
                               features,
                               bonus.state.precision_matrix,
                               features)

    return 0.5 * jnp.log(1.0 + quadratic_form)


@partial(jax.jit, static_argnums=(2,))
def _update_precision_matrix(
    current_precision: jnp.ndarray,
    regularized_cov: jnp.ndarray,
    n_iterations: int
) -> jnp.ndarray:
    """Update precision matrix using iterative inversion or direct inversion.

    Args:
        current_precision: Current precision matrix estimate
        regularized_cov: New regularized covariance matrix
        n_iterations: Number of iterations for iterative method (0 = direct)

    Returns:
        Updated precision matrix
    """
    if n_iterations == 0:
        # Direct matrix inversion
        return jnp.linalg.inv(regularized_cov)

    # Check if iterative method is stable
    residual_norm = jnp.linalg.norm(
        jnp.eye(regularized_cov.shape[0]) - regularized_cov @ current_precision
    )

    precision = jax.lax.cond(
        residual_norm >= 1.0,
        lambda: jnp.eye(regularized_cov.shape[0]) * (1.0 / jnp.linalg.norm(regularized_cov)),
        lambda: current_precision,
    )

    # Iterative refinement using Neumann series
    for _ in range(n_iterations):
        # precision = precision @ (2 * jnp.eye(regularized_cov.shape[0]) -
        #                        regularized_cov @ precision)
        precision = precision @ (3 * jnp.eye(regularized_cov.shape[0]) - regularized_cov @ precision @ (
                    3 * jnp.eye(regularized_cov.shape[0]) - regularized_cov @ precision))
        # Ensure symmetry
        precision = (precision + precision.T) / 2

    return precision


def update_rnk(
    bonus: RNKBonus,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[RNKBonus, Dict[str, Any]]:
    """Update RNK state with new observations.

    Args:
        bonus: Current RNK bonus state
        observations: New batch of observations
        key: Random key (unused in current implementation)
        pmap_axis_name: Axis name for parallel mapping (unused)

    Returns:
        Updated RNK bonus and diagnostic information
    """
    observations = flatten_batch(batch.obs)

    if bonus.params.n_samples is not None:
        indices = jax.random.choice(key, observations.shape[0], shape=(bonus.params.n_samples,), replace=False)
        observations = observations[indices]

    # Compute features for new observations
    features = bonus.state.feature_fn(bonus.state.length_scale, observations)
    batch_size = observations.shape[0]

    # Update count
    new_count = bonus.state.count + batch_size

    # Update covariance matrix: Σ = Σ_old + Φᵀ Φ
    new_cov_matrix = bonus.state.cov_matrix + features.T @ features

    # Add regularization scaled by count
    regularized_cov = (new_cov_matrix + bonus.params.reg * jnp.eye(new_cov_matrix.shape[0]))

    # Update precision matrix
    new_precision_matrix = _update_precision_matrix(
        bonus.state.precision_matrix,
        regularized_cov,
        bonus.params.n_iterations
    )

    # Create new state
    new_state = RNKState(
        cov_matrix=new_cov_matrix,
        precision_matrix=new_precision_matrix,
        feature_fn=bonus.state.feature_fn,
        count=new_count,
        length_scale=bonus.state.length_scale,
    )

    # Compute diagnostics
    #condition_number = jnp.linalg.cond(regularized_cov)
    inversion_error = jnp.linalg.norm(
        jnp.eye(regularized_cov.shape[0]) - regularized_cov @ new_precision_matrix
    )

    diagnostics = {
        #"bonus/cond": condition_number,
        "bonus/error": inversion_error,
    }

    return RNKBonus(state=new_state, params=bonus.params), diagnostics