import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Optional, Tuple, Dict, Any
from flax import struct

from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch

# =============================================================================
# PARAMETERS (Added 3 new optional fields)
# =============================================================================

@struct.dataclass
class RNKParams(ExplorationBonusParams):
    """RNK parameters with optional action support."""
    n_features: int = 1024
    length_scale: float = None
    reg: float = 1.
    n_iterations: int = struct.field(pytree_node=False, default=0)
    n_samples: int = struct.field(pytree_node=False, default=256)
    use_effective_dim: bool = struct.field(pytree_node=False, default=True)
    
    # NEW: Optional action parameters (auto-detected by default)
    use_actions: bool = struct.field(pytree_node=False, default=True)
    discrete_actions: bool = struct.field(pytree_node=False, default=None)  # Auto-detected
    num_discrete_actions: int = struct.field(pytree_node=False, default=None)  # Auto-detected

# =============================================================================
# STATE AND BONUS (Unchanged)
# =============================================================================

@struct.dataclass
class RNKState:
    """RNK state."""
    cov_matrix: jnp.ndarray
    precision_matrix: jnp.ndarray
    feature_fn: Callable
    count: jnp.ndarray
    length_scale: jnp.ndarray

@struct.dataclass
class RNKBonus:
    """RNK exploration bonus."""
    state: RNKState
    params: RNKParams

# =============================================================================
# FEATURE FUNCTION (Modified to handle actions)
# =============================================================================

def compute_effective_dimension(X, gamma):
    T = X.shape[0]
    X = X.reshape(T, -1)
    # Covariance matrix C = X^T X + gamma * I
    C = X.T @ X + gamma * jnp.eye(X.shape[1])

    # Eigenvalues in decreasing order
    eigenvals = jnp.sort(jnp.linalg.eigvals(C))[::-1]

    # Lambda_{T,j} = sum of eigenvals[j:] for j=1,2,...,d
    cumsum_reverse = jnp.cumsum(eigenvals[::-1])[::-1]
    Lambda_T = jnp.concatenate([cumsum_reverse[1:], jnp.array([0.0])])

    # Find min j: j*gamma*ln(T) >= Lambda_{T,j}
    j_vals = jnp.arange(1, len(eigenvals) + 1)
    condition = j_vals * gamma * jnp.log(T) >= Lambda_T

    return jnp.argmax(condition) + 1

def _create_random_fourier_features(
    key: jnp.ndarray,
    obs_dim: int,
    action_dim: int,
    n_features: int,
    use_actions: bool = True,
    discrete_actions: bool = False,
    num_discrete_actions: int = None,
) -> Callable:
    """Create random Fourier features with optional action support."""
    
    # Calculate input dimension
    if use_actions:
        if discrete_actions:
            effective_action_dim = num_discrete_actions
        else:
            effective_action_dim = action_dim
        input_dim = obs_dim + effective_action_dim
    else:
        input_dim = obs_dim
    
    # Create random frequencies and phases
    key_freq, key_phase = jax.random.split(key)
    frequencies = jax.random.normal(key_freq, (n_features, input_dim))
    phases = jax.random.uniform(key_phase, (n_features,), maxval=2 * jnp.pi)
    
    def feature_fn(length_scale: jnp.ndarray, observations: jnp.ndarray, actions: jnp.ndarray = None) -> jnp.ndarray:
        """Compute random Fourier features."""
        # Flatten observations
        obs_flat = observations.reshape((observations.shape[0], -1))
        
        # Optionally include actions
        if use_actions and actions is not None:
            if discrete_actions:
                # One-hot encode discrete actions
                actions_flat = actions.reshape(-1)
                actions_onehot = jax.nn.one_hot(actions_flat, num_discrete_actions)
                inputs = jnp.concatenate([obs_flat, actions_onehot], axis=-1)
            else:
                # Use continuous actions
                actions_flat = actions.reshape((actions.shape[0], -1))
                inputs = jnp.concatenate([obs_flat, actions_flat], axis=-1)
        else:
            # Use only observations
            inputs = obs_flat
        
        # Compute features
        projections = jnp.dot(inputs, frequencies.T / length_scale) + phases
        return jnp.sqrt(2.0 / n_features) * jnp.cos(projections)
    
    return feature_fn

# =============================================================================
# INITIALIZATION (Modified to accept action parameters)
# =============================================================================

def init_rnk(key: jnp.ndarray, obs_size: int, action_size: int, params: RNKParams, 
             discrete_actions: bool = False, num_discrete_actions: int = None) -> RNKBonus:
    """Initialize RNK with auto-detected action properties."""
    
    # Use auto-detected discrete info, but allow user override
    if params.use_actions:
        # Use auto-detected values unless user explicitly overrides
        final_discrete = params.discrete_actions if hasattr(params, 'discrete_actions') and params.discrete_actions is not None else discrete_actions
        final_num_discrete = params.num_discrete_actions if hasattr(params, 'num_discrete_actions') and params.num_discrete_actions is not None else num_discrete_actions
        
        # Default fallback for num_discrete_actions
        if final_discrete and final_num_discrete is None:
            final_num_discrete = action_size
    else:
        final_discrete = False
        final_num_discrete = None
    
    # Create feature function
    feature_fn = _create_random_fourier_features(
        key, obs_size, action_size, params.n_features,
        params.use_actions, final_discrete, final_num_discrete
    )

    # Calculate effective input dimension
    if params.use_actions:
        if final_discrete:
            effective_input_dim = obs_size + final_num_discrete
        else:
            effective_input_dim = obs_size + action_size
    else:
        effective_input_dim = obs_size

    # Initialize matrices
    cov_matrix = jnp.eye(params.n_features) * params.n_features
    precision_matrix = jnp.zeros((params.n_features, params.n_features))

    state = RNKState(
        cov_matrix=cov_matrix,
        precision_matrix=precision_matrix,
        feature_fn=jax.tree_util.Partial(feature_fn),
        count=jnp.array(0.0),
        length_scale=params.length_scale if params.length_scale is not None else jnp.sqrt(effective_input_dim),
    )

    return RNKBonus(state=state, params=params)

# =============================================================================
# BONUS COMPUTATION (Modified to optionally use actions)
# =============================================================================

@jax.jit
def compute_rnk_bonus(bonus: RNKBonus, observations: jnp.ndarray, actions: jnp.ndarray = None) -> jnp.ndarray:
    """Compute RNK bonus with optional actions."""
    
    # Compute features
    if bonus.params.use_actions and actions is not None:
        features = bonus.state.feature_fn(bonus.state.length_scale, observations, actions)
    else:
        features = bonus.state.feature_fn(bonus.state.length_scale, observations)

    # Compute uncertainty
    quadratic_form = jnp.einsum('bi,ij,bj->b',
                               features,
                               bonus.state.precision_matrix,
                               features)

    return 0.5 * jnp.log1p(quadratic_form)

# =============================================================================
# UPDATE (Modified to optionally use actions)  
# =============================================================================

def update_rnk(
    bonus: RNKBonus,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[RNKBonus, Dict[str, Any]]:
    """Update RNK with new observations and optionally actions."""
    
    # Flatten observations
    observations = flatten_batch(batch.obs)
    
    # Optionally flatten actions
    if bonus.params.use_actions:
        actions = flatten_batch(batch.action)
    else:
        actions = None

    # Subsample if specified
    if bonus.params.n_samples is not None:
        indices = jax.random.choice(key, observations.shape[0], shape=(bonus.params.n_samples,), replace=False)
        observations = observations[indices]
        if actions is not None:
            actions = actions[indices]

    # Compute features
    if bonus.params.use_actions and actions is not None:
        features = bonus.state.feature_fn(bonus.state.length_scale, observations, actions)
    else:
        features = bonus.state.feature_fn(bonus.state.length_scale, observations)
    
    batch_size = observations.shape[0]

    # Update covariance matrix
    new_count = bonus.state.count + batch_size
    new_cov_matrix = bonus.state.cov_matrix + features.T @ features

    # Regularize and invert
    regularized_cov = (new_cov_matrix + bonus.params.reg * jnp.eye(new_cov_matrix.shape[0]))
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

    # Diagnostics
    inversion_error = jnp.linalg.norm(
        jnp.eye(regularized_cov.shape[0]) - regularized_cov @ new_precision_matrix
    )

    diagnostics = {
        "bonus/error": inversion_error,
    }

    return RNKBonus(state=new_state, params=bonus.params), diagnostics

# =============================================================================
# HELPER FUNCTION (Unchanged)
# =============================================================================

@partial(jax.jit, static_argnums=(2,))
def _update_precision_matrix(
    current_precision: jnp.ndarray,
    regularized_cov: jnp.ndarray,
    n_iterations: int
) -> jnp.ndarray:
    """Update precision matrix using iterative inversion or direct inversion."""
    if n_iterations == 0:
        return jnp.linalg.inv(regularized_cov)

    residual_norm = jnp.linalg.norm(
        jnp.eye(regularized_cov.shape[0]) - regularized_cov @ current_precision
    )

    precision = jax.lax.cond(
        residual_norm >= 1.0,
        lambda: jnp.eye(regularized_cov.shape[0]) * (1.0 / jnp.linalg.norm(regularized_cov)),
        lambda: current_precision,
    )

    def newton_step(precision, _):
        precision = precision @ (2 * jnp.eye(regularized_cov.shape[0]) - regularized_cov @ precision)
        precision = (precision + precision.T) / 2
        return precision, None

    precision, _ = jax.lax.scan(newton_step, precision, None, length=n_iterations)
    return precision