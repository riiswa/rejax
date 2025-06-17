"""Hash-based exploration bonus implementation using learned autoencoder - simplified version."""

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax import linen as nn

from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch


# -----------------------------------------------------------------------------
# Hash Parameters
# -----------------------------------------------------------------------------

@struct.dataclass
class HashParams(ExplorationBonusParams):
    """Parameters for hash-based exploration with learned autoencoder."""
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)
    code_dim: int = 16  # Binary code dimension
    bonus_learning_rate: float = 1e-4

    @property
    def hash_table_size(self) -> int:
        """Auto-compute hash table size from code dimension."""
        return 2 ** self.code_dim


# -----------------------------------------------------------------------------
# Hash State
# -----------------------------------------------------------------------------

@struct.dataclass
class HashState:
    """State for hash-based exploration."""
    ae_params: Any
    optimizer_state: optax.OptState
    apply_fn: Callable
    hash_counts: jnp.ndarray  # Fixed-size count array [hash_table_size]


# -----------------------------------------------------------------------------
# Hash Bonus
# -----------------------------------------------------------------------------

@struct.dataclass
class HashBonus:
    """Hash exploration bonus state."""
    state: HashState
    params: HashParams


# -----------------------------------------------------------------------------
# Autoencoder Network
# -----------------------------------------------------------------------------

class AutoencoderModule(nn.Module):
    """Simple MLP autoencoder with binary bottleneck layer."""
    hidden_layer_sizes: Tuple[int, ...]
    code_dim: int
    input_dim: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Encoder
        encoded = x
        for i, size in enumerate(self.hidden_layer_sizes):
            encoded = nn.Dense(size, name=f"enc_{i}")(encoded)
            encoded = nn.relu(encoded)

        # Binary bottleneck layer
        code_logits = nn.Dense(self.code_dim, name="code")(encoded)
        code_probs = nn.sigmoid(code_logits)

        # Add uniform noise during training for better binary representation
        if training:
            noise = jax.random.uniform(
                self.make_rng('noise'),
                code_probs.shape,
                minval=-0.3,
                maxval=0.3
            )
            code_probs = code_probs + noise

        # Binary code (rounded for hash computation)
        binary_code = (code_probs > 0.5).astype(jnp.int32)

        # Decoder (use probabilities for gradient flow)
        decoded = code_probs
        for i, size in enumerate(reversed(self.hidden_layer_sizes)):
            decoded = nn.Dense(size, name=f"dec_{i}")(decoded)
            decoded = nn.relu(decoded)

        # Output layer
        reconstructed = nn.Dense(self.input_dim, name="output")(decoded)

        return {
            'reconstructed': reconstructed,
            'binary_code': binary_code,
            'code_probs': code_probs
        }


# -----------------------------------------------------------------------------
# Hash Implementation
# -----------------------------------------------------------------------------

def init_hash(key: jnp.ndarray, obs_size: int, params: HashParams) -> HashBonus:
    """Initialize hash-based exploration state."""
    ae_module = AutoencoderModule(
        hidden_layer_sizes=params.hidden_layer_sizes,
        code_dim=params.code_dim,
        input_dim=obs_size
    )

    dummy_obs = jnp.zeros((1, obs_size))
    ae_params = ae_module.init({'params': key, 'noise': key}, dummy_obs, training=True)

    optimizer = optax.adam(learning_rate=params.bonus_learning_rate)
    optimizer_state = optimizer.init(ae_params)

    apply_fn = jax.tree_util.Partial(ae_module.apply)

    # Initialize fixed-size hash table
    hash_counts = jnp.zeros(params.hash_table_size, dtype=jnp.int32)

    state = HashState(
        ae_params=ae_params,
        optimizer_state=optimizer_state,
        apply_fn=apply_fn,
        hash_counts=hash_counts
    )

    return HashBonus(state=state, params=params)


@jax.jit
def _binary_code_to_hash_index(binary_code: jnp.ndarray, table_size: int) -> jnp.ndarray:
    """Convert binary code to hash table index."""
    # Convert binary code to integer using powers of 2
    # Ensure we handle the shape correctly for both single codes and batches
    if binary_code.ndim == 1:
        # Single binary code
        powers = 2 ** jnp.arange(binary_code.shape[0])
        hash_value = jnp.sum(binary_code * powers)
    else:
        # Batch of binary codes
        powers = 2 ** jnp.arange(binary_code.shape[-1])
        hash_value = jnp.sum(binary_code * powers, axis=-1)

    # Use modulo to fit into hash table and ensure positive
    return jnp.abs(hash_value) % table_size


@jax.jit
def compute_hash_bonus(bonus: HashBonus, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Compute hash-based exploration bonus."""
    # Get binary codes
    outputs = bonus.state.apply_fn(
        bonus.state.ae_params,
        observations,
        training=False,
        rngs={'noise': jax.random.PRNGKey(0)}  # No noise during inference
    )

    binary_codes = outputs['binary_code']

    # Convert to hash indices
    hash_indices = _binary_code_to_hash_index(binary_codes, bonus.params.hash_table_size)

    # Get counts for each hash index
    counts = bonus.state.hash_counts[hash_indices]

    # Compute bonus: 1 / √(count + 1)
    bonuses = 1.0 / jnp.sqrt(counts + 1)

    return bonuses


def update_hash(
    bonus: HashBonus,
    batch: Trajectory,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[HashBonus, Dict[str, Any]]:
    """Update hash-based exploration bonus."""
    observations = flatten_batch(batch.obs)

    if key is None:
        key = jax.random.PRNGKey(0)

    # Update autoencoder
    def loss_fn(ae_params):
        outputs = bonus.state.apply_fn(
            ae_params,
            observations,
            training=True,
            rngs={'noise': key}
        )

        # Reconstruction loss
        recon_loss = jnp.mean(jnp.square(outputs['reconstructed'] - observations))

        # Binary regularization loss (encourage binary values)
        code_probs = outputs['code_probs']
        binary_loss = jnp.mean(jnp.minimum((1 - code_probs)**2, code_probs**2))

        total_loss = recon_loss + 0.1 * binary_loss
        return total_loss

    loss, grads = jax.value_and_grad(loss_fn)(bonus.state.ae_params)

    optimizer = optax.adam(learning_rate=bonus.params.bonus_learning_rate)
    updates, new_optimizer_state = optimizer.update(
        grads, bonus.state.optimizer_state
    )
    new_ae_params = optax.apply_updates(bonus.state.ae_params, updates)

    # Update hash counts
    outputs = bonus.state.apply_fn(
        new_ae_params,
        observations,
        training=False,
        rngs={'noise': jax.random.PRNGKey(0)}
    )

    binary_codes = outputs['binary_code']
    hash_indices = _binary_code_to_hash_index(binary_codes, bonus.params.hash_table_size)

    # Update counts: increment count for each observed hash
    # Use scatter_add to increment counts at hash indices
    new_hash_counts = bonus.state.hash_counts.at[hash_indices].add(1)

    # Create new state
    new_state = HashState(
        ae_params=new_ae_params,
        optimizer_state=new_optimizer_state,
        apply_fn=bonus.state.apply_fn,
        hash_counts=new_hash_counts
    )

    return HashBonus(state=new_state, params=bonus.params), {"bonus/ae_loss": loss}