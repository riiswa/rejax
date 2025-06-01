import chex
from flax import struct


class Trajectory(struct.PyTreeNode):
    """Trajectory with separate intrinsic and extrinsic components."""
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    extrinsic_reward: chex.Array  # Environment reward
    intrinsic_reward: chex.Array  # Exploration bonus reward
    extrinsic_value: chex.Array   # Extrinsic critic's value
    intrinsic_value: chex.Array   # Intrinsic critic's value
    done: chex.Array

@struct.dataclass
class ExplorationBonusParams:
    """Base parameters for exploration bonuses."""
    reward_scale: float = 0.5
    normalize_bonus: bool = False

def flatten_batch(x):
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])