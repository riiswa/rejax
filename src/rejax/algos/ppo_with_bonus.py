"""PPO algorithm with exploration bonuses using dual critics."""

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.core import unfreeze, freeze
from flax.training.train_state import TrainState
from jax import numpy as jnp
from typing import NamedTuple, Any, Optional, Callable

from rejax.algos.algorithm import Algorithm, register_init, INIT_REGISTRATION_KEY
from rejax.algos.exploration.defs import Trajectory, ExplorationBonusParams, flatten_batch
from rejax.algos.exploration.rnd import RNDParams
from rejax.algos.exploration.rnk import RNKParams, compute_effective_dimension
from rejax.algos.exploration.utils import update_bonus, compute_bonus, create_exploration_bonus
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
    RewardRMSState
)
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork

from rejax.algos.exploration.drnd import DRNDParams

from rejax.algos.exploration.hash import HashParams


# -----------------------------------------------------------------------------
# Data structures for dual-reward setting
# -----------------------------------------------------------------------------

class DualAdvantages(NamedTuple):
    """Container for intrinsic and extrinsic advantages."""
    extrinsic: jnp.ndarray
    intrinsic: jnp.ndarray
    extrinsic_targets: jnp.ndarray
    intrinsic_targets: jnp.ndarray


class AdvantageMinibatch(struct.PyTreeNode):
    """Minibatch for PPO updates with dual advantages."""
    trajectories: Trajectory
    extrinsic_advantages: chex.Array
    intrinsic_advantages: chex.Array
    extrinsic_targets: chex.Array
    intrinsic_targets: chex.Array

# -----------------------------------------------------------------------------
# PPO implementation with exploration bonuses
# -----------------------------------------------------------------------------

class PPO(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
    """PPO algorithm with exploration bonuses using dual critics."""

    # Core PPO components
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    extrinsic_critic: nn.Module = struct.field(pytree_node=False, default=None)
    intrinsic_critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)
    anneal_lr: bool = struct.field(pytree_node=False, default=True)

    # Exploration bonus settings
    bonus_type: str = struct.field(pytree_node=False, default="none")
    bonus_params: ExplorationBonusParams = struct.field(pytree_node=True, default=None)

    # Coefficients for mixing intrinsic and extrinsic rewards/advantages
    ext_coef: float = struct.field(pytree_node=True, default=10.)
    int_coef: float = struct.field(pytree_node=True, default=1.)
    int_gamma: float = struct.field(pytree_node=True, default=0.99)  # Separate discount for intrinsic rewards

    num_iterations_obs_norm_init: int = struct.field(pytree_node=False, default=1)

    # Flag for normalizing intrinsic rewards
    normalize_intrinsic_rewards: bool = struct.field(pytree_node=False, default=True)

    logging_callback: Optional[Callable] = struct.field(pytree_node=False, default=None)

    def make_act(self, ts):
        def act(obs, rng):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.actor.apply(ts.actor_ts.params, obs, rng, method="act")
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
        else:
            actor = GaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                **agent_kwargs,
            )

        # Create two separate critics
        extrinsic_critic = VNetwork(**agent_kwargs)

        # Parse exploration bonus parameters
        bonus_type = config.pop("bonus_type", "none")
        bonus_config = config.pop("bonus_params", {})

        intrinsic_critic = None
        if bonus_type != "none":
            intrinsic_critic = VNetwork(**agent_kwargs)

        if bonus_type == "rnd":
            bonus_params = RNDParams(**bonus_config)
        elif bonus_type == "rnk":
            bonus_params = RNKParams(**bonus_config)
        elif bonus_type == "drnd":  # Add DRND support
            bonus_params = DRNDParams(**bonus_config)
        elif bonus_type == "hash":  # Add DRND support
            bonus_params = HashParams(**bonus_config)
        elif bonus_type != "none":
            raise ValueError(f"Unknown exploration bonus type: {bonus_type}")
        else:
            bonus_params = None

        # Get coefficients
        ext_coef = config.pop("ext_coef", 10.0)
        int_coef = config.pop("int_coef", 1.0)
        int_gamma = config.pop("int_gamma", 0.99)

        # Get normalization settings
        normalize_intrinsic_rewards = config.pop("normalize_intrinsic_rewards", True)

        return {
            "actor": actor,
            "extrinsic_critic": extrinsic_critic,
            "intrinsic_critic": intrinsic_critic,
            "bonus_type": bonus_type,
            "bonus_params": bonus_params,
            "ext_coef": ext_coef,
            "int_coef": int_coef,
            "int_gamma": int_gamma,
            "normalize_intrinsic_rewards": normalize_intrinsic_rewards
        }

    def warmup_obs_normalization(self, ts, num_warmup_steps):
        """Warm up observation normalization by collecting observations without training."""

        def warmup_step(ts, unused):
            # Get random action for environment stepping
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample random actions (we don't use policy during warm-up)
            action_space = self.env.action_space(self.env_params)
            if self.discrete:
                action = jax.random.randint(rng_action, (self.num_envs,), 0, action_space.n)
            else:
                low = action_space.low
                high = action_space.high
                action = jax.random.uniform(
                    rng_action, (self.num_envs, *action_space.shape),
                    minval=low, maxval=high
                )

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, _, done, _ = t

            # Update observation normalization statistics
            if self.normalize_observations:
                obs_rms_state, _ = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                ts = ts.replace(obs_rms_state=obs_rms_state)

            return ts, next_obs

        ts, observations = jax.lax.scan(warmup_step, ts, None, num_warmup_steps)
        return ts.obs_rms_state, observations

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_ext_critic, rng_int_critic, rng_bonus = jax.random.split(rng, 5)
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        obs_size = np.prod(self.env.observation_space(self.env_params).shape)

        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        extrinsic_critic_params = self.extrinsic_critic.init(rng_ext_critic, obs_ph)

        if self.anneal_lr:
            num_iterations = self.total_timesteps // (self.num_envs * self.num_steps)
            total_updates = num_iterations * self.num_epochs * self.num_minibatches

            schedule_fn = optax.linear_schedule(
                init_value=self.learning_rate,
                end_value=0.0,  # Decay to exactly zero
                transition_steps=total_updates
            )

            tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(learning_rate=schedule_fn, eps=1e-5)
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(learning_rate=self.learning_rate, eps=1e-5)
            )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        extrinsic_critic_ts = TrainState.create(apply_fn=(), params=extrinsic_critic_params, tx=tx)

        intrinsic_critic_params = None
        intrinsic_critic_ts = None
        if self.bonus_type != "none" and self.intrinsic_critic is not None:
            intrinsic_critic_params = self.intrinsic_critic.init(rng_int_critic, obs_ph)
            intrinsic_critic_ts = TrainState.create(
                apply_fn=(),
                params=intrinsic_critic_params,
                tx=tx
            )

        # Initialize exploration bonus if specified
        exploration_bonus = None
        if self.bonus_type != "none" and self.bonus_params is not None:
            exploration_bonus = create_exploration_bonus(
                self.bonus_type, rng_bonus, obs_size, self.bonus_params
            )

        return {
            "actor_ts": actor_ts,
            "extrinsic_critic_ts": extrinsic_critic_ts,
            "intrinsic_critic_ts": intrinsic_critic_ts,
            "exploration_bonus": exploration_bonus
        }

    @register_init
    def initialize_intrinsic_reward_rms_state(self, rng):
        """Initialize running statistics for intrinsic rewards."""
        batch_size = getattr(self, "num_envs", ())
        return {"intrinsic_rew_rms_state": RewardRMSState.create(batch_size)}

    def print_progress(self, timesteps):
        if timesteps % (10 * self.num_envs * self.num_steps) == 0:
            progress = (timesteps / self.total_timesteps) * 100

            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Print with carriage return to stay on same line, add padding to clear previous text
            print(f"\r🚀 [{bar}] {progress:5.1f}% | {timesteps:,}/{self.total_timesteps:,} timesteps    ", end='', flush=True)

    def train_iteration(self, ts):
        jax.experimental.io_callback(self.print_progress, (), ts.global_step)

        ts, trajectories = self.collect_trajectories(ts)

        # Get the last values from both critics
        last_ext_val = self.extrinsic_critic.apply(ts.extrinsic_critic_ts.params, ts.last_obs)
        last_ext_val = jnp.where(ts.last_done, 0, last_ext_val)

        last_int_val = jnp.zeros_like(last_ext_val)  # Default value
        if self.bonus_type != "none" and self.intrinsic_critic is not None:
            last_int_val = self.intrinsic_critic.apply(ts.intrinsic_critic_ts.params, ts.last_obs)
            last_int_val = jnp.where(ts.last_done, 0, last_int_val)

        # Calculate advantages for both critics
        advantages = self.calculate_dual_gae(
            trajectories, last_ext_val, last_int_val
        )

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(
                trajectories,
                advantages.extrinsic,
                advantages.intrinsic,
                advantages.extrinsic_targets,
                advantages.intrinsic_targets
            )
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)

        # Update exploration bonus if present
        if hasattr(ts, "exploration_bonus") and ts.exploration_bonus is not None:
            rng, bonus_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)

            metrics = {}

            if self.bonus_type != "none":
                new_bonus, metrics = update_bonus(self.bonus_type, ts.exploration_bonus, trajectories, bonus_rng)
                ts = ts.replace(exploration_bonus=new_bonus)

            if self.logging_callback is not None:
                jax.experimental.io_callback(
                    self.logging_callback,
                    (),  # result_shape_dtypes (wandb.log returns None)
                    ts.global_step,
                    metrics,
                    ts.seed
                )

        return ts

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample action
            unclipped_action, log_prob = self.actor.apply(
                ts.actor_ts.params, ts.last_obs, rng_action, method="action_log_prob"
            )

            # Get values from both critics
            extrinsic_value = self.extrinsic_critic.apply(ts.extrinsic_critic_ts.params, ts.last_obs)
            intrinsic_value = jnp.zeros_like(extrinsic_value)  # Default value
            if self.bonus_type != "none" and self.intrinsic_critic is not None:
                intrinsic_value = self.intrinsic_critic.apply(ts.intrinsic_critic_ts.params, ts.last_obs)

            # Clip action
            if self.discrete:
                action = unclipped_action
            else:
                low = self.env.action_space(self.env_params).low
                high = self.env.action_space(self.env_params).high
                action = jnp.clip(unclipped_action, low, high)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, extrinsic_reward, done, _ = t

            # Initialize intrinsic reward to zeros
            intrinsic_reward = jnp.zeros_like(extrinsic_reward)

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                ts = ts.replace(obs_rms_state=obs_rms_state)

            # Normalize extrinsic rewards if needed
            if self.normalize_rewards:
                rew_rms_state, extrinsic_reward = self.update_and_normalize_rew(
                    ts.rew_rms_state, extrinsic_reward, done
                )
                ts = ts.replace(rew_rms_state=rew_rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs,
                unclipped_action,
                log_prob,
                extrinsic_reward,
                intrinsic_reward,
                extrinsic_value,
                intrinsic_value,
                done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)

        # Calculate intrinsic reward (exploration bonus)
        if hasattr(ts, "exploration_bonus") and ts.exploration_bonus is not None:
            # Compute bonus
            trajectories = trajectories.replace(
                intrinsic_reward=compute_bonus(self.bonus_type, ts.exploration_bonus, trajectories.obs, trajectories.action)
            )

        if self.logging_callback is not None:
            jax.experimental.io_callback(
                self.logging_callback,
                (),  # result_shape_dtypes (wandb.log returns None)
                ts.global_step,
                {"bonus/mean": trajectories.intrinsic_reward.mean(), "bonus/max": trajectories.intrinsic_reward.max(), "bonus/std":trajectories.intrinsic_reward.std()},
                ts.seed
            )

        if hasattr(ts, "exploration_bonus") and ts.exploration_bonus is not None and self.normalize_intrinsic_rewards:
            intrinsic_rew_rms_state, intrinsic_reward = self.update_and_normalize_rew(
             ts.intrinsic_rew_rms_state,
             flatten_batch(trajectories.intrinsic_reward),
             None,
             update_return=False
            )
            trajectories = trajectories.replace(intrinsic_reward=intrinsic_reward.reshape(trajectories.intrinsic_reward.shape[0], trajectories.intrinsic_reward.shape[1], *trajectories.intrinsic_reward.shape[2:]))
            ts = ts.replace(intrinsic_rew_rms_state=intrinsic_rew_rms_state)

        return ts, trajectories

    def calculate_dual_gae(self, trajectories, last_ext_val, last_int_val):
        """Calculate GAE separately for intrinsic and extrinsic rewards."""

        def get_extrinsic_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.extrinsic_reward.squeeze()
                + self.gamma * next_value * (1 - transition.done)
                - transition.extrinsic_value
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.extrinsic_value), advantage

        def get_intrinsic_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.intrinsic_reward.squeeze()
                + self.int_gamma * next_value * (1 - transition.done)
                - transition.intrinsic_value
            )
            advantage = (
                delta + self.int_gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.intrinsic_value), advantage

        # Calculate extrinsic advantages
        _, ext_advantages = jax.lax.scan(
            get_extrinsic_advantages,
            (jnp.zeros_like(last_ext_val), last_ext_val),
            trajectories,
            reverse=True,
        )
        ext_targets = ext_advantages + trajectories.extrinsic_value

        # Calculate intrinsic advantages conditionally
        int_advantages = jnp.zeros_like(ext_advantages)
        int_targets = jnp.zeros_like(ext_targets)
        if self.bonus_type != "none" and self.intrinsic_critic is not None:
            # Calculate intrinsic advantages as before
            _, int_advantages = jax.lax.scan(
                get_intrinsic_advantages,
                (jnp.zeros_like(last_int_val), last_int_val),
                trajectories,
                reverse=True,
            )
            int_targets = int_advantages + trajectories.intrinsic_value

        return DualAdvantages(
            ext_advantages,
            int_advantages,
            ext_targets,
            int_targets
        )

    def update_actor(self, ts, batch):
        def actor_loss_fn(params):
            log_prob, entropy = self.actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss using weighted combination of advantages
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)

            # Normalize advantages separately
            ext_advantages = (batch.extrinsic_advantages - batch.extrinsic_advantages.mean()) / (
                batch.extrinsic_advantages.std() + 1e-8
            )
            int_advantages = (batch.intrinsic_advantages - batch.intrinsic_advantages.mean()) / (
                batch.intrinsic_advantages.std() + 1e-8
            )

            # Combine advantages with coefficients - handle case with no exploration
            if self.bonus_type == "none":
                # Only use extrinsic advantages
                combined_advantages = ext_advantages
            else:
                # Use both intrinsic and extrinsic advantages
                combined_advantages = self.ext_coef * ext_advantages + self.int_coef * int_advantages

            # Calculate clipped objective
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * combined_advantages
            pi_loss2 = clipped_ratio * combined_advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()

            return pi_loss - self.ent_coef * entropy

        loss, grads = jax.value_and_grad(actor_loss_fn)(ts.actor_ts.params)
        return ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads)), loss

    def update_extrinsic_critic(self, ts, batch):
        def critic_loss_fn(params):
            value = self.extrinsic_critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.extrinsic_value + (
                value - batch.trajectories.extrinsic_value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.extrinsic_targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.extrinsic_targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        loss, grads = jax.value_and_grad(critic_loss_fn)(ts.extrinsic_critic_ts.params)
        return ts.replace(extrinsic_critic_ts=ts.extrinsic_critic_ts.apply_gradients(grads=grads)), loss

    def update_intrinsic_critic(self, ts, batch):
        def critic_loss_fn(params):
            value = self.intrinsic_critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.intrinsic_value + (
                value - batch.trajectories.intrinsic_value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.intrinsic_targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.intrinsic_targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        loss, grads = jax.value_and_grad(critic_loss_fn)(ts.intrinsic_critic_ts.params)
        return ts.replace(intrinsic_critic_ts=ts.intrinsic_critic_ts.apply_gradients(grads=grads)), loss

    def update(self, ts, batch):
        ts, actor_loss = self.update_actor(ts, batch)
        ts, extrinsic_critic_loss = self.update_extrinsic_critic(ts, batch)
        # Update intrinsic critic conditionally
        intrinsic_critic_loss = 0.0
        if self.bonus_type != "none" and hasattr(ts, "intrinsic_critic_ts") and ts.intrinsic_critic_ts is not None:
            ts, intrinsic_critic_loss = self.update_intrinsic_critic(ts, batch)

        if self.logging_callback is not None:
            jax.experimental.io_callback(
                self.logging_callback,
                (),  # result_shape_dtypes (wandb.log returns None)
                ts.global_step,
                {"loss/actor": actor_loss, "loss/extrinsic": extrinsic_critic_loss, "loss/intrinsic": intrinsic_critic_loss},
                ts.seed
            )

        return ts

    # def init_state(self, rng: chex.PRNGKey, seed = None) -> Any:
    #     state_values = {}
    #     for name in dir(self):
    #         func = getattr(self, name)
    #         if getattr(func, INIT_REGISTRATION_KEY, False):
    #             rng, rng_init = jax.random.split(rng, 2)
    #             state_values.update(func(rng_init))
    #
    #     cls_name = f"{self.__class__.__name__}State"
    #     state = {k: struct.field(pytree_node=True) for k in state_values.keys()}
    #     print(state)
    #     state_hints = {k: type(v) for k, v in state_values.items()}
    #     d = {**state, "__annotations__": state_hints}
    #     clz = type(cls_name, (struct.PyTreeNode,), d)
    #     return clz(**state_values)

    def init_state(self, rng: chex.PRNGKey) -> Any:
        state_values = {}
        for name in dir(self):
            func = getattr(self, name)
            if getattr(func, INIT_REGISTRATION_KEY, False):
                rng, rng_init = jax.random.split(rng, 2)
                state_values.update(func(rng_init))

        state_values["seed"] = jnp.zeros((1,))

        cls_name = f"{self.__class__.__name__}State"
        state = {k: struct.field(pytree_node=True) for k in state_values.keys()}
        state_hints = {k: type(v) for k, v in state_values.items()}
        d = {**state, "__annotations__": state_hints}
        clz = type(cls_name, (struct.PyTreeNode,), d)
        return clz(**state_values)

    def train_with_seed(self, seed):
        rng = jax.random.PRNGKey(seed)

        ts = self.init_state(rng)

        ts = ts.replace(seed=seed)

        if self.normalize_observations and self.num_iterations_obs_norm_init > 0:
            obs_rms_state, observations = self.warmup_obs_normalization(ts, self.num_steps * self.num_iterations_obs_norm_init)
            ts = ts.replace(obs_rms_state=obs_rms_state)
            if self.bonus_type == "rnk" and self.bonus_params.length_scale is None and self.bonus_params.use_effective_dim:
                X = self.normalize_obs(obs_rms_state, observations).reshape(
                    (observations.shape[0] * observations.shape[1], *observations.shape[2:])
                )
                ts = ts.replace(
                    exploration_bonus=ts.exploration_bonus.replace(
                        state=ts.exploration_bonus.state.replace(
                            length_scale=jnp.sqrt(compute_effective_dimension(X, self.bonus_params.reg))
                        )
                    )
                )

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts, unused):
            # Run a few training iterations
            iteration_steps = self.num_envs * self.num_steps
            num_iterations = np.ceil(self.eval_freq / iteration_steps).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng)

        num_evals = np.ceil(self.total_timesteps / self.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(eval_iteration, ts, None, num_evals)

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation

if __name__ == "__main__":
    from aim import Run

    config = {
        "env": "custom/pointmaze-large-v0",
        "bonus_type": "hash",
        "normalize_observations": True,
        "normalize_intrinsic_rewards": True,
        "total_timesteps": 1_000_000,
        "anneal_lr": False,
        "eval_freq": 8192,
        "num_envs": 32,
        "num_steps": 128,
        "num_epochs": 4,
        "num_minibatches": 32,
        "learning_rate": 0.0003,
        "max_grad_norm": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "ext_coef": 2.,
    }

    run = Run()
    run["hparams"] = config

    ppo = PPO.create(**config)
    eval_callback = ppo.eval_callback

    def log(step, data, seed):
        step = step.item()
        seed = seed.item()
        for k, v in data.items():
            run.track(v, name=k, step=step, context={"seed": seed})

    def logging_callback(ppo, train_state, rng):
        lengths, returns = eval_callback(ppo, train_state, rng)
        jax.experimental.io_callback(
            log,
            (),
            train_state.global_step,
            {"episode_length": lengths.mean(), "return": returns.mean()},
            train_state.seed
        )

        return lengths, returns


    ppo = ppo.replace(eval_callback=logging_callback)

    rng = jax.random.PRNGKey(0)

    train_fn = jax.jit(ppo.train_with_seed)
    vmapped_train_fn = jax.vmap(train_fn)

    n_seeds = 1

    train_state, (episode_lengths, returns) = vmapped_train_fn(jnp.arange(n_seeds))
