"""JAX implementation of CartPole Swing-Up environment."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 10.0
    tau: float = 0.02
    upright_threshold_radians: float = 0.2  # Threshold for being "upright" (sparse reward)
    center_threshold: float = 0.5  # Threshold for being "at center" (sparse reward)
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500


class CartPoleSwingUp(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of CartPole Swing-Up environment.

    Key differences from standard CartPole:
    - Pole starts downward (inverted)
    - Sparse reward only when BOTH pole is upright AND cart is centered
    - Episode only ends on position limits or max steps (not angle limits)
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole Swing-Up
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        force = params.force_mag * action - params.force_mag * (1 - action)
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Euler integration
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Sparse reward: only when BOTH pole is upright AND cart is at center
        # Normalize theta to [-pi, pi] for reward calculation
        theta_normalized = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
        is_upright = jnp.abs(theta_normalized) < params.upright_threshold_radians
        is_centered = jnp.abs(x) < params.center_threshold

        # Reward only when both conditions are satisfied
        reward = jnp.where(jnp.logical_and(is_upright, is_centered), 1.0, 0.0)

        # Update state
        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Performs resetting of environment."""
        # Initialize with small random perturbations around the downward position
        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Cart position and velocity start near zero
        x = jax.random.uniform(key1, minval=-0.05, maxval=0.05, shape=())
        x_dot = jax.random.uniform(key2, minval=-0.05, maxval=0.05, shape=())

        # Pole starts downward (theta ≈ π) with small perturbation
        theta = jnp.pi + jax.random.uniform(key3, minval=-0.05, maxval=0.05, shape=())
        theta_dot = jax.random.uniform(key4, minval=-0.05, maxval=0.05, shape=())

        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Only terminate on position limits or max steps (no angle limits!)
        done_position = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_position, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole-SwingUp"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,  # No theta limit for swing-up
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,  # No theta limit for swing-up
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )