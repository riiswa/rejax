import warnings
from copy import copy
from typing import Optional

from brax.envs import Env as BraxEnv, Env
from brax.envs import create
from brax.envs.wrappers import training
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp

from rejax.compat.wrappers import MilestoneRewardWrapper

from rejax.compat.envs.reacher import Reacher
from rejax.compat.envs.pusher import Pusher

def custom_create(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
  env = env

  if episode_length is not None:
    env = training.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = training.VmapWrapper(env, batch_size)
  if auto_reset:
    env = training.AutoResetWrapper(env)

  return env


def create_brax(env_name, **kwargs):
    if env_name.startswith('sparse-'):
        is_sparse = True
        env_name = env_name.replace('sparse-', '')
        if env_name == "ant":
            milestone_distance = 5.
        elif env_name == "halfcheetah":
            milestone_distance = 10.
        else:
            milestone_distance = 1.
    else:
        is_sparse = False
        milestone_distance = 0.

    if env_name == 'reacher':
        env = custom_create(Reacher(**kwargs))
    elif env_name == 'pusher':
        env = custom_create(Pusher(**kwargs))
    else:
        env = create(env_name, **kwargs)

    if is_sparse:
        env = MilestoneRewardWrapper(env, milestone_distance=milestone_distance)
    env = Brax2GymnaxEnv(env)
    return env, env.default_params


@struct.dataclass
class EnvParams:
    # CAUTION: Passing params with a different value than on init has no effect
    max_steps_in_episode: int = 1000


class Brax2GymnaxEnv(GymnaxEnv):
    def __init__(self, env: BraxEnv):
        self.env = env
        self.max_steps_in_episode = env.episode_length

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    def step_env(self, key, state, action, params):
        state = self.env.step(state, action)
        return state.obs, state, state.reward, state.done.astype(bool), state.info

    def reset_env(self, key, params):
        state = self.env.reset(key)
        return state.obs, state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state):
        return state.done.astype(bool)

    @property
    def name(self):
        return self.env.unwrapped.__class__.__name__

    def action_space(self, params):
        # All brax evironments have action limit of -1 to 1
        return spaces.Box(low=-1, high=1, shape=(self.env.action_size,))

    def observation_space(self, params):
        # All brax evironments have observation limit of -inf to inf
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(self.env.observation_size,)
        )

    @property
    def num_actions(self) -> int:
        return self.env.action_size

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a brax env. "
            "Brax envs throw an error when deepcopying, so a shallow copy is returned."
        )
        return copy(self)
