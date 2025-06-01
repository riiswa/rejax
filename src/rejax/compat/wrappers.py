from typing import Callable, Optional
from brax.envs import PipelineEnv, State, Wrapper
import jax
from jax import numpy as jp


class MilestoneRewardWrapper(Wrapper):
    """Wrapper that adds milestone-based rewards to any Brax environment.

    This wrapper gives a reward whenever the agent reaches specified distance
    milestones (e.g., every 1.0 unit of forward movement).
    """

    def __init__(
            self,
            env: PipelineEnv,
            milestone_distance: float = 1.0,
            reward_scale: float = 1.0,
            position_fn: Optional[Callable[[State], jp.ndarray]] = lambda state: state.pipeline_state.x.pos[0, 0],
    ):
        """Initializes the milestone reward wrapper.

        Args:
          env: The environment to wrap.
          milestone_distance: Distance between reward milestones.
          reward_scale: Scale factor for milestone rewards.
          position_fn: Function that extracts position from state.
                       Default extracts x position from first body.
        """
        super().__init__(env)
        self._milestone_distance = milestone_distance
        self._reward_scale = reward_scale
        self._position_fn = position_fn

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment and initializes milestone reward tracking."""
        state = self.env.reset(rng)

        # Get initial position
        initial_position = self._position_fn(state)

        # Add milestone reward tracking info
        info = state.info.copy()
        info.update({
            'initial_position': initial_position,
            'last_milestone': 0.0,
            'total_milestones': 0,
            'distance_traveled': 0.0,
            'current_milestone': 0.0,
        })

        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        """Steps the environment and adds milestone rewards."""
        # Get tracking info
        initial_position = state.info.get('initial_position')
        last_milestone = state.info.get('last_milestone', 0.0)
        total_milestones = state.info.get('total_milestones', 0)

        # Step the environment
        next_state = self.env.step(state, action)

        # Get current position and calculate distance traveled
        current_position = self._position_fn(next_state)
        distance_traveled = current_position - initial_position

        # Calculate the current milestone
        current_milestone = jp.floor(distance_traveled / self._milestone_distance)

        # Check if we've reached a new milestone
        new_milestone_reached = current_milestone > last_milestone

        # Calculate milestone reward
        reward = jp.where(
            new_milestone_reached,
            self._reward_scale * (current_milestone - last_milestone),
            0.0
        )

        # Update the total milestones count
        total_milestones = jp.where(
            new_milestone_reached,
            total_milestones + jp.int32(current_milestone - last_milestone),
            total_milestones
        )

        # Update the last milestone
        last_milestone = jp.where(new_milestone_reached, current_milestone, last_milestone)

        # Update info
        info = next_state.info.copy()
        info.update({
            'initial_position': initial_position,
            'last_milestone': last_milestone,
            'total_milestones': total_milestones,
            'distance_traveled': distance_traveled,
            'current_milestone': current_milestone,
        })

        return next_state.replace(reward=reward, info=info)