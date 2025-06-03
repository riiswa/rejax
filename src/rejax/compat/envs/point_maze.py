from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Dict, Any
import chex
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    position: chex.Array  # [x, y] position of the point ball
    velocity: chex.Array  # [vx, vy] velocity of the point ball
    desired_goal: chex.Array  # [x, y] position of the desired goal
    time: int  # Current timestep


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 1000
    dt: float = 0.1  # 10 Hz control frequency
    goal_threshold: float = 0.45  # Distance threshold (0.45, not 0.5!)
    max_velocity: float = 5.0  # m/s velocity clipping
    maze_size_scaling: float = 1.0  # Scaling factor for maze
    maze_height: float = 0.4  # Height of maze walls
    position_noise_range: float = 0.25  # Uniform noise range
    maze_map: chex.Array = None  # Discrete maze representation
    continuing_task: bool = False  # From original Gymnasium Robotics
    reset_target: bool = False  # From original Gymnasium Robotics
    reward_type: int = 0  # 0 = sparse, 1 = dense (for JAX compatibility)
    empty_locations: chex.Array = None  # Pre-computed empty cell locations
    num_empty: int = 0  # Number of actual empty cells
    # Optimization: Pre-computed maze boundaries for faster collision detection
    x_map_center: float = 0.0
    y_map_center: float = 0.0
    map_length: int = 0
    map_width: int = 0


class PointMazeEnv(environment.Environment[EnvState, EnvParams]):
    """
    Optimized JAX implementation of Point Maze environment.

    Key optimizations:
    1. Pre-computed maze boundaries
    2. Simplified collision detection
    3. Removed unnecessary lax.cond operations
    4. Vectorized operations where possible
    5. Reduced memory allocations
    """

    def __init__(
            self,
            maze_id: str = "UMaze",
            reward_type: str = "sparse",
            continuing_task: bool = False,
            reset_target: bool = False
    ):
        super().__init__()
        self.maze_id = maze_id
        self.reward_type_str = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target

    @property
    def default_params(self) -> EnvParams:
        """Default parameters with pre-computed optimizations."""
        maze_map = self._get_maze_map(self.maze_id)
        reward_type_int = 0 if self.reward_type_str == "sparse" else 1

        # Pre-compute empty locations once
        empty_locations, num_empty = self._compute_empty_locations(maze_map)

        # Optimization: Pre-compute maze boundaries
        map_length, map_width = maze_map.shape
        x_map_center = map_width / 2 * 1.0
        y_map_center = map_length / 2 * 1.0

        return EnvParams(
            maze_map=maze_map,
            continuing_task=self.continuing_task,
            reset_target=self.reset_target,
            reward_type=reward_type_int,
            empty_locations=empty_locations,
            num_empty=num_empty,
            # Pre-computed values for faster access
            x_map_center=x_map_center,
            y_map_center=y_map_center,
            map_length=map_length,
            map_width=map_width
        )

    def _get_maze_map(self, maze_id: str) -> chex.Array:
        """Get official maze layouts from Gymnasium Robotics."""
        if maze_id == "UMaze":
            maze = [[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]]
        elif maze_id == "Open":
            maze = [[1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1]]
        elif maze_id == "Medium":
            maze = [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]
        elif maze_id == "Large":
            maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        elif maze_id == 'Giant':
            maze = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        else:
            # Default to U_MAZE
            maze = [[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]]
        return jnp.array(maze)

    def _compute_empty_locations(self, maze_map: chex.Array) -> Tuple[chex.Array, int]:
        """Pre-compute empty cell locations once during initialization."""
        map_length, map_width = maze_map.shape
        x_map_center = map_width / 2 * 1.0
        y_map_center = map_length / 2 * 1.0

        # Create all possible coordinates using meshgrid
        i_coords, j_coords = jnp.meshgrid(jnp.arange(map_length), jnp.arange(map_width), indexing='ij')

        # Vectorized coordinate conversion
        x_coords = (j_coords + 0.5) * 1.0 - x_map_center
        y_coords = y_map_center - (i_coords + 0.5) * 1.0

        # Flatten everything
        all_coords = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        empty_mask = (maze_map == 0).flatten()

        # Get indices where mask is True
        empty_indices = jnp.where(empty_mask, size=map_length * map_width, fill_value=0)[0]

        # Select coordinates at empty indices
        empty_coords = all_coords[empty_indices]

        # Count actual empty cells
        num_empty = int(jnp.sum(empty_mask))

        # Ensure at least one valid location
        if num_empty == 0:
            empty_coords = empty_coords.at[0].set(jnp.array([0.0, 0.0]))
            num_empty = 1

        return empty_coords, num_empty

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: chex.Array,
            params: EnvParams
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
        """Optimized step environment."""

        # Optimization: Combine clipping operations
        action = jnp.clip(action, -1.0, 1.0)
        clipped_velocity = jnp.clip(state.velocity, -params.max_velocity, params.max_velocity)

        # Apply forces and update velocity (vectorized)
        new_velocity = jnp.clip(
            clipped_velocity + action * params.dt,
            -params.max_velocity,
            params.max_velocity
        )

        # Update position
        new_position = state.position + new_velocity * params.dt

        # Handle wall collisions (optimized)
        new_position, new_velocity = self._handle_wall_collisions_fast(
            new_position, new_velocity, state.position, params
        )

        # Optimization: Combine distance and reward computation
        distance_to_goal = jnp.linalg.norm(new_position - state.desired_goal)
        goal_reached = distance_to_goal <= params.goal_threshold

        # Simplified reward computation (no lax.cond needed)
        reward = lax.select(
            params.reward_type == 0,
            goal_reached.astype(jnp.float32),  # sparse
            jnp.exp(-distance_to_goal)  # dense
        )

        # Optimized goal reset logic
        should_reset_goal = params.continuing_task & params.reset_target & goal_reached
        new_goal = lax.select(
            should_reset_goal,
            self._generate_goal_position(key, params),
            state.desired_goal
        )

        # Update state (single operation)
        new_state = state.replace(
            position=new_position,
            velocity=new_velocity,
            desired_goal=new_goal,
            time=state.time + 1
        )

        # Simplified termination check
        done_steps = new_state.time >= params.max_steps_in_episode
        done_goal = (~params.continuing_task) & goal_reached
        done = done_steps | done_goal

        # Optimized observation creation (pre-allocated concatenation)
        obs = jnp.concatenate([new_position, new_velocity, new_goal])

        # Info dict
        info = {
            "is_success": goal_reached,
            "discount": self.discount(new_state, params)
        }

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(new_state),
            reward,
            done,
            info
        )

    def reset_env(
            self,
            key: chex.PRNGKey,
            params: EnvParams,
            options: Optional[Dict] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment following Gymnax patterns."""

        key, pos_key, goal_key = jax.random.split(key, 3)

        # Generate goal and reset positions
        goal_position = self._generate_goal_position(goal_key, params)
        goal_position = self._add_xy_position_noise(goal_position, goal_key, params)

        reset_position = self._generate_reset_pos(pos_key, goal_position, params)
        reset_position = self._add_xy_position_noise(reset_position, pos_key, params)

        # Initialize state
        state = EnvState(
            position=reset_position,
            velocity=jnp.zeros(2),
            desired_goal=goal_position,
            time=0
        )

        # Get initial observation (optimized)
        obs = jnp.concatenate([reset_position, jnp.zeros(2), goal_position])

        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Optimized observation creation."""
        return jnp.concatenate([state.position, state.velocity, state.desired_goal])

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        done_steps = state.time >= params.max_steps_in_episode

        # Compute both paths, then select with lax.select
        distance_to_goal = jnp.linalg.norm(state.position - state.desired_goal)
        done_goal = distance_to_goal <= params.goal_threshold

        return lax.select(
            params.continuing_task,
            done_steps,  # If continuing task
            done_steps | done_goal  # If not continuing
        )
    def _compute_reward(
            self,
            achieved_goal: chex.Array,
            desired_goal: chex.Array,
            params: EnvParams
    ) -> chex.Array:
        """Optimized reward computation."""
        distance = jnp.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Simplified without lax.cond
        sparse_reward = (distance <= params.goal_threshold).astype(jnp.float32)
        dense_reward = jnp.exp(-distance)

        return lax.select(params.reward_type == 0, sparse_reward, dense_reward)

    def _generate_goal_position(self, key: chex.PRNGKey, params: EnvParams) -> chex.Array:
        """Generate goal position from pre-computed empty cells (already optimized)."""
        goal_idx = jax.random.randint(key, (), 0, params.num_empty)
        return params.empty_locations[goal_idx]

    def _generate_reset_pos(
            self,
            key: chex.PRNGKey,
            goal_position: chex.Array,
            params: EnvParams
    ) -> chex.Array:
        """Generate reset position using pre-computed empty cells (already optimized)."""
        reset_idx = jax.random.randint(key, (), 0, params.num_empty)
        pos = params.empty_locations[reset_idx]

        return pos

    def _add_xy_position_noise(
            self,
            xy_pos: chex.Array,
            key: chex.PRNGKey,
            params: EnvParams
    ) -> chex.Array:
        """Add position noise using JAX operations."""
        noise_range = params.position_noise_range * params.maze_size_scaling
        noise = jax.random.uniform(
            key, (2,),
            minval=-noise_range,
            maxval=noise_range
        )
        return xy_pos + noise

    def _handle_wall_collisions_fast(
            self,
            new_pos: chex.Array,
            velocity: chex.Array,
            old_pos: chex.Array,
            params: EnvParams
    ) -> Tuple[chex.Array, chex.Array]:
        """Optimized wall collision detection using pre-computed values."""

        x, y = new_pos

        # Use pre-computed values (faster than accessing params.maze_map.shape)
        # Convert to cell indices
        i = jnp.floor((params.y_map_center - y) / params.maze_size_scaling).astype(int)
        j = jnp.floor((x + params.x_map_center) / params.maze_size_scaling).astype(int)

        # Clamp to valid indices using pre-computed values
        i = jnp.clip(i, 0, params.map_length - 1)
        j = jnp.clip(j, 0, params.map_width - 1)

        # Check if cell is a wall
        is_wall = params.maze_map[i, j] == 1

        # Use lax.select for conditional assignment
        final_pos = lax.select(is_wall, old_pos, new_pos)
        final_vel = lax.select(is_wall, jnp.zeros(2), velocity)

        return final_pos, final_vel

    @property
    def name(self) -> str:
        """Environment name."""
        dense_suffix = "Dense" if self.reward_type_str == "dense" else ""
        return f"PointMaze_{self.maze_id}{dense_suffix}-v3"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams = None) -> spaces.Box:
        """Action space: Box(-1.0, 1.0, (2,), float32) - linear forces in x,y."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: [observation(4), desired_goal(2)] = 6 elements."""
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(6,), dtype=jnp.float32
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "position": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(2,), dtype=jnp.float32),
            "velocity": spaces.Box(low=-params.max_velocity, high=params.max_velocity, shape=(2,), dtype=jnp.float32),
            "desired_goal": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(2,), dtype=jnp.float32),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })


# Factory functions for the basic maze layouts
def make_point_maze_umaze(reward_type: str = "sparse") -> PointMazeEnv:
    """PointMaze_UMaze-v3 / PointMaze_UMazeDense-v3"""
    return PointMazeEnv(maze_id="UMaze", reward_type=reward_type)


def make_point_maze_open(reward_type: str = "sparse") -> PointMazeEnv:
    """PointMaze_Open-v3 / PointMaze_OpenDense-v3"""
    return PointMazeEnv(maze_id="Open", reward_type=reward_type)


def make_point_maze_medium(reward_type: str = "sparse") -> PointMazeEnv:
    """PointMaze_Medium-v3 / PointMaze_MediumDense-v3"""
    return PointMazeEnv(maze_id="Medium", reward_type=reward_type)


def make_point_maze_large(reward_type: str = "sparse") -> PointMazeEnv:
    """PointMaze_Large-v3 / PointMaze_LargeDense-v3"""
    return PointMazeEnv(maze_id="Large", reward_type=reward_type)


# Simple demo with pygame animation
if __name__ == "__main__":
    import pygame
    import numpy as np
    import time

    print("=== Point Maze Environment Demo ===")

    # Create environment
    env = make_point_maze_umaze(reward_type="sparse")
    params = env.default_params

    print(f"Environment: {env.name}")
    print(f"Maze shape: {params.maze_map.shape}")
    print(f"Pre-computed empty locations: {params.num_empty} cells")

    # Test JIT compilation
    reset_fn = jax.jit(env.reset_env)
    step_fn = jax.jit(env.step_env)

    # Warm up JIT
    key = jax.random.PRNGKey(0)
    obs_jit, state_jit = reset_fn(key, params)
    action = jnp.array([1.0, 0.5])
    obs_jit, state_jit, _, _, _ = step_fn(key, state_jit, action, params)
    print(f"JIT compilation successful")

    print("\n=== Running Random Agent Trajectory (1000 steps, auto-reset) ===")


    # Create jitted trajectory runner for 1000 steps with auto-reset
    @partial(jax.jit, static_argnums=2)
    def run_long_trajectory(key, params, total_steps=1000):
        """Run long trajectory with auto-reset when episodes end."""

        # Initialize first episode
        obs, state = env.reset_env(key, params)

        # Pre-allocate arrays for full trajectory
        positions = jnp.zeros((total_steps + 1, 2))
        goals = jnp.zeros((total_steps + 1, 2))  # Store goal positions too
        rewards = jnp.zeros(total_steps)
        episode_ends = jnp.zeros(total_steps, dtype=bool)

        # Store initial position and goal
        positions = positions.at[0].set(state.position)
        goals = goals.at[0].set(state.desired_goal)

        def trajectory_step(carry, i):
            key, state = carry

            # Generate random action
            key, action_key = jax.random.split(key)
            action = jax.random.uniform(action_key, (2,), minval=-1.0, maxval=1.0)

            # Step environment
            key, step_key = jax.random.split(key)
            obs, new_state, reward, done, info = env.step_env(step_key, state, action, params)

            # Check if episode ended (success or max steps)
            episode_ended = done

            # If episode ended, reset environment
            def reset_env_fn(reset_key):
                new_obs, new_state = env.reset_env(reset_key, params)
                return new_state

            def keep_state_fn(reset_key):
                return new_state

            key, reset_key = jax.random.split(key)
            final_state = lax.cond(
                episode_ended,
                reset_env_fn,
                keep_state_fn,
                reset_key
            )

            # Store data for this step
            step_data = {
                'position': final_state.position,
                'goal': final_state.desired_goal,  # Store actual goal from environment
                'reward': reward,
                'episode_end': episode_ended
            }

            return (key, final_state), step_data

        # Run trajectory loop
        initial_carry = (key, state)
        final_carry, trajectory_data = lax.scan(trajectory_step, initial_carry, jnp.arange(total_steps))

        # Extract trajectory data
        all_positions = jnp.concatenate([
            positions[0:1],  # Initial position
            trajectory_data['position']  # All step positions
        ])

        all_goals = jnp.concatenate([
            goals[0:1],  # Initial goal
            trajectory_data['goal']  # All step goals
        ])

        # Count episodes completed
        num_episodes = jnp.sum(trajectory_data['episode_end']) + 1
        total_reward = jnp.sum(trajectory_data['reward'])

        return {
            'positions': all_positions,
            'goals': all_goals,  # Include actual goals from environment
            'rewards': trajectory_data['reward'],
            'episode_ends': trajectory_data['episode_end'],
            'total_reward': total_reward,
            'num_episodes': num_episodes,
            'total_steps': total_steps
        }


    # Run long trajectory
    key = jax.random.PRNGKey(123)

    print("Running 1000-step trajectory with auto-reset...")
    start_time = time.time()
    trajectory = run_long_trajectory(key, params, total_steps=1000)
    execution_time = time.time() - start_time

    print(f"Trajectory completed in {execution_time:.3f}s")
    print(f"Total steps: {trajectory['total_steps']}")
    print(f"Episodes completed: {trajectory['num_episodes']}")
    print(f"Total reward: {trajectory['total_reward']:.3f}")


    # Very minimalist pygame animation
    def create_simple_animation(trajectory, params, save_gif=True):
        """Very minimalist pygame animation - using actual goals from environment."""

        # Initialize pygame
        pygame.init()

        # Minimalist constants
        CELL_SIZE = 60
        WALL_COLOR = (20, 20, 20)  # Black walls
        EMPTY_COLOR = (250, 250, 250)  # White background
        AGENT_COLOR = (120, 120, 120)  # Grey agent
        GOAL_COLOR = (60, 180, 60)  # Green goal
        GOAL_ALPHA = 80  # Low opacity for goal

        # Maze dimensions
        maze_map = params.maze_map
        map_height, map_width = maze_map.shape

        # Screen dimensions
        screen_width = map_width * CELL_SIZE
        screen_height = map_height * CELL_SIZE

        # Create screen
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Point Maze")

        # Convert trajectory positions to screen coordinates
        def world_to_screen(pos):
            x_map_center = map_width / 2 * params.maze_size_scaling
            y_map_center = map_height / 2 * params.maze_size_scaling

            grid_x = (pos[0] + x_map_center) / params.maze_size_scaling
            grid_y = (y_map_center - pos[1]) / params.maze_size_scaling

            screen_x = int(grid_x * CELL_SIZE)
            screen_y = int(grid_y * CELL_SIZE)

            return screen_x, screen_y

        # Get positions and goals from environment
        positions = trajectory['positions']
        goals = trajectory['goals']
        total_steps = int(trajectory['total_steps'])

        # Calculate agent size
        agent_radius = 8  # 8px agent size as requested

        # Calculate correct goal radius in screen coordinates
        goal_radius_world = params.goal_threshold  # 0.45 in world units
        goal_radius_screen = int(goal_radius_world * CELL_SIZE / params.maze_size_scaling)
        goal_radius_screen = max(goal_radius_screen, 3)  # Minimum visible size

        # Create surface for goal with alpha
        goal_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

        # Animation frames - sample every 3 steps
        frames = []
        clock = pygame.time.Clock()
        step_sample = 3
        animation_steps = list(range(0, total_steps + 1, step_sample))

        print(f"Creating {len(animation_steps)} frames...")
        print(f"Using actual goals from environment")
        print(f"Agent radius: {agent_radius} pixels")
        print(f"Goal radius: {goal_radius_world} world units -> {goal_radius_screen} pixels")

        for step in animation_steps:
            # Fill background
            screen.fill(EMPTY_COLOR)

            # Draw maze walls - pure minimalist
            for i in range(map_height):
                for j in range(map_width):
                    if maze_map[i, j] == 1:  # Wall
                        wall_x = j * CELL_SIZE
                        wall_y = i * CELL_SIZE
                        wall_rect = pygame.Rect(wall_x, wall_y, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, WALL_COLOR, wall_rect)

            # Get current goal position from environment
            if step < len(goals):
                current_goal = goals[step]
                goal_screen_pos = world_to_screen(current_goal)

                # Draw goal area with correct size and transparency - GREEN
                goal_surface.fill((0, 0, 0, 0))  # Clear with full transparency
                pygame.draw.circle(goal_surface, (*GOAL_COLOR, GOAL_ALPHA), goal_screen_pos, goal_radius_screen)
                screen.blit(goal_surface, (0, 0))

            # Get current position and draw agent
            if step < len(positions):
                current_pos = positions[step]
                screen_pos = world_to_screen(current_pos)

                # Draw agent - realistic tiny point, GREY
                pygame.draw.circle(screen, AGENT_COLOR, screen_pos, agent_radius)

            # Save frame for gif
            if save_gif:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frames.append(frame)

            pygame.display.flip()
            clock.tick(20)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return frames if save_gif else None

        pygame.quit()
        return frames if save_gif else None


    print(f"\nCreating minimalist pygame animation...")

    try:
        frames = create_simple_animation(trajectory, params, save_gif=True)

        if frames:
            try:
                import imageio

                print(f"Saving animation as 'point_maze_simple.gif'...")
                imageio.mimsave('point_maze_simple.gif', frames, fps=15)
                print(f"✅ Animation saved! ({len(frames)} frames)")
            except ImportError:
                print("⚠️ imageio not available. Install with: pip install imageio")
                print(f"Animation displayed with {len(frames)} frames")

    except Exception as e:
        print(f"⚠️ Pygame animation failed: {e}")
        print("Make sure you have pygame installed: pip install pygame")

    print("✅ Demo completed!")