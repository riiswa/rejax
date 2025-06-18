from rejax.compat.envs.cartpole_swing_up import CartPoleSwingUp
from rejax.compat.envs.point_maze import PointMazeEnv


def create_custom(env_name, **kwargs):
    if env_name == 'cartpole-swingup-v0':
        env = CartPoleSwingUp(**kwargs)
        return env, env.default_params
    elif env_name == 'pointmaze-umaze-v0':
        env = PointMazeEnv("UMaze")
        return env, env.default_params
    elif env_name == 'pointmaze-open-v0':
        env = PointMazeEnv("Open")
        return env, env.default_params
    elif env_name == 'pointmaze-medium-v0':
        env = PointMazeEnv("Medium")
        return env, env.default_params
    elif env_name == 'pointmaze-large-v0':
        env = PointMazeEnv("Large")
        return env, env.default_params
    elif env_name == 'pointmaze-giant-v0':
        env = PointMazeEnv("Giant")
        return env, env.default_params
