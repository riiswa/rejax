from rejax.compat.envs.cartpole_swing_up import CartPoleSwingUp
import pointax


def create_custom(env_name, **kwargs):
    if env_name == 'cartpole-swingup-v0':
        env = CartPoleSwingUp(**kwargs)
        return env, env.default_params
    elif env_name == 'pointmaze-umaze-v0':
        env = pointax.make_umaze(**kwargs)
        return env, env.default_params
    elif env_name == 'pointmaze-open-v0':
        env = pointax.make_open(**kwargs)
        return env, env.default_params
    elif env_name == 'pointmaze-medium-v0':
        env = pointax.make_medium(**kwargs)
        return env, env.default_params
    elif env_name == 'pointmaze-large-v0':
        env = pointax.make_large(**kwargs)
        return env, env.default_params
    elif env_name == 'pointmaze-giant-v0':
        env = pointax.make_giant(**kwargs)
        return env, env.default_params
