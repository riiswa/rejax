import os
import time  # Add this import
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
from aim import Run
import pandas as pd
import numpy as np
from rejax.algos.ppo_with_bonus import PPO


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert Hydra config to dict for PPO.create
    config = OmegaConf.to_container(cfg.training)

    # Create experiment name only if not provided in config
    if hasattr(cfg.experiment, 'name') and cfg.experiment.name:
        experiment_name = cfg.experiment.name
        print(f"🚀 Starting experiment: {experiment_name} (from config)")
    else:
        # Auto-generate: env_exploration_seed
        env_name = cfg.training.env.split("/")[-1]  # Extract name after "/"
        exploration_type = cfg.training.bonus_type
        experiment_name = f"{env_name}_{exploration_type}_seed{cfg.experiment.seed}"
        print(f"🚀 Starting experiment: {experiment_name} (auto-generated)")

    print(f"📊 Environment: {cfg.training.env}")
    print(f"🔍 Exploration: {cfg.training.bonus_type}")
    print(f"🌱 Seeds: {cfg.experiment.n_seeds} (starting from {cfg.experiment.seed})")

    if cfg.experiment.logging:
        run = Run(experiment=experiment_name)
        run["hparams"] = config

    ppo = PPO.create(**config)
    eval_callback = ppo.eval_callback

    def log(step, data, seed):
        step = step.item()
        seed = seed.item() - cfg.experiment.seed
        for k, v in data.items():
            try:
                run.track(v.item(), name=k, step=step, context={"seed": seed})
            except ValueError:
                print(f"Error: {(k, v)}")

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

    if cfg.experiment.logging:
        ppo = ppo.replace(eval_callback=logging_callback, logging_callback=log)

    train_fn = jax.jit(ppo.train_with_seed)
    vmapped_train_fn = jax.vmap(train_fn)

    n_seeds = cfg.experiment.n_seeds

    # Time the training execution
    start_time = time.time()
    train_state, (episode_lengths, returns) = vmapped_train_fn(jnp.arange(n_seeds) + cfg.experiment.seed)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"⏱️  Training completed in {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")

    print(f"✅ Experiment '{experiment_name}' completed!")

    # Save returns to CSV
    # returns shape: (n_seeds, n_steps, n_evals) -> average over n_evals
    returns_avg = np.array(returns.mean(axis=-1))  # Shape: (n_seeds, n_steps)
    n_seeds, n_steps = returns_avg.shape

    # Create timesteps column
    timesteps = np.arange(n_steps) * cfg.training.eval_freq

    # Create DataFrame
    data = {"timestep": timesteps}
    for seed_idx in range(n_seeds):
        data[f"seed_{seed_idx}"] = returns_avg[seed_idx]

    df = pd.DataFrame(data)
    os.makedirs("results/", exist_ok=True)
    csv_filename = f"results/{experiment_name}_returns.csv"
    df.to_csv(csv_filename, index=False)

    print(f"💾 Returns saved to: {csv_filename}")
    print(f"📊 CSV shape: {df.shape} (timesteps x seeds)")


if __name__ == "__main__":
    main()