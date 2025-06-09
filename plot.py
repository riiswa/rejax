import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define consistent colors and markers for algorithms
ALGORITHM_COLORS = {
    'PPO': '#2E86AB',  # Blue
    'PPO+RND': '#A23B72',  # Purple/Magenta
    'PPO+RFIG': '#F18F01'  # Orange (baseline)
}

ALGORITHM_MARKERS = {
    'PPO': 'o',  # Circle
    'PPO+RND': 's',  # Square
    'PPO+RFIG': '^'  # Triangle (baseline)
}

ALGORITHM_MAPPING = {
    'none': 'PPO',
    'rnd': 'PPO+RND',
    'rnk': 'PPO+RFIG'
}

# Define consistent order for algorithms (baseline last for emphasis)
ALGORITHM_ORDER = ['PPO', 'PPO+RND', 'PPO+RFIG']


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract environment and algorithm.
    Expected format: {Environment}_{algorithm}_seed0_returns.csv
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')

    # Split by underscores and find the algorithm part
    parts = name.split('_')

    # The algorithm should be the second to last part (before 'seed0')
    if len(parts) >= 3 and parts[-2] == 'seed0' and parts[-1] == 'returns':
        algorithm = parts[-3]
        environment = '_'.join(parts[:-3])
    else:
        # Fallback parsing
        match = re.match(r'(.+)_(none|rnd|rnk)_seed0_returns', name)
        if match:
            environment, algorithm = match.groups()
        else:
            raise ValueError(f"Cannot parse filename: {filename}")

    return environment, algorithm


def calculate_iqr_stats(data: np.ndarray, axis: int = 1) -> Dict[str, np.ndarray]:
    """
    Calculate interquartile mean and quartiles.

    Args:
        data: Array where each row is a timestep and each column is a seed
        axis: Axis along which to calculate statistics

    Returns:
        Dictionary with mean, q25, q75, and iqr_mean
    """
    q25 = np.percentile(data, 25, axis=axis)
    q75 = np.percentile(data, 75, axis=axis)
    median = np.percentile(data, 50, axis=axis)
    mean = np.mean(data, axis=axis)

    # Interquartile mean (mean of values between q25 and q75)
    iqr_mean = []
    for i in range(data.shape[0]):
        row_data = data[i, :]
        iqr_mask = (row_data >= q25[i]) & (row_data <= q75[i])
        if np.any(iqr_mask):
            iqr_mean.append(np.mean(row_data[iqr_mask]))
        else:
            iqr_mean.append(median[i])

    return {
        'mean': mean,
        'iqr_mean': np.array(iqr_mean),
        'q25': q25,
        'q75': q75,
        'median': median
    }


def load_and_process_data(results_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all CSV files and organize by environment and algorithm.

    Returns:
        Nested dictionary: {environment: {algorithm: dataframe}}
    """
    results_path = Path(results_dir)
    data = {}

    # Find all CSV files
    csv_files = list(results_path.glob("*_returns.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    print(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        try:
            environment, algorithm = parse_filename(csv_file.name)

            # Load the CSV
            df = pd.read_csv(csv_file)

            # Initialize nested dict structure
            if environment not in data:
                data[environment] = {}

            data[environment][algorithm] = df
            print(f"Loaded: {environment} - {algorithm}")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    return data


def create_environment_plot(env_name: str, env_data: Dict[str, pd.DataFrame],
                            save_path: str = None) -> plt.Figure:
    """
    Create a plot for a single environment with all algorithms.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.25))

    # Sort algorithms by consistent order
    def get_algorithm_order(algorithm):
        display_name = ALGORITHM_MAPPING.get(algorithm, algorithm)
        try:
            return ALGORITHM_ORDER.index(display_name)
        except ValueError:
            return len(ALGORITHM_ORDER)  # Put unknown algorithms at the end

    sorted_algorithms = sorted(env_data.keys(), key=get_algorithm_order)

    for algorithm in sorted_algorithms:
        df = env_data[algorithm]
        # Get algorithm display name, color, and marker
        display_name = ALGORITHM_MAPPING.get(algorithm, algorithm)
        color = ALGORITHM_COLORS.get(display_name, '#333333')
        marker = ALGORITHM_MARKERS.get(display_name, 'o')

        # Extract timesteps and seed data
        timesteps = df['timestep'].values

        # Get all seed columns
        seed_cols = [col for col in df.columns if col.startswith('seed_')]
        seed_data = df[seed_cols].values

        # Calculate IQR statistics
        stats = calculate_iqr_stats(seed_data)

        # Plot IQR mean with quartile bands and markers
        ax.plot(timesteps, stats['iqr_mean'], label=display_name,
                color=color, linewidth=2, alpha=0.9, marker=marker,
                markersize=4, markevery=max(1, len(timesteps) // 15))

        ax.fill_between(timesteps, stats['q25'], stats['q75'],
                        color=color, alpha=0.2)

    # Styling
    ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Returns', fontsize=11, fontweight='bold')

    # Clean up the environment name for display
    clean_env_name = env_name.replace('-v0', '').replace('-v1', '')
    clean_env_name = clean_env_name.replace('pointmaze-', 'PointMaze-')
    ax.set_title(clean_env_name, fontsize=12, fontweight='bold', pad=15)

    # No legend on individual plots

    # Grid and spines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    return fig


def create_standalone_legend(output_dir: str):
    """
    Create a standalone horizontal legend and save as PDF.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 1))

    # Create dummy plots to generate legend
    for display_name in ALGORITHM_ORDER:
        color = ALGORITHM_COLORS.get(display_name, '#333333')
        marker = ALGORITHM_MARKERS.get(display_name, 'o')

        ax.plot([], [], label=display_name, color=color, linewidth=2,
                marker=marker, markersize=6, alpha=0.9)

    # Remove all axes elements
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Create horizontal legend
    legend = ax.legend(loc='center', frameon=False, fontsize=12,
                       ncol=len(ALGORITHM_ORDER), columnspacing=2.0)

    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'legend.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    plt.close(fig)

    print(f"Legend saved to: {save_path}")


def create_all_plots(results_dir: str, output_dir: str = None):
    """
    Create plots for all environments and save them.
    """
    # Load all data
    data = load_and_process_data(results_dir)

    if output_dir is None:
        output_dir = os.path.join(results_dir, 'plots')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating plots for {len(data)} environments...")

    # Create individual plots for each environment
    for env_name, env_data in data.items():
        print(f"Plotting {env_name}...")

        # Clean filename
        clean_name = env_name.replace('-v0', '').replace('-v1', '')
        filename = f"{clean_name}_results.pdf"
        save_path = os.path.join(output_dir, filename)

        fig = create_environment_plot(env_name, env_data, save_path)
        plt.close(fig)  # Close to free memory

    # Create standalone legend
    create_standalone_legend(output_dir)

    print(f"\nPlots saved to: {output_dir}")


def main():
    """
    Main function to run the plotting script.
    Usage: python plot_rl_results.py
    """
    # Directory containing the results
    results_dir = "results"

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        print("Please make sure your CSV files are in a 'results' folder.")
        return

    try:
        # Create all plots
        create_all_plots(results_dir)
        print("✅ All plots created successfully!")

    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        raise


if __name__ == "__main__":
    main()