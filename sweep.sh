#!/bin/bash
#SBATCH --job-name=exploration         # Job name
#SBATCH --output=output.log        # Output file
#SBATCH --error=error.log          # Error file
#SBATCH --time=12:00:00            # Maximum run time
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --constraint=a100

# Maximum number of parallel jobs (consider reducing for GPU memory)
MAX_PARALLEL=2  # 2 per GPU instead of 4 per GPU
# Alternative: MAX_PARALLEL=6  # 3 per GPU

# Define bonus types
BONUS_TYPES=("rnk" "rnd" "vime" "hash" "none")

# Define environments
ENVS=(
  "MountainCar-v0"
  "Acrobot-v1"
  "Asterix-MinAtar"
  "Breakout-MinAtar"
  "Freeway-MinAtar"
  "SpaceInvaders-MinAtar"
#  "Seaquest-MinAtar"
#  "navix/Navix-Empty-8x8-v0"
#  "navix/Navix-Empty-16x16-v0"
#  "navix/Navix-DoorKey-5x5-v0"
#  "navix/Navix-DoorKey-6x6-v0"
#  "navix/Navix-DoorKey-8x8-v0"
#  "navix/Navix-DoorKey-16x16-v0"
#  "navix/Navix-FourRooms-v0"
#  "navix/Navix-SimpleCrossingS9N1-v0"
#  "navix/Navix-SimpleCrossingS9N2-v0"
#  "navix/Navix-SimpleCrossingS9N3-v0"
#  "navix/Navix-SimpleCrossingS11N5-v0"
#  "navix/Navix-DistShift1-v0"
#  "navix/Navix-DistShift2-v0"
#  "navix/Navix-LavaGapS5-v0"
#  "navix/Navix-LavaGapS6-v0"
#  "navix/Navix-LavaGapS7-v0"
#  "navix/Navix-GoToDoor-5x5-v0"
#  "navix/Navix-GoToDoor-6x6-v0"
#  "navix/Navix-GoToDoor-8x8-v0"
  "custom/pointmaze-umaze-v0"
  "custom/pointmaze-medium-v0"
  "custom/pointmaze-large-v0"
  "custom/pointmaze-giant-v0"
  "brax/sparse-ant"
  "brax/sparse-halfcheetah"
  "brax/sparse-walker2d"
  "brax/sparse-hopper"
  "brax/pusher"
  "brax/reacher"
#  "brax/ant"
#  "brax/halfcheetah"
#  "brax/walker2d"
#  "brax/hopper"
#  "brax/humanoid"
#  "brax/humanoidstandup"
)

# Function to wait for a job slot to become available
wait_for_slot() {
  while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 1
  done
}

# Function to run a single experiment
run_experiment() {
  local bonus_type=$1
  local env=$2
  local gpu_id=$3
  echo "Starting: bonus_type=$bonus_type, env=$env on GPU $gpu_id"
  if [[ "$env" == *"brax"* || "$env" == *"MinAtar"* ]]; then
    CUDA_VISIBLE_DEVICES=$gpu_id XLA_PYTHON_CLIENT_PREALLOCATE=false python experiment.py training.bonus_type=$bonus_type training.env=$env training.total_timesteps=2_500_000
  else
    CUDA_VISIBLE_DEVICES=$gpu_id XLA_PYTHON_CLIENT_PREALLOCATE=false python experiment.py training.bonus_type=$bonus_type training.env=$env
  fi
  echo "Completed: bonus_type=$bonus_type, env=$env on GPU $gpu_id"
}

# Main execution loop
echo "Starting experiments with $MAX_PARALLEL parallel jobs..."
echo "Total experiments: $((${#ENVS[@]} * ${#BONUS_TYPES[@]}))"

experiment_count=0
for env in "${ENVS[@]}"; do
  for bonus_type in "${BONUS_TYPES[@]}"; do
    # Wait for an available slot
    wait_for_slot

    # Alternate between GPU 0 and 1
    gpu_id=$((experiment_count % 2))

    # Start the experiment in background
    run_experiment "$bonus_type" "$env" "$gpu_id" &

    # Increment experiment counter
    ((experiment_count++))
  done
done

# Wait for all remaining jobs to complete
wait

echo "All experiments completed!"