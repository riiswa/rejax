#!/bin/bash
#SBATCH --job-name=exploration_rnd     # Job name
#SBATCH --output=output_rnd.log        # Output file
#SBATCH --error=error_rnd.log          # Error file
#SBATCH --time=12:00:00                # Maximum run time
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --constraint=p100

# Maximum number of parallel jobs (consider reducing for GPU memory)
MAX_PARALLEL=2  # 2 per GPU instead of 4 per GPU
# Alternative: MAX_PARALLEL=6  # 3 per GPU

# Define bonus type for this script
BONUS_TYPE="rnd"

# Source common environment definitions
source envs.sh

# Function to wait for a job slot to become available
wait_for_slot() {
  while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 1
  done
}

# Common results file for all experiments
RESULTS_FILE="experiment_results.log"

# Function to log experiment result
log_result() {
  local status=$1
  local env=$2
  local message=${3:-""}
  local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$timestamp|$BONUS_TYPE|$env|$status|$message" >> "$RESULTS_FILE"
}

# Function to run a single experiment
run_experiment() {
  local env=$1
  local gpu_id=$2
  echo "Starting: bonus_type=$BONUS_TYPE, env=$env on GPU $gpu_id"
  
  if [[ "$env" == *"brax"* || "$env" == *"MinAtar"* ]]; then
    if CUDA_VISIBLE_DEVICES=$gpu_id XLA_PYTHON_CLIENT_PREALLOCATE=false python experiment.py training.bonus_type=$BONUS_TYPE training.env=$env training.total_timesteps=2_500_000; then
      log_result "SUCCESS" "$env" "Completed successfully"
      echo "Completed: bonus_type=$BONUS_TYPE, env=$env on GPU $gpu_id"
    else
      log_result "FAILED" "$env" "Python script failed"
      echo "Failed: bonus_type=$BONUS_TYPE, env=$env on GPU $gpu_id"
    fi
  else
    if CUDA_VISIBLE_DEVICES=$gpu_id XLA_PYTHON_CLIENT_PREALLOCATE=false python experiment.py training.bonus_type=$BONUS_TYPE training.env=$env; then
      log_result "SUCCESS" "$env" "Completed successfully"
      echo "Completed: bonus_type=$BONUS_TYPE, env=$env on GPU $gpu_id"
    else
      log_result "FAILED" "$env" "Python script failed"
      echo "Failed: bonus_type=$BONUS_TYPE, env=$env on GPU $gpu_id"
    fi
  fi
}

# Main execution loop
echo "Starting experiments for bonus_type=$BONUS_TYPE with $MAX_PARALLEL parallel jobs..."
echo "Total experiments: ${#ENVS[@]}"

experiment_count=0
for env in "${ENVS[@]}"; do
  # Wait for an available slot
  wait_for_slot

  # Alternate between GPU 0 and 1
  gpu_id=$((experiment_count % 2))

  # Start the experiment in background
  run_experiment "$env" "$gpu_id" &

  # Increment experiment counter
  ((experiment_count++))
done

# Wait for all remaining jobs to complete
wait

echo "All experiments for bonus_type=$BONUS_TYPE completed!"