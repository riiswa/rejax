#!/bin/bash
#SBATCH --job-name=exploration_job
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
# Constraint must be passed during sbatch submission: sbatch --constraint=<gpu> ...

# === ARGUMENTS ===
EXPLORATION_TYPE="$1"

if [[ -z "$EXPLORATION_TYPE" ]]; then
  echo "Usage: sbatch --constraint=<constraint> $0 <exploration_type>"
  echo "Example: sbatch --constraint=p100 $0 vime"
  exit 1
fi

# === CONSTANTS ===
MAX_PARALLEL=2
RESULTS_FILE="experiment_results.log"

# === ENV SETUP ===
source envs.sh

# === UTILITY FUNCTIONS ===

wait_for_slot() {
  while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 1
  done
}

log_result() {
  local status=$1
  local env=$2
  local message=${3:-""}
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$timestamp|$EXPLORATION_TYPE|$env|$status|$message" >> "$RESULTS_FILE"
}

run_experiment() {
  local env=$1
  local gpu_id=$2

  echo ">>> Starting: bonus_type=$EXPLORATION_TYPE, env=$env on GPU $gpu_id"

  local extra_args=""
  if [[ "$env" == *"brax"* || "$env" == *"MinAtar"* ]]; then
    extra_args="training.total_timesteps=2_500_000"
  fi

  if CUDA_VISIBLE_DEVICES=$gpu_id python experiment.py training.bonus_type="$EXPLORATION_TYPE" training.env="$env" $extra_args; then
    log_result "SUCCESS" "$env"
    echo "✓✓✓ Completed: $env"
  else
    log_result "FAILED" "$env" "Python script failed"
    echo "✗✗✗ Failed: $env"
  fi
}

# === MAIN EXECUTION ===

echo "=== Launching experiments for bonus_type=$EXPLORATION_TYPE with max $MAX_PARALLEL parallel jobs ==="
echo "Constraint is passed via sbatch (e.g., --constraint=p100)"
echo "Total environments: ${#ENVS[@]}"

experiment_count=0
for env in "${ENVS[@]}"; do
  wait_for_slot
  gpu_id=$((experiment_count % 2))
  run_experiment "$env" "$gpu_id" &
  ((experiment_count++))
done

wait
echo "=== All experiments for bonus_type=$EXPLORATION_TYPE completed ==="
