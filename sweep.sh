#!/bin/bash
#SBATCH --job-name=exploration         # Job name
#SBATCH --output=output.log        # Output file
#SBATCH --error=error.log          # Error file
#SBATCH --time=08:00:00            # Maximum run time
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --constraint=a100

# Define bonus types
BONUS_TYPES="rnk,rnd,none"

# Define environments
ENVS=(
  "MountainCar-v0"
  "Acrobot-v1"
  "navix/Navix-FourRooms-v0"
  "navix/Navix-DoorKey-8x8-v0"
  "navix/Navix-KeyCorridorS5R3-v0"
  "navix/Navix-SimpleCrossingS9N1-v0"
  "navix/Navix-LavaCrossingS9N1-v0"
  "navix/Navix-DistShift1-v0"
  "navix/Navix-LavaGapS7-v0"
  "navix/Navix-GoToDoor-8x8-v0"
  "custom/pointmaze-umaze-v0"
  "custom/pointmaze-medium-v0"
  "custom/pointmaze-large-v0"
  "brax/sparse-ant"
  "brax/sparse-halfcheetah"
  "brax/sparse-walker2d"
  "brax/sparse-hopper"
)

# Convert environments to a comma-separated string
ENV_LIST=$(IFS=, ; echo "${ENVS[*]}")

# Run hydra multirun
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 python experiment.py -m training.bonus_type=$BONUS_TYPES training.env=$ENV_LIST