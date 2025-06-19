for bonus in rnk rnd hash vime none; do sbatch --constraint=a100 run.sh "$bonus"; done
