for bonus in rnk rnd hash vime none; do sbatch --constraint=p100 run.sh "$bonus"; done
