for bonus in rnk rnd hash vime none; do sbatch --constraint=p100 exploration_generic.sh "$bonus"; done
