#!/bin/bash -l

#SBATCH
#SBATCH --job-name=random_effects
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=24
#SBATCH --exclude=node064

c1=$1
c2=$2

conda activate nilearn

python group_random_effects.py \
    -c1 $c1 -c2 $c2

