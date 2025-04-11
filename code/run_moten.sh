#!/bin/bash -l

#SBATCH
#SBATCH --job-name=moten
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=24

conda activate moten

python motion_energy_activations.py
