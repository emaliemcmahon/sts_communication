#!/bin/bash -l

#SBATCH
#SBATCH --job-name=group_runwise_results
#SBATCH --time=3020:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=2
#SBATCH --exclude=node064
#SBATCH --output=logs/%x_%A.out

s=$1

echo "Starting script with s=$s"

# Initialize Conda (adjust path if needed)
source ~/.bashrc

# Activate environment
conda activate nilearn || { echo "Failed to activate conda env"; exit 1; }

# Debug info
echo "Using python: $(which python)"
python --version

# All participants
subs=(1 2 3 4 5 7 8 9 11 12 13 14 15 16)

# Run Python script
python code/group_runwise_results.py "${subs[@]}" || { echo "Python script failed"; exit 1; }

echo "Script completed"
