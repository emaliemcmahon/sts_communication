#!/bin/bash -l

#SBATCH
#SBATCH --job-name=random_effects
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=24
#SBATCH --exclude=node064

s=$1
task=${2:-"communicate"}

echo "Starting script with s=$s and task=$task"

# Initialize Conda (adjust path if needed)
source ~/.bashrc

# Activate environment
conda activate nilearn || { echo "Failed to activate conda env"; exit 1; }

# Debug info
echo "Using python: $(which python)"
python --version

# Run Python script
python nilearn_glm.py -s "$s" -t "$task" || { echo "Python script failed"; exit 1; }

echo "Script completed"