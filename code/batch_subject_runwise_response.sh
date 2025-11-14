#!/bin/bash -l

#SBATCH
#SBATCH --job-name=subject_runwise_response
#SBATCH --partition=mit_normal
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%A.out

s=$1

dir=$(pwd)

echo "Starting script with s=$s and task=$task"

# Initialize Conda (adjust path if needed)
source ~/.bashrc

# Activate environment
conda activate nilearn || { echo "Failed to activate conda env"; exit 1; }

# Debug info
echo "Using python: $(which python)"
python --version

# Run Python script
python code/runwise_response.py -s "$s" -d "$dir" || { echo "Python script failed"; exit 1; }

echo "Script completed"
