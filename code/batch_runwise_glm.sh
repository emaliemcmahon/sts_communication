#!/bin/bash -l

#SBATCH
#SBATCH --job-name=glm_runwise
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%A.out

s=$1
task=$2

echo "Starting script with s=$s"

# Initialize Conda (adjust path if needed)
source ~/.bashrc

# Activate environment
conda activate nilearn || { echo "Failed to activate conda env"; exit 1; }

# Debug info
echo "Using python: $(which python)"
python --version

# Run Python script
python code/nilearn_glm_runwise.py -s "$s" -t "$task" || { echo "Python script failed"; exit 1; }

echo "Script completed"
