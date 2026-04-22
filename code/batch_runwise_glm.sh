#!/bin/bash -l

#SBATCH
#SBATCH --job-name=glm_runwise
#SBATCH --partition=mit_normal
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A.out
#SBATCH --exclude=node3806,node3909,node3908,node3808

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
python code/nilearn_glm_runwise.py \
-s "$s" -t "$task" || { echo "Python script failed"; exit 1; }

echo "Script completed"
