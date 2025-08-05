#!/bin/bash -l

#SBATCH
#SBATCH --job-name=runwise_response
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=8
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

# Run Python script
python code/nilearn_glm_runwise.py -s "$s" || { echo "Python script failed"; exit 1; }
# python code/runwise_response.py -s "$s" || { echo "Python script failed"; exit 1; }
# python code/visualize_rois.py -s "$s" || { echo "Python script failed"; exit 1; }

echo "Script completed"
