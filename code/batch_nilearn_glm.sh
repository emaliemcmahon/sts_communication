#!/bin/bash -l

#SBATCH --job-name=nilearn_glm
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A.out

sub=$1
task=$2

USER=$(whoami)
top_dir="/orcd/data/ngk/001/users/${USER}/sts_communication"

echo "Top dir: ${top_dir}"
echo "Running first-level GLM for subject $sub, task $task"

echo "Starting script with s=$sub and task=$task"

# Initialize Conda (adjust path if needed)
source ~/.bashrc

# Activate environment
conda activate nilearn || { echo "Failed to activate conda env"; exit 1; }

# Debug info
echo "Using python: $(which python)"
python --version

python ${top_dir}/code/nilearn_glm.py $sub \
-t $task || { echo "Python script failed"; exit 1; }

echo "Script completed"
