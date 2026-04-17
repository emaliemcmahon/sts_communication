#!/bin/bash -l

#SBATCH --job-name=random_effects
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%A.out

c1=$1
c2=$2
task=$3

USER=$(whoami)
top_dir="/orcd/data/ngk/001/users/${USER}/sts_communication"

echo "Top dir: ${top_dir}"
echo "Starting script with c1=$c1, c2=$c2, and task=$task"

# Initialize Conda (adjust path if needed)
source ~/.bashrc

# Activate environment
conda activate nilearn || { echo "Failed to activate conda env"; exit 1; }

# Debug info
echo "Using python: $(which python)"
python --version

python ${top_dir}/code/group_random_effects.py \
    -c1 $c1 -c2 $c2 -t $task || { echo "Python script failed"; exit 1; }
python ${top_dir}/code/plot_surfaces.py \
    -c1 $c1 -c2 $c2 -t $task || { echo "Plotting script failed"; exit 1; }

echo "Script completed"
