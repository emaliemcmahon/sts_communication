#!/bin/bash -l

#SBATCH --job-name=surface_plots
#SBATCH --partition=mit_normal
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A.out

conda activate nilearn

c1=$1
c2=$2

USER=$(whoami)
top_dir="/scratch/ngk/001/users/${USER}/sts_communication"

echo "Plotting surfaces for contrast: $c1 vs $c2"

python ${top_dir}/code/plot_surfaces.py -d $top_dir -c1 $c1 -c2 $c2