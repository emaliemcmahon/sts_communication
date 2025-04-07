#!/bin/bash -l

#SBATCH
#SBATCH --job-name=fmriprep
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=48
#SBATCH --exclude=node072
#SBATCH --exclusive

sid=$1

scratch_top=/mindhive/nklab3/users/emaliem
singularity_dir=${scratch_top}/singularity
bids_dir=${scratch_top}/communicate_pilot
fs_license_file=${scratch_top}/singularity/freesurfer_license.txt
output_dir=${bids_dir}/derivatives/fmriprep

cd $singularity_dir

singularity run --cleanenv -B /mindhive \
fmriprep-24.1.1.simg \
$bids_dir $output_dir participant \
--participant-label $sid \
--fs-license-file $fs_license_file \
--n_cpus 24 --omp-nthreads 6 \
--output-space T1w MNI152NLin2009cAsym \
--bold2anat-dof 6
# --bold2anat-dof 12 \
# --use-syn-sdc \
# --ignore fieldmaps

