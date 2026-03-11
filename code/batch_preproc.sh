#!/bin/bash -l

#SBATCH
#SBATCH --partition=ou_bcs_normal
#SBATCH --job-name=fmriprep
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%x_%A.out
#SBATCH --mail-user=emaliem@mit.edu

sub=$1

[[ ${#sub} -eq 1 ]] && sub="0${sub}"

username=$(whoami)
scratch_top=/orcd/data/ngk/001/users/${username}
project=sts_communication
singularity_dir=${scratch_top}/singularity
bids_dir=${scratch_top}/${project}
fs_license_file=${scratch_top}/singularity/freesurfer_license.txt
output_dir=${bids_dir}/derivatives/fmriprep

cd $singularity_dir

module load gcc/12.2.0
module load apptainer/1.1.9

singularity run --cleanenv -B $scratch_top \
fmriprep-24.1.1.simg \
$bids_dir $output_dir participant \
--participant-label $sub \
--n_cpus 22 --omp-nthreads 8 \
--output-space T1w MNI152NLin2009cAsym \
--fs-license-file $fs_license_file \
--clean-workdir