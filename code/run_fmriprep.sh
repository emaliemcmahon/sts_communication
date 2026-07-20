#!/bin/bash
# Preprocessing call used to generate `derivatives/fmriprep/` in this dataset.
#
# This is the exact fMRIPrep 24.1.1 command that was invoked (once per
# participant). It is preserved here for reproducibility. It is NOT wrapped in
# a scheduler-specific launcher (e.g. SLURM); adapt it to your local compute
# environment (`sbatch`, `qsub`, `docker run`, etc.) as needed.
#
# Requirements
#   - Apptainer / Singularity >= 1.1
#   - fMRIPrep 24.1.1 container image (fmriprep-24.1.1.simg)
#   - A valid FreeSurfer license file
#
# Usage
#   ./code/run_fmriprep.sh <subject_label>
# Example
#   ./code/run_fmriprep.sh 01
set -euo pipefail

sub=$1
# Zero-pad single-digit subject IDs
[[ ${#sub} -eq 1 ]] && sub="0${sub}"

# --- Paths (edit for your environment) ---------------------------------------
bids_dir=$(pwd)                              # BIDS root (this repository)
output_dir=${bids_dir}/derivatives/fmriprep  # fMRIPrep output directory
singularity_dir=${HOME}/singularity          # directory holding fmriprep-24.1.1.simg
fs_license_file=${singularity_dir}/freesurfer_license.txt

# --- Environment modules (edit / drop for your environment) ------------------
module load gcc/12.2.0
module load apptainer/1.1.9

# --- Run fMRIPrep 24.1.1 -----------------------------------------------------
cd "${singularity_dir}"
singularity run --cleanenv -B "${bids_dir}" \
    fmriprep-24.1.1.simg \
    "${bids_dir}" "${output_dir}" participant \
    --participant-label "${sub}" \
    --n_cpus 22 --omp-nthreads 8 \
    --output-space T1w MNI152NLin2009cAsym \
    --fs-license-file "${fs_license_file}" \
    --clean-workdir
