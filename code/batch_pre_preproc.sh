#!/bin/bash -l

#SBATCH
#SBATCH --job-name=unpack
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=24
#SBATCH --exclude=node064

par=$1
src=$2

ses=01

scratch_top=/orcd/data/ngk/001/users/emaliem/sts_communication/

# Convert the DICOM to NII and convert to BIDS format
conda activate dcm2bids
dcm2bids -p ${par} -s ${ses} \
  -d sourcedata/${src}/ \
  -c code/dcm2bids.config -l DEBUG

# Remove runs that were aborted
conda deactivate
conda activate nilearn
python code/rm_aborted_runs.py \
  -d ${scratch_top}/sub-${par}/ses-${ses}/func

# Plot the anatomical image
python code/visualize_anatomy.py \
  -f sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz \
  -i orig_anat.jpg

# Deface the anatomical image
conda deactivate
conda activate pydeface_env
module load community-modules
module load fsl
pydeface sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz \
  --outfile sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w-defaced.nii.gz

# Plot the anatomical image after defacing
conda deactivate
conda activate nilearn
python code/visualize_anatomy.py \
  -f sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w-defaced.nii.gz \
  -i defaced_anat.jpg

# Check that the anatomy was successfully defaced 
read -p "Check that the anatomy was successfully defaced. Press enter to continue: "
rm sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz
mv sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w-defaced.nii.gz \
  sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz
rm *jpg