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
scratch_top=/orcd/data/ngk/001/users/emaliem/sts_communication
behavior_top=/orcd/data/ngk/001/users/emaliem/communication_exp

# Copy the DICOMs to the current directory
module use /orcd/compute/bcs/001/modulefiles
module add slicer
module add mrimages
GetDicoms $src ${scratch_top}/sourcedata/
mv ${scratch_top}/sourcedata/${src}/dicom/* ${scratch_top}/sourcedata/${src}/
rm -rf ${scratch_top}/sourcedata/${src}/dicom

# Convert the DICOM to NII and convert to BIDS format
conda activate dcm2bids
dcm2bids -p ${par} -s ${ses} \
  -d ${scratch_top}/sourcedata/${src}/ \
  -c code/dcm2bids.config -l DEBUG

# Remove runs that were aborted
conda deactivate
conda activate nilearn
python code/rm_aborted_runs.py \
  -d ${scratch_top}/sub-${par}/ses-${ses}/func

#Copy the behavioral data to the current directory
cp ${behavior_top}/sts_communication_experiment/data/subj0${par}/bids/* sub-${par}/ses-${ses}/func/
python ${behavior_top}/tomloc/write_event_files.py \
  --subj ${par} \
  --behav-dir ${behavior_top}/tomloc/behavioural \
  --bids-root ${scratch_top}
python ${behavior_top}/point_light_social/para2bids.py \
  --input_subj sub-${par} \
  --output_subj ${par} \
  --input_dir ${behavior_top}/point_light_social/data/sts_interaction \
  --output_dir ${scratch_top}/sub-${par}/ses-${ses}/func \

# Plot the anatomical image
python code/visualize_anatomy.py \
  -f ${scratch_top}/sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz \
  -i orig_anat.jpg

# Deface the anatomical image
conda deactivate
conda activate pydeface_env
module load community-modules
module load fsl
pydeface ${scratch_top}/sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz \
  --outfile ${scratch_top}/sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w-defaced.nii.gz

# Plot the anatomical image after defacing
conda deactivate
conda activate nilearn
python code/visualize_anatomy.py \
  -f ${scratch_top}/sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w-defaced.nii.gz \
  -i defaced_anat.jpg

# Check that the anatomy was successfully defaced 
read -p "Check that the anatomy was successfully defaced. Press enter to continue: "
rm sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz
mv sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w-defaced.nii.gz \
  sub-${par}/ses-${ses}/anat/sub-${par}_ses-${ses}_T1w.nii.gz
rm *jpg