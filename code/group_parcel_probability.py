import argparse
import os
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
import nibabel as nib

parcel_name = {'communicate': {'face_third+face_first+com_phy+com_ind-face_noncom+phy+ind': 'communication_parcel'},
               'pointlight': {'interact-noninteract': 'social_interact_parcel'}}

class GroupParcelProbability:
    def __init__(self, args):
        self.process = 'GroupParcelProbability'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.glm_path = f'{self.derivatives_path}/NilearnGLM'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.condition_one = args.condition_one
        self.condition_two = args.condition_two
        self.contrast_name = f'{self.condition_one}-{self.condition_two}'
        self.task_label = args.task_label
        
        # Get parcel name from dictionary, fallback to contrast_name
        self.parcel_name = parcel_name.get(self.task_label, {}).get(self.contrast_name, self.contrast_name)
        
        # Find all subjects that have the mask file
        self.mask_files = sorted(glob(f'{self.glm_path}/sub-*/task-{self.task_label}/contrast-{self.contrast_name}_mask.nii.gz'))
        self.subjs = [Path(f).parent.parent.name.replace('sub-', '') for f in self.mask_files]
        
        print(f'Found {len(self.mask_files)} subjects with mask for contrast {self.contrast_name} for task {self.task_label}: {self.subjs}')
        print(f'Using parcel name: {self.parcel_name}')
        Path(f'{self.out_path}/sub-group').mkdir(parents=True, exist_ok=True)

    def create_probability_maps(self):
        print('Creating probability maps...')
        n_subjects = len(self.mask_files)
        
        if n_subjects == 0:
            print('No mask files found. Exiting.')
            return
        
        # Load first mask to get shape and affine
        first_mask = nib.load(self.mask_files[0])
        affine = first_mask.affine
        shape = first_mask.shape
        
        # Initialize sum array
        mask_sum = np.zeros(shape, dtype=np.float32)
        
        # Sum all masks
        for mask_file in tqdm(self.mask_files, desc='Summing masks'):
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata()
            mask_sum += mask_data
        
        # Create probability map (sum / n_subjects)
        probability_map = mask_sum / n_subjects
        
        # Create thresholded mask (probability >= 0.5)
        thresholded_mask = np.where(probability_map >= 0.5, 1, 0).astype(np.int32)
        
        # Create nibabel images
        prob_img = nib.Nifti1Image(probability_map, affine)
        thresh_img = nib.Nifti1Image(thresholded_mask, affine)
        
        # Save results
        nib.save(prob_img, f'{self.out_path}/sub-group/{self.parcel_name}_probability.nii.gz')
        nib.save(thresh_img, f'{self.out_path}/sub-group/{self.parcel_name}_mask-thresholded.nii.gz')
        
        print(f'Saved probability map and thresholded mask for {self.parcel_name}')
        print(f'Probability map range: {probability_map.min():.3f} - {probability_map.max():.3f}')
        print(f'Number of voxels in thresholded mask: {thresholded_mask.sum()}')


def main():
    parser = argparse.ArgumentParser(description='Create probability maps from first-level contrast masks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, required=True,
                         help='The first condition for the contrast')
    parser.add_argument('--condition_two', '-c2', type=str, required=True,
                         help='The second condition for the contrast')
    parser.add_argument('--task_label', '-t', type=str, required=True,
                         help='Task label (communicate, tom, pointlight)')
    args = parser.parse_args()

    processor = GroupParcelProbability(args)
    processor.create_probability_maps()


if __name__ == '__main__':
    main()