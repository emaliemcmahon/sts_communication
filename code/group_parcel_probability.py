import argparse
import os
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf_stat_map
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import SurfaceImage
from nilearn.image import smooth_img
from utils.mri import vol2surf
from nilearn.masking import intersect_masks, apply_mask

parcel_name = {'communicate': {'face_third+face_first+com_phy+com_ind-face_noncom+phy+ind': 'communication_parcel',
                               'face_third+face_noncom-object': 'face_parcel',
                               'body-object': 'body_parcel'},
               'pointlight': {'interact-noninteract': 'social_interact_parcel'},
               'tom': {'belief-photo': 'tom_parcel'}}

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
        Path(f'{self.out_path}').mkdir(parents=True, exist_ok=True)

    def plot_surface(self, prob_img):
        """Plot the probability map on the right lateral surface"""
        print('Creating surface plot for right hemisphere...')
        
        # Convert volume to surface
        fsaverage_meshes = load_fsaverage(mesh="fsaverage")
        fsaverage_sulcal = load_fsaverage_data(
            mesh="fsaverage",
            data_type="sulcal",
            mesh_type="inflated",
        )
        
    def plot_surface(self, img, threshold=0.1, cmap='hot', ylabel='Probability', filename_suffix=''):
        """Plot the image on the right lateral surface"""
        print(f'Creating surface plot for {ylabel.lower()} (right hemisphere)...')
        surf_img, fsaverage_meshes, fsaverage_sulcal = vol2surf(img)  # Ensure img is in the correct format for surface plotting

        for hemi in ['left', 'right']:
            out_suffix = filename_suffix + f'_surface_{hemi}'
            # Plot right lateral surface
            fig = plot_surf_stat_map(
                stat_map=surf_img,
                surf_mesh=fsaverage_meshes["inflated"],
                hemi=hemi,
                threshold=threshold,
                bg_map=fsaverage_sulcal,
                darkness=None,
                cmap=cmap,
                vmax=0.75#img.get_fdata().max() if ylabel == 'Probability' else None
            )
            
            # Add colorbar label
            cbar = fig.axes[-1]
            cbar.set_ylabel(ylabel, rotation=270, labelpad=20)
            
            # Make background transparent
            fig.patch.set_alpha(0)
            for ax in fig.axes:
                ax.patch.set_alpha(0)
            
            # Save the plot
            plot_path = f'{self.out_path}/{self.parcel_name}{out_suffix}.png'
            fig.savefig(plot_path, transparent=True, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f'Saved surface plot: {plot_path}')


    def load_parcel_masks(self):
        parcel_path = f'{self.derivatives_path}/parcels-MNI152NLin2009cAsym/*anatSTS.nii.gz'
        parcel_files = sorted(glob(parcel_path))
        combined_parcel = intersect_masks(parcel_files, threshold=0, connected=False)
        return combined_parcel

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
        
        sts_mask = self.load_parcel_masks()

        # Sum all masks
        for mask_file in tqdm(self.mask_files, desc='Summing masks'):
            mask_img = nib.load(mask_file)
            masked_mask_img = intersect_masks([mask_img, sts_mask], threshold=1)
            mask_sum += masked_mask_img.get_fdata()
        
        # Create probability map (sum / n_subjects)
        probability_map = mask_sum / n_subjects
        
        # Create thresholded mask (probability >= 0.5)
        thresholded_mask = np.where(probability_map >= 0.25, 1, 0).astype(np.int32)
        
        # Create nibabel images
        prob_img = nib.Nifti1Image(probability_map, affine)
        thresh_img = nib.Nifti1Image(thresholded_mask, affine)
        
        # # Save results
        # nib.save(prob_img, f'{self.out_path}/{self.parcel_name}_probability.nii.gz')
        # nib.save(thresh_img, f'{self.out_path}/{self.parcel_name}_mask-thresholded.nii.gz')
        
        print(f'Saved probability map and thresholded mask for {self.parcel_name}')
        print(f'Probability map range: {probability_map.min():.3f} - {probability_map.max():.3f}')
        print(f'Number of voxels in thresholded mask: {thresholded_mask.sum()}')
        
        # Create surface plot
        self.plot_surface(prob_img, threshold=0.05)
        # self.plot_surface(thresh_img, threshold=0.5, cmap='Reds', ylabel='Mask', filename_suffix='_mask')


def main():
    parser = argparse.ArgumentParser(description='Create probability maps from first-level contrast masks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='interact',
                         help='The first condition for the contrast')
    parser.add_argument('--condition_two', '-c2', type=str, default='noninteract',
                         help='The second condition for the contrast')
    parser.add_argument('--task_label', '-t', type=str, default='pointlight',
                         help='Task label (communicate, tom, pointlight)')
    args = parser.parse_args()

    processor = GroupParcelProbability(args)
    processor.create_probability_maps()


if __name__ == '__main__':
    main()