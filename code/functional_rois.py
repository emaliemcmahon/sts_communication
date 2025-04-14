import argparse
import shutil
from pathlib import Path
import numpy as np
from itertools import product
import nibabel as nib
from nilearn.plotting import plot_glass_brain, view_img_on_surf


task_rois = {'pointlight': {'interactMinusNoninteract': ['SI-STS']},
             'eploc': {'emotionalMinusPhysical': ['TPJ']},
             'tom': {'beliefMinusPhoto': ['TPJ']}}

roi_size = {'SI-STS': .05, 'TPJ': .1}

roi_parc = {'SI-STS': 'anatSTS'}

def roi_switcher(roi):
    if roi in list(roi_parc.keys()):
        return roi_parc[roi]
    else:
        return roi


def mask_img(mask, img, keep_prop=.1):
    print(f'Starting shape: {img.shape}')
    print(f'Starting with {img.size} total voxels')
    
    # Flatten the mask and image
    mask_flat = mask.flatten().astype(bool)
    img_flat = img.flatten()
    
    # Calculate the number of voxels to keep
    voxels_to_keep = int(np.sum(mask_flat) * keep_prop)
    print(f'Parcel size: {np.sum(mask_flat)}')
    print(f'Selecting {voxels_to_keep} voxels from parcel')
    
    # Get the response values within the mask
    parc_response = img_flat[mask_flat]
    
    # Get the indices of the highest response values
    idx = np.argsort(parc_response)[::-1]
    
    # Get the indices in the original flattened image space
    original_indices = np.where(mask_flat)[0][idx[:voxels_to_keep]]
    
    # Convert flattened indices back to 3D coordinates
    coords_3d = np.unravel_index(original_indices, shape=img.shape)
    coords_3d = np.column_stack(coords_3d)  # Stack into (N, 3) array
    
    # Create a new boolean array in the 3D space of the original mask
    new_mask = np.zeros(img.shape)  # Initialize with False
    new_mask[coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]] = 1  # Set selected voxels to True
    
    # Verify that the correct number of voxels are 1
    print(f'Number of True values in new_mask: {np.sum(new_mask)} (expected: {voxels_to_keep})')  # Debug
    return new_mask


class FunctionalROIs:
    def __init__(self, args):
        self.process = 'FunctionalROIs'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.out_path = f'{self.derivatives_path}/{self.process}/sub-{self.subject_label}'
        self.glm_path = f'{self.derivatives_path}/NilearnGLM'
        self.parcel_path = f'{self.derivatives_path}/parcels-{self.space_label}'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def define_rois(self):
        for contrast, rois in task_rois[self.task_label].items():
            if contrast != 'None': 
                for hemi, roi in product(['l', 'r'], rois):
                    print(f'{roi=}')
                    output_file = f'{self.out_path}/sub-{self.subject_label}_{hemi}{roi}'
                    parc_name = roi_switcher(roi)
                    contrast_file = f'{self.glm_path}/sub-{self.subject_label}/sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast}_stat-effect_statmap.nii.gz'
                    mask_arr = nib.load(f'{self.parcel_path}/{hemi}{parc_name}.nii.gz').get_fdata()
                    img = nib.load(contrast_file)
                    img_arr = img.get_fdata()
                    new_mask = mask_img(mask_arr, img_arr, keep_prop=roi_size[roi])
                    new_mask_img = nib.Nifti1Image(new_mask, img.affine,
                                                nib.Nifti1Header())
                    plot_glass_brain(new_mask_img,
                                    colorbar=True,
                                    threshold=0.1,
                                    display_mode="x",
                                    output_file=f'{output_file}.pdf')
                    view = view_img_on_surf(new_mask_img,
                                            threshold=0.1)
                    view.save_as_html(f'{output_file}.html')
                    nib.save(new_mask_img, f'{output_file}.nii.gz')
            else:
                for hemi in ['l', 'r']:
                    parc_file = f'{self.parcel_path}/{hemi}{self.task_label}.nii.gz'
                    output_file = f'{self.out_path}/sub-{self.subject_label}_{hemi}{self.task_label}'
                    shutil.copyfile(parc_file, f'{output_file}.nii.gz')

                    img = nib.load(f'{output_file}.nii.gz')
                    plot_glass_brain(img,
                                    colorbar=True,
                                    threshold=0.1,
                                    display_mode="x",
                                    output_file=f'{output_file}.pdf')
                    view = view_img_on_surf(img,
                                            threshold=0.1)
                    view.save_as_html(f'{output_file}.html')

def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=str, default='05',
                         help='Subject for the GLM')
    parser.add_argument('--task_label', '-t', type=str, default='eploc',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = FunctionalROIs(args)
    processor.define_rois()

if __name__ == '__main__':
    main()