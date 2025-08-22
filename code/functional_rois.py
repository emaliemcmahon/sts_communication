import argparse
import shutil
from pathlib import Path
import numpy as np
from itertools import product
import nibabel as nib
from nilearn.plotting import plot_glass_brain, view_img_on_surf
import matplotlib.pyplot as plt


task_rois = {'pointlight': {'interactMinusNoninteract': ['SI-STS']},
             'eploc': {'emotionalMinusPhysical': ['TPJ']},
             'tom': {'beliefMinusPhoto': ['TPJ']},
             'communicate': {'bodyMinusObject': ['EBA'],
                            '05faceThirdPlus05FaceNoncomMinusObject': ['fSTS', 'FFA']}}

roi_size = {'SI-STS': .05, 'TPJ': .1, 
            'EBA': .1, 'fSTS': .1, 'FFA': .1}

roi_parc = {'SI-STS': 'anatSTS'}

def roi_switcher(roi):
    if roi in list(roi_parc.keys()):
        return roi_parc[roi]
    else:
        return roi


def selective_mask_img(mask_file, img_file, keep_prop=0.1, debug_output=None):
    """
    Create a new mask by selecting top positive voxels within a parcel.
    Number of voxels to keep is based on total parcel size.
    
    Args:
        mask_file: Path to binary mask NIfTI file
        img_file: Path to reference image NIfTI file
        keep_prop: Proportion of total parcel voxels to keep (0-1)
        debug_output: Path to save debug plots (None to skip saving)
        
    Returns:
        New NIfTI image with selected voxels
    """
    # Load data with sanity checks
    mask = nib.load(mask_file)
    img = nib.load(img_file)
    
    print("\n=== INPUT VALIDATION ===")
    print(f"Image shape: {img.shape} | Mask shape: {mask.shape}")
    
    if img.shape != mask.shape:
        raise ValueError("Image and mask must have identical dimensions")
    
    # Initialize debug plot if needed
    if debug_output is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        plot_glass_brain(img, title="Original Image", 
                        axes=axes[0,0], 
                        plot_abs=False,
                        colorbar=True)
        plot_glass_brain(mask, title="Original Mask", axes=axes[0,1])
    
    # Get data arrays
    mask_data = mask.get_fdata()
    img_data = img.get_fdata()
    
    # Check mask is binary
    unique_mask_vals = np.unique(mask_data)
    print(f"\n=== MASK VALIDATION ===")
    print(f"Unique mask values: {unique_mask_vals}")
    
    if len(unique_mask_vals) > 2:
        print("WARNING: Mask appears non-binary - thresholding at 0.5")
        mask_data = (mask_data > 0.5).astype(np.int8)
    
    # Calculate parcel information
    parcel_size = np.sum(mask_data > 0)
    voxels_to_keep = int(parcel_size * keep_prop)
    
    print(f"\n=== VOXEL SELECTION ===")
    print(f"Parcel size: {parcel_size} voxels")
    print(f"Attempting to select top {voxels_to_keep} positive voxels ({keep_prop*100:.1f}% of parcel)")
    
    if voxels_to_keep == 0:
        raise ValueError("No voxels to select - check your mask and keep_prop")
    
    # Get positive voxels within mask
    positive_voxels_mask = (mask_data > 0) & (img_data > 0)
    positive_voxel_indices = np.where(positive_voxels_mask.flatten())[0]
    positive_voxel_values = img_data.flatten()[positive_voxel_indices]
    
    print(f"Found {len(positive_voxel_values)} positive voxels in parcel")
    print(f"Response range: {np.min(positive_voxel_values):.2f} to {np.max(positive_voxel_values):.2f}")
    
    # Determine how many we can actually select (up to voxels_to_keep)
    actual_voxels_to_select = min(voxels_to_keep, len(positive_voxel_values))
    
    if actual_voxels_to_select < voxels_to_keep:
        print(f"WARNING: Only selecting {actual_voxels_to_select} voxels (not enough positive values)")
    
    # Select top voxels
    if actual_voxels_to_select > 0:
        sorted_indices = np.argsort(positive_voxel_values)[::-1][:actual_voxels_to_select]
        selected_flat_indices = positive_voxel_indices[sorted_indices]
    else:
        selected_flat_indices = np.array([], dtype=int)
    
    # Create new mask
    new_mask_flat = np.zeros(img_data.size)
    new_mask_flat[selected_flat_indices] = 1
    new_mask = new_mask_flat.reshape(img_data.shape)
    
    # Verification
    actual_voxels_kept = np.sum(new_mask)
    print(f"\n=== VERIFICATION ===")
    print(f"Requested voxels: {voxels_to_keep} | Selected voxels: {actual_voxels_kept}")
    
    # Create output image
    output_img = nib.Nifti1Image(new_mask.astype(np.int8), img.affine)
    
    # Visualize results
    if debug_output is not None:
        plot_glass_brain(output_img, 
                        title=f"Selected {actual_voxels_kept} voxels", 
                        axes=axes[1,0])
        
        # Plot histogram
        axes[1,1].hist(positive_voxel_values, bins=50, alpha=0.7, label='All positive voxels')
        if actual_voxels_to_select > 0:
            selected_values = positive_voxel_values[sorted_indices]
            axes[1,1].hist(selected_values, bins=50, alpha=0.7, 
                          label='Selected voxels', color='red')
        axes[1,1].set_title("Response Value Distribution")
        axes[1,1].legend()
        axes[1,1].set_xlabel("Response value")
        axes[1,1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(debug_output)
        plt.close()
    
    return output_img

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
                    output_file = f'{self.out_path}/sub-{self.subject_label}_hemi-{hemi}_roi-{roi}_mask'
                    parc_name = roi_switcher(roi)
                    contrast_file = f'{self.glm_path}/sub-{self.subject_label}/sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast}_stat-z_statmap.nii.gz'
                    mask_file = f'{self.parcel_path}/{hemi}{parc_name}.nii.gz'
                    new_mask = selective_mask_img(mask_file, contrast_file, 
                                                  keep_prop=roi_size[roi],
                                                  debug_output=f'{output_file}.pdf')
                    nib.save(new_mask, f'{output_file}.nii.gz')
            else:
                for hemi in ['l', 'r']:
                    parc_file = f'{self.parcel_path}/{hemi}{self.task_label}.nii.gz'
                    output_file = f'{self.out_path}/sub-{self.subject_label}_hemi-{hemi}_task-{self.task_label}_mask'
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