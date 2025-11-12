import os
import warnings
import argparse
from glob import glob
from pathlib import Path
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.interfaces.fmriprep import load_confounds
import nibabel as nib
from tqdm import tqdm
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_glass_brain
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


n_groups = {'pointlight': 4, 'tom': 2, 'communicate': 9}

response_contrasts = {'pointlight': ['interact', 'noninteract'],
                      'tom': ['belief', 'photo'],
                      'communicate': ['com_phy', 'com_ind', 'phy', 'ind',
                    'face_third', 'face_first', 'face_noncom',
                    'body', 'object']}

froi_contrasts = {'pointlight': {'interact-noninteract': ['SI-STS']}, 
                'tom': {'belief-photo': ['TPJ']},
                  'communicate': {'body-object': ['EBA'],
                                  '0.5*face_third+0.5*face_noncom-object': ['fSTS', 'FFA'],
                                  'com_phy-phy': ['comphy-STS'],
                                  'com_ind-ind': ['comind-STS']}}

roi_size = {'comphy-STS': .05, 'comind-STS': .05, 'TPJ': .1,
            'EBA': .1, 'fSTS': .1, 'FFA': .1, 'SI-STS': .05}

roi_parc = {'comphy-STS': 'anatSTS',
            'comind-STS': 'anatSTS',
            'SI-STS': 'anatSTS'}


def roi_switcher(roi):
    if roi in list(roi_parc.keys()):
        return roi_parc[roi]
    else:
        return roi


def info2vars(model_info):
    (models, imgs, events, confounds) = model_info
    return models[0], imgs[0], events[0], confounds[0]


def check_motion_filtering(sample_masks, n_trs, threshold=12, one_indexed=False):
    """
    Return a list of runs where more than `threshold` frames were removed.

    Parameters
    ----------
    sample_masks : array-like or list of array-like
        Each element is a numpy array of kept-volume indices (from load_confounds).
        If a single run, can pass a single array directly.
    n_trs : int or list of int
        Total number of volumes (TRs) in each corresponding run.
    threshold : int, default=12
        Number of removed frames above which a run is considered exceeding.
    one_indexed : bool, default=True
        If True, return runs numbered from 1 (run 1, 2, ...). 
        If False, return 0-indexed run indices.

    Returns
    -------
    list of int
        Runs that exceed the threshold for removed frames.
    """
    import numpy as np

    # Normalize to lists
    if not isinstance(sample_masks, (list, tuple)):
        sample_masks = [sample_masks]
    if not isinstance(n_trs, (list, tuple, np.ndarray)):
        n_trs = [n_trs] * len(sample_masks)

    bad_runs = []
    good_runs = []
    for i, (mask, total) in enumerate(zip(sample_masks, n_trs)):
        n_kept = len(mask) if mask is not None else total
        n_removed = total - n_kept
        if n_removed > threshold:
            bad_runs.append(i + 1 if one_indexed else i)
        else:
            good_runs.append(i + 1 if one_indexed else i)   

    return bad_runs, good_runs

def split_into_groups(items, n_groups=3):
    return [items[i::n_groups] for i in range(n_groups)]


def hyphen_to_camel_case(contrast_name):
    """
    Converts a hyphen-separated contrast name into a camelCase-style contrast name.

    Parameters:
    contrast_name (str): The contrast name with hyphens (e.g., 'emotional-physical').

    Returns:
    str: The contrast name in camelCase (e.g., 'emotionalMinusPhysical').
    """
    def replace_symbol(out, symbol, name=None):
        parts = out.split(symbol)   
        part1 =  parts[0]
        part2 = f'{symbol}'.join(parts[1:])
        if name is not None:
            out = part1 + name + part2[0].upper() + part2[1:]
        else:
            out = part1 + part2[0].upper() + part2[1:]
        return out
    out = contrast_name.replace('.', '').replace('*', '')
    for symbol, name in zip(['+', '-', '_'], ['Plus', 'Minus', None]):
        while symbol in out: 
            out = replace_symbol(out, symbol, name)
    return out


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


class NilearnGLMRunwise:
    def __init__(self, args):
        self.process = 'NilearnGLMRunwise'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.parcel_path = f'{self.derivatives_path}/parcels-{self.space_label}'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.TR = 2
        self.frame_threshold = 12
        print(vars(self))

    def load_mask(self):
        mask_files = sorted(glob(f'{self.fmriprep_path}/sub-{self.subject_label}/ses-01/func/*task-{self.task_label}*{self.space_label}*brain_mask.nii.gz'))
        print(mask_files)
        masks = [nib.load(mask_file) for mask_file in mask_files]
        return intersect_masks(masks)

    def glm(self):
        mask = self.load_mask()

        files = sorted(glob(f'{self.fmriprep_path}/sub-{self.subject_label}/ses-*/func/*{self.task_label}*{self.space_label}*bold.nii.gz'))
        print(files)
        
        # Load confounds with motion filtering strategy to get sample_masks
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            confounds_filtered, sample_masks = load_confounds(files, strategy=('motion', 'scrub'), 
                                                        fd_threshold=1, 
                                                        std_dvars_threshold=3, 
                                                        scrub=0,
                                                        motion='basic')

        # Check which runs exceed motion threshold
        n_trs = nib.load(files[0]).shape[-1]
        excluded_runs, included_runs = check_motion_filtering(sample_masks, n_trs,
                                                                threshold=self.frame_threshold)
        print(f'{len(excluded_runs)=}')
        print(f'{len(included_runs)=}')
        for i, m in enumerate(sample_masks):
            if m is not None:
                print(f'Run {i}: {len(m)=}/{n_trs=}')

        # Load model with first_level_from_bids but use our filtered confounds
        model_info = flfb(self.dataset_path,
                            self.task_label,
                            self.space_label,
                            mask_img=mask,
                            sub_labels=[self.subject_label],
                            slice_time_ref=None, # Load from the BIDS data
                            smoothing_fwhm=5.0,
                            img_filters=[("desc", "preproc")],
                            derivatives_folder=self.fmriprep_path,
                            minimize_memory=False, 
                            hrf_model='spm',
                            n_jobs=-1)
        
        # Get model, imgs, and events from BIDS, but use our confounds_filtered
        model, imgs, events, _ = info2vars(model_info)
        confounds = confounds_filtered

        # Shift the time series because fMRIPrep slice time corrects to the middle volume
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
        events_shifted = []
        for event in events: 
            event['onset'] = event['onset'] + 1
            events_shifted.append(event)

        run_groups = split_into_groups(included_runs, n_groups=n_groups[self.task_label])
        for igroup, runs in tqdm(enumerate(run_groups),
                                 total=n_groups[self.task_label], desc='fitting run groups'):
            # Compute the model and contrasts to define the fROIs
            model.fit([imgs[r] for r in included_runs if r not in runs], 
                                [events_shifted[r] for r in included_runs if r not in runs],
                                [confounds[r] for r in included_runs if r not in runs])
            for contrast in froi_contrasts[self.task_label].keys():
                contrast_name = hyphen_to_camel_case(contrast)
                title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast_name}_run-{igroup+1}'
                contrast_file = f'{self.out_path}/sub-{self.subject_label}/{title}.nii.gz'            
                stat_map = model.compute_contrast(contrast, output_type='z_score')
                nib.save(stat_map, contrast_file)

                for hemi, roi in product(['l', 'r'], froi_contrasts[self.task_label][contrast]):
                    output_file = f'{self.out_path}/sub-{self.subject_label}/sub-{self.subject_label}_run-{igroup+1}_{hemi}{roi}'      
                    mask_file = f'{self.parcel_path}/{hemi}{roi_switcher(roi)}.nii.gz'
                    new_mask = selective_mask_img(mask_file, contrast_file, 
                                                  keep_prop=roi_size[roi],
                                                  debug_output=f'{output_file}.pdf')
                    nib.save(new_mask, f'{output_file}.nii.gz')

            # Compute the model and contrasts to estimate the responses
            model.fit([imgs[r] for r in included_runs if r in runs], 
                      [events_shifted[r] for r in included_runs if r in runs],
                      [confounds[r] for r in included_runs if r in runs])
            for contrast in response_contrasts[self.task_label]:
                title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast}_run-{igroup+1}'
                contrast_file = f'{self.out_path}/sub-{self.subject_label}/{title}.nii.gz' 
                stat_map = model.compute_contrast(contrast, output_type='effect_size')
                nib.save(stat_map, contrast_file)
    
    def run(self):
        if not os.path.exists(f'{self.out_path}/sub-{self.subject_label}'):
            Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)
            self.glm()
        else:
            if self.overwrite:
                Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)
                self.glm()
            else:
                print('Output already exists. To re-run pass --overwrite')

def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=str, default='02',
                         help='Subject for the GLM')
    parser.add_argument('--task_label', '-t', type=str, default='pointlight',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    processor = NilearnGLMRunwise(args)
    processor.run()

if __name__ == '__main__':
    main()