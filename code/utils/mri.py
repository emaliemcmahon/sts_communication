from glob import glob

from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import SurfaceImage, vol_to_surf
import numpy as np
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_glass_brain
import matplotlib.pyplot as plt
import nibabel as nib


roi_size = {'TPJ': .1,
            'EBA': .1, 'fSTS': .1, 'FFA': .1, 'SI-STS': .05,
            'EVC': 0.05, 'MT': 0.1}


roi_parc = {'SI-STS': 'anatSTS'}


def roi_switcher(roi):
    if roi in list(roi_parc.keys()):
        return roi_parc[roi]
    else:
        return roi
    

def load_brain_mask(fmriprep_path, subject, task_label, space_label):
    """
    Whole-brain analysis mask for one subject/task: the intersection of
    each run's fmriprep brain mask, matching the mask nilearn_glm.py and
    nilearn_glm_runwise.py fit their GLM with (not saved separately to disk).
    """
    mask_files = sorted(glob(f'{fmriprep_path}/sub-{subject}/ses-01/func/'
                              f'*task-{task_label}*{space_label}*brain_mask.nii.gz'))
    if not mask_files:
        raise FileNotFoundError(f'No brain masks found for sub-{subject} task-{task_label} '
                                 f'space-{space_label}')
    masks = [nib.load(mask_file) for mask_file in mask_files]
    return intersect_masks(masks)


def info2vars(model_info):
    (models, imgs, events, confounds) = model_info
    return models[0], imgs[0], events[0], confounds[0]


def vol2surf_int(stat_img, output_int=1, fsaverage_meshes=None, fsaverage_sulcal=None):
    """
    Project volume data to surface with depth sampling and binarize.
    
    Parameters
    ----------
    stat_img : nib.Nifti1Image
        Statistical image in volume space
    output_int : int
        Integer value to assign to binarized ROI
    fsaverage_meshes : dict
        Dict with keys 'white_left', 'pial_left', 'white_right', 'pial_right'
        containing paths to mesh files
    fsaverage_sulcal : str
        Path to sulcal depth map
    
    Returns
    -------
    surf_img : SurfaceImage
        Projected and binarized surface image
    """
    from nilearn.surface import SurfaceImage
    
    if fsaverage_meshes is None:
        raise ValueError("fsaverage_meshes dict must be provided")
    
    vol_to_surf_kwargs = {
        "kind": "depth", 
        "depth": [0, 0.5, 1], 
        "interpolation": "nearest_most_frequent"
    }
    
    # Create mesh dict for both hemispheres (pial surface)
    mesh = {
        "left": fsaverage_meshes["pial_left"],
        "right": fsaverage_meshes["pial_right"]
    }
    
    # Create inner_mesh dict for both hemispheres (white matter surface)
    inner_mesh = {
        "left": fsaverage_meshes["white_left"],
        "right": fsaverage_meshes["white_right"]
    }
    
    # Use SurfaceImage.from_volume which handles both hemispheres automatically
    surf_img = SurfaceImage.from_volume(
        mesh=mesh,
        volume_img=stat_img,
        inner_mesh=inner_mesh,
        **vol_to_surf_kwargs
    )
    
   # Binarize the surface data
    # Access .parts dict to get left/right data
    surf_img.data.parts['left'] = (surf_img.data.parts['left'] > 0.5).astype(np.int32) * output_int
    surf_img.data.parts['right'] = (surf_img.data.parts['right'] > 0.5).astype(np.int32) * output_int

    return surf_img
    


def vol2surf(stat_img):
    fsaverage_meshes = load_fsaverage(mesh="fsaverage")
    fsaverage_sulcal = load_fsaverage_data(
        mesh="fsaverage",
        data_type="sulcal",
        mesh_type="inflated",
    )

    vol_to_surf_kwargs={"kind": "depth", "depth": [0, 0.5, 1], 
                        "interpolation": "linear"}
    surf_img = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        inner_mesh=fsaverage_meshes["white_matter"],
        volume_img=stat_img,
        **vol_to_surf_kwargs
            )
    return surf_img, fsaverage_meshes, fsaverage_sulcal


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
    list of int
        Runs that do not exceed the threshold for removed frames.
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


def parse_contrast(model, c1, c2):
    """
    Parse contrast strings and build contrast vector for GLM.

    Parameters
    ----------
    model : FirstLevelModel
        The fitted first-level GLM model.
    c1 : str
        Condition string for positive contrast (e.g., 'face_third+face_first').
    c2 : str
        Condition string for negative contrast (e.g., 'face_noncom').

    Returns
    -------
    numpy.ndarray
        Contrast vector with weights for each regressor.
    """
    columns = list(model.design_matrices_[0].columns)
    contrast = np.zeros(len(columns))

    # Parse condition one (positive weights)
    cond1_averaging = c1.split('+')
    for c in cond1_averaging:
        c = c.strip()
        if '*' in c:
            # Parse weighted contrast
            weight, cond = c.split('*')
            weight = float(weight.strip())
            cond = cond.strip()
        else:
            # No weight specified, use equal weighting
            weight = 1/len(cond1_averaging)
            cond = c
        contrast[columns.index(cond)] = weight

    # Parse condition two (negative weights)
    if c2 is not None:
        cond2_averaging = c2.split('+')
        for c in cond2_averaging:
            c = c.strip()
            if '*' in c:
                # Parse weighted contrast
                weight, cond = c.split('*')
                weight = float(weight.strip())
                cond = cond.strip()
            else:
                # No weight specified, use equal weighting
                weight = 1/len(cond2_averaging)
                cond = c
            contrast[columns.index(cond)] = -weight

    return contrast


def selective_mask_img(mask_file, img_file, keep_prop=0.1, debug_output=None, return_nifti=True):
    """
    Create a new mask by selecting top positive voxels within a parcel.
    Number of voxels to keep is based on total parcel size.
    
    Args:
        mask_file: Path to binary mask NIfTI file
        img_file: Path to reference image NIfTI file
        keep_prop: Proportion of total parcel voxels to keep (0-1)
        debug_output: Path to save debug plots (None to skip saving)
        return_nifti: If True, return NIfTI image; if False, return numpy array
        
    Returns:
        New NIfTI image or numpy array with selected voxels
    """
    # Load data with sanity checks
    mask = nib.load(mask_file)
    img = nib.load(img_file)
    if debug_output is not None:
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
    if debug_output is not None:
        print(f"\n=== MASK VALIDATION ===")
        print(f"Unique mask values: {unique_mask_vals}")
    
    if len(unique_mask_vals) > 2:
        print("WARNING: Mask appears non-binary - thresholding at 0.5")
        mask_data = (mask_data > 0.5).astype(np.int8)
    
    # Calculate parcel information
    parcel_size = np.sum(mask_data > 0)
    voxels_to_keep = int(parcel_size * keep_prop)
    
    if debug_output is not None:
        print(f"\n=== VOXEL SELECTION ===")
        print(f"Parcel size: {parcel_size} voxels")
        print(f"Attempting to select top {voxels_to_keep} positive voxels ({keep_prop*100:.1f}% of parcel)")
        
    if voxels_to_keep == 0:
        raise ValueError("No voxels to select - check your mask and keep_prop")
    
    # Get positive voxels within mask
    positive_voxels_mask = (mask_data > 0) & (img_data > 0)
    positive_voxel_indices = np.where(positive_voxels_mask.flatten())[0]
    positive_voxel_values = img_data.flatten()[positive_voxel_indices]
    
    if debug_output is not None:
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
    if debug_output is not None:
        actual_voxels_kept = np.sum(new_mask)
        print(f"\n=== VERIFICATION ===")
        print(f"Requested voxels: {voxels_to_keep} | Selected voxels: {actual_voxels_kept}")
    
    # Create output image
    if return_nifti:
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
    else:
        return new_mask.astype(np.int8)
