import os
import nibabel as nib
import numpy as np

from utils.mri import roi_size, roi_switcher, selective_mask_img, winner_take_all


# (task, contrast_name) used to define each functional ROI, matching the
# whole-brain NilearnGLM contrasts described in methods.md ("Functional ROIs")
ROI_DEFINING_CONTRAST = {
    'SI-STS': ('pointlight', 'interact-noninteract'),
    'TPJ': ('tom', 'belief-photo'),
    'fSTS': ('communicate', 'face_third+face_noncom-object'),
    'FFA': ('communicate', 'face_third+face_noncom-object'),
    'EBA': ('communicate', 'body-object'),
    'EVC': ('communicate', 'com_phy+phy+com_ind+ind+face_third+face_first+face_noncom+object+body'),
    'MT': ('communicate', 'com_phy+phy+com_ind+ind+face_third+face_first+face_noncom+object+body'),
}

# Task each condition's beta (effect-size vs. fixation) map was estimated in
CONDITION_TASK = {
    'interact': 'pointlight',
    'noninteract': 'pointlight',
    'com_ind': 'communicate',
    'ind': 'communicate',
    'com_phy': 'communicate',
    'phy': 'communicate',
    'face_first': 'communicate',
    'face_third': 'communicate',
    'face_noncom': 'communicate',
    'body': 'communicate',
    'object': 'communicate',
    'belief': 'tom',
    'photo': 'tom',
}


def roi_mask(glm_path, parcel_path, subject, roi, hemi):
    """
    Boolean voxel mask (flattened) for one ROI/hemisphere/subject, defined as
    the top-responding voxels within the ROI's anatomical/functional parcel
    for that ROI's whole-brain (all-runs-pooled) defining contrast.

    Returns None if the defining contrast or parcel file is missing.
    """
    task, contrast_name = ROI_DEFINING_CONTRAST[roi]
    contrast_file = os.path.join(glm_path, f'sub-{subject}', f'task-{task}',
                                  f'contrast-{contrast_name}_stat-tmap.nii.gz')
    mask_file = os.path.join(parcel_path, f'{hemi}{roi_switcher(roi)}.nii.gz')
    if not (os.path.exists(contrast_file) and os.path.exists(mask_file)):
        return None

    mask = selective_mask_img(mask_file, contrast_file, keep_prop=roi_size[roi],
                               return_nifti=False)
    return mask.astype(bool).flatten()


def roi_masks_resolved(glm_path, parcel_path, subject, hemi, rois=None, method='winner_take_all'):
    """
    Boolean voxel masks (flattened) for multiple ROIs in one hemisphere, with
    voxels independently selected by more than one ROI's roi_mask() resolved
    via `method`, following video_sentence_analysis's
    first_level_univariate/build_froi_masks.py:
      - 'winner_take_all' (default): each contested voxel is kept only for
        whichever competing ROI has the higher t-stat in its own defining
        contrast (ROI_DEFINING_CONTRAST) at that voxel.
      - 'drop': contested voxels are removed from every claiming ROI.
      - None: no resolution (equivalent to calling roi_mask() per ROI).

    ROIs missing their contrast/parcel file are silently dropped (matching
    roi_mask's None-return behavior).

    Returns
    -------
    dict[str, np.ndarray] mapping roi -> flattened boolean mask, only for
    ROIs whose files were found.
    """
    if rois is None:
        rois = list(ROI_DEFINING_CONTRAST.keys())

    masks, stats = {}, {}
    for roi in rois:
        mask = roi_mask(glm_path, parcel_path, subject, roi, hemi)
        if mask is None:
            continue
        task, contrast_name = ROI_DEFINING_CONTRAST[roi]
        contrast_file = os.path.join(glm_path, f'sub-{subject}', f'task-{task}',
                                      f'contrast-{contrast_name}_stat-tmap.nii.gz')
        masks[roi] = mask
        stats[roi] = nib.load(contrast_file).get_fdata().flatten()

    if not masks or method is None:
        return masks
    if method == 'drop':
        mask_stack = np.stack(list(masks.values()), axis=0)
        contested = mask_stack.sum(axis=0) > 1
        return {roi: m & ~contested for roi, m in masks.items()}
    if method == 'winner_take_all':
        return winner_take_all(masks, stats)
    raise ValueError(f'Unknown overlap resolution method: {method}')


def condition_pattern(glm_path, subject, condition, voxels):
    """
    Beta (effect-size vs. fixation) pattern for one condition, restricted to
    `voxels` (a flattened boolean mask from `roi_mask`).

    Returns None if the condition's beta map is missing.
    """
    task = CONDITION_TASK[condition]
    beta_file = os.path.join(glm_path, f'sub-{subject}', f'task-{task}',
                              f'contrast-{condition}_stat-beta.nii.gz')
    if not os.path.exists(beta_file):
        return None

    beta = nib.load(beta_file).get_fdata().flatten()
    return beta[voxels]
