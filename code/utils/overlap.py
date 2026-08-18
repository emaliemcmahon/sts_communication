import os
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting as nplot

from utils.mri import vol2surf_int

HEMIS = {'left': 'l', 'right': 'r'}  # display name -> parcel filename prefix
DEFAULT_PERCENT_THRESHOLDS = [0.05, 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]


def load_hemi_parcel_masks(parcel_path, parcel_stem='anatSTS'):
    """
    Left and right parcel masks, kept separate (never pooled across hemispheres).
    Returns (hemi_masks, affine, header) where hemi_masks is
    {'left': bool array, 'right': bool array}.
    """
    hemi_masks = {}
    affine = header = None
    for hemi, prefix in HEMIS.items():
        img = nib.load(os.path.join(parcel_path, f'{prefix}{parcel_stem}.nii.gz'))
        hemi_masks[hemi] = img.get_fdata().astype(bool)
        if affine is None:
            affine, header = img.affine, img.header
    return hemi_masks, affine, header


def top_percent_mask(values, parcel_mask, percent):
    """
    Boolean mask of the top `percent` of parcel voxels by value.
    Only positive values are eligible (no significance threshold applied).
    `percent` is relative to the total number of voxels in the parcel.
    """
    parcel_size = int(parcel_mask.sum())
    n_keep = int(round(parcel_size * percent))
    candidate_idx = np.flatnonzero(parcel_mask & (values > 0))
    if candidate_idx.size == 0 or n_keep == 0:
        return np.zeros_like(parcel_mask, dtype=bool)
    n_keep = min(n_keep, candidate_idx.size)
    order = np.argsort(values.flat[candidate_idx])[::-1][:n_keep]
    keep_idx = candidate_idx[order]
    mask = np.zeros(values.size, dtype=bool)
    mask[keep_idx] = True
    return mask.reshape(values.shape)


def top_percent_mask_bilateral(values, hemi_masks, percent):
    """Union of the per-hemisphere top-`percent` selections (each hemisphere
    ranked independently within its own parcel)."""
    combined = np.zeros(values.shape, dtype=bool)
    for parcel_mask in hemi_masks.values():
        combined |= top_percent_mask(values, parcel_mask, percent)
    return combined


def dice(mask_a, mask_b):
    n_a, n_b = mask_a.sum(), mask_b.sum()
    if n_a + n_b == 0:
        return np.nan
    overlap = np.logical_and(mask_a, mask_b).sum()
    return 2 * overlap / (n_a + n_b)


def plot_bilateral_overlap_surface(mask_a, mask_b, affine, header, fsaverage, fsaverage_meshes,
                                    label_a, label_b, title, out_file):
    """
    Categorical surface plot of two binary volume masks (already unioned across
    hemispheres) on left+right lateral inflated surfaces: label_a-only,
    label_b-only, and their overlap.
    """
    img_a = nib.Nifti1Image(mask_a.astype(np.int8), affine, header)
    img_b = nib.Nifti1Image(mask_b.astype(np.int8), affine, header)
    surf_a = vol2surf_int(img_a, fsaverage_meshes=fsaverage_meshes)
    surf_b = vol2surf_int(img_b, fsaverage_meshes=fsaverage_meshes)

    # Categorical code: 1 = a only, 2 = b only, 3 = overlap
    def categorize(part_a, part_b):
        cat = np.zeros_like(part_a, dtype=float)
        cat[(part_a > 0) & (part_b == 0)] = 1
        cat[(part_a == 0) & (part_b > 0)] = 2
        cat[(part_a > 0) & (part_b > 0)] = 3
        return cat

    left = categorize(surf_a.data.parts['left'], surf_b.data.parts['left'])
    right = categorize(surf_a.data.parts['right'], surf_b.data.parts['right'])
    n_left = left.shape[0]
    plot_data = np.concatenate([left, right])

    colors = {1: (0.20, 0.45, 0.85, 0.85), 2: (0.85, 0.35, 0.15, 0.85), 3: (0.55, 0.15, 0.65, 0.9)}
    cmap = ListedColormap([colors[1], colors[2], colors[3]])

    fig = plt.figure(figsize=(15, 7))
    ax_left = fig.add_subplot(1, 2, 1, projection='3d')
    ax_right = fig.add_subplot(1, 2, 2, projection='3d')

    nplot.plot_surf_stat_map(fsaverage.infl_left, plot_data[:n_left], hemi='left', view='lateral',
                              bg_map=fsaverage.sulc_left, cmap=cmap, threshold=0.5, vmin=1, vmax=3,
                              colorbar=False, bg_on_data=True, darkness=None, axes=ax_left, figure=fig)
    ax_left.set_title('Left Lateral', fontsize=12)

    nplot.plot_surf_stat_map(fsaverage.infl_right, plot_data[n_left:], hemi='right', view='lateral',
                              bg_map=fsaverage.sulc_right, cmap=cmap, threshold=0.5, vmin=1, vmax=3,
                              colorbar=False, bg_on_data=True, darkness=None, axes=ax_right, figure=fig)
    ax_right.set_title('Right Lateral', fontsize=12)

    fig.text(0.03, 0.95, title, ha='left', va='top', fontsize=15, fontweight='bold')
    legend_labels = [(f'{label_a} only', colors[1]), (f'{label_b} only', colors[2]), ('Overlap', colors[3])]
    for i, (label, color) in enumerate(legend_labels):
        fig.text(0.03, 0.90 - i * 0.04, label, ha='left', va='top', color=color, fontsize=13, fontweight='bold')

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_fsaverage_meshes():
    from nilearn.datasets import fetch_surf_fsaverage
    fsaverage = fetch_surf_fsaverage(mesh='fsaverage7')
    meshes = {
        'white_left': fsaverage.white_left,
        'pial_left': fsaverage.pial_left,
        'white_right': fsaverage.white_right,
        'pial_right': fsaverage.pial_right,
    }
    return fsaverage, meshes
