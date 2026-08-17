import argparse
import os
from pathlib import Path
from itertools import product, combinations

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from nilearn.masking import intersect_masks
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import plotting as nplot

from utils.mri import vol2surf_int

DYAD_CONTRASTS = ['com_ind-ind', 'com_phy-phy']
FACE_CONTRASTS = ['face_first-face_noncom', 'face_third-face_noncom']
ALL_CONTRASTS = DYAD_CONTRASTS + FACE_CONTRASTS
# All pairwise combinations among the four contrasts (dyad-dyad, face-face, and dyad-face)
CONTRAST_PAIRS = list(combinations(ALL_CONTRASTS, 2))
PERCENT_THRESHOLDS = [0.05, 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
SURFACE_PERCENT = 0.5  # threshold used for individual-subject surface overlap plots


def pair_type(contrast_a, contrast_b):
    a_is_dyad, b_is_dyad = contrast_a in DYAD_CONTRASTS, contrast_b in DYAD_CONTRASTS
    a_is_face, b_is_face = contrast_a in FACE_CONTRASTS, contrast_b in FACE_CONTRASTS
    if a_is_dyad and b_is_dyad:
        return 'dyad-dyad'
    if a_is_face and b_is_face:
        return 'face-face'
    assert (a_is_dyad and b_is_face) or (a_is_face and b_is_dyad)
    return 'dyad-face'


class VoxelOverlap:
    """
    Voxel-level overlap between dyad contrasts (com_ind-ind, com_phy-phy) and
    face-perception contrasts (face_first-face_noncom, face_third-face_noncom)
    within the anatomical STS parcel, in a single subject.

    Voxels are ranked by t-value (positive values only, no significance
    threshold) and the top X% of STS-parcel voxels are kept per contrast.
    Overlap between two such binary masks is summarized with the Dice
    coefficient, swept across several percent levels, for every pairwise
    combination of the four contrasts above (including the two within-domain
    pairs, com_ind-ind vs com_phy-phy and face_first-face_noncom vs
    face_third-face_noncom, as a baseline for the cross-domain dyad-vs-face
    comparisons).

    This script processes one subject at a time; run it once per subject
    (see batch_voxel_overlap.sh / `make voxel_overlap`) and then run
    voxel_overlap_group.py to aggregate results across subjects.
    """

    def __init__(self, args):
        self.process = 'VoxelOverlap'
        self.subject = args.subject
        self.dataset_path = args.dataset_path
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.glm_path = os.path.join(self.derivatives_path, 'NilearnGLM')
        self.parcel_path = os.path.join(self.derivatives_path, f'parcels-{args.space_label}')
        self.task_label = args.task_label
        self.out_path = os.path.join(self.derivatives_path, self.process)
        self.subject_out_path = os.path.join(self.out_path, f'sub-{self.subject}')
        Path(self.subject_out_path).mkdir(parents=True, exist_ok=True)

    def contrast_file(self, contrast):
        return os.path.join(self.glm_path, f'sub-{self.subject}', f'task-{self.task_label}',
                             f'contrast-{contrast}_stat-tmap.nii.gz')

    def load_sts_parcel(self):
        """Union of left + right anatomical STS parcels."""
        parcel_files = [os.path.join(self.parcel_path, f'{h}anatSTS.nii.gz') for h in ('l', 'r')]
        return intersect_masks(parcel_files, threshold=0, connected=False)

    @staticmethod
    def top_percent_mask(values, parcel_mask, percent):
        """
        Boolean mask of the top `percent` of parcel voxels by t-value.
        Only positive t-values are eligible (no significance threshold applied).
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

    @staticmethod
    def dice(mask_a, mask_b):
        n_a, n_b = mask_a.sum(), mask_b.sum()
        if n_a + n_b == 0:
            return np.nan
        overlap = np.logical_and(mask_a, mask_b).sum()
        return 2 * overlap / (n_a + n_b)

    def compute_dice_table(self, sts_mask):
        rows = []
        for contrast_a, contrast_b in CONTRAST_PAIRS:
            file_a = self.contrast_file(contrast_a)
            file_b = self.contrast_file(contrast_b)
            if not (os.path.exists(file_a) and os.path.exists(file_b)):
                print(f'WARNING: missing contrast for sub-{self.subject}: '
                      f'{contrast_a} or {contrast_b}, skipping')
                continue
            vals_a = nib.load(file_a).get_fdata()
            vals_b = nib.load(file_b).get_fdata()
            for percent in PERCENT_THRESHOLDS:
                mask_a = self.top_percent_mask(vals_a, sts_mask, percent)
                mask_b = self.top_percent_mask(vals_b, sts_mask, percent)
                rows.append({
                    'subject': self.subject,
                    'contrast_a': contrast_a,
                    'contrast_b': contrast_b,
                    'pair_type': pair_type(contrast_a, contrast_b),
                    'percent_threshold': percent,
                    'n_a_voxels': int(mask_a.sum()),
                    'n_b_voxels': int(mask_b.sum()),
                    'n_overlap_voxels': int(np.logical_and(mask_a, mask_b).sum()),
                    'dice': self.dice(mask_a, mask_b),
                })
        df = pd.DataFrame(rows)
        outfile = os.path.join(self.subject_out_path, f'sub-{self.subject}_dice_coefficients.csv')
        df.to_csv(outfile, index=False)
        print(f'Saved {outfile}')
        return df

    # ---- individual subject surface visualization ----

    def load_fsaverage(self):
        fsaverage = fetch_surf_fsaverage(mesh='fsaverage7')
        meshes = {
            'white_left': fsaverage.white_left,
            'pial_left': fsaverage.pial_left,
            'white_right': fsaverage.white_right,
            'pial_right': fsaverage.pial_right,
        }
        return fsaverage, meshes

    def plot_subject_overlap(self, dyad_contrast, face_contrast, sts_mask, affine, header,
                              fsaverage, fsaverage_meshes, percent):
        dyad_file = self.contrast_file(dyad_contrast)
        face_file = self.contrast_file(face_contrast)
        if not (os.path.exists(dyad_file) and os.path.exists(face_file)):
            return

        dyad_vals = nib.load(dyad_file).get_fdata()
        face_vals = nib.load(face_file).get_fdata()
        dyad_mask = self.top_percent_mask(dyad_vals, sts_mask, percent)
        face_mask = self.top_percent_mask(face_vals, sts_mask, percent)

        dyad_img = nib.Nifti1Image(dyad_mask.astype(np.int8), affine, header)
        face_img = nib.Nifti1Image(face_mask.astype(np.int8), affine, header)
        dyad_surf = vol2surf_int(dyad_img, fsaverage_meshes=fsaverage_meshes)
        face_surf = vol2surf_int(face_img, fsaverage_meshes=fsaverage_meshes)

        # Categorical code: 1 = dyad only, 2 = face only, 3 = overlap
        def categorize(dyad_part, face_part):
            cat = np.zeros_like(dyad_part, dtype=float)
            cat[(dyad_part > 0) & (face_part == 0)] = 1
            cat[(dyad_part == 0) & (face_part > 0)] = 2
            cat[(dyad_part > 0) & (face_part > 0)] = 3
            return cat

        left = categorize(dyad_surf.data.parts['left'], face_surf.data.parts['left'])
        right = categorize(dyad_surf.data.parts['right'], face_surf.data.parts['right'])
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

        fig.text(0.03, 0.95, f'sub-{self.subject}: {dyad_contrast}  vs  {face_contrast}  (top {int(percent * 100)}%)',
                  ha='left', va='top', fontsize=15, fontweight='bold')
        legend_labels = [(f'{dyad_contrast} only', colors[1]),
                          (f'{face_contrast} only', colors[2]),
                          ('Overlap', colors[3])]
        for i, (label, color) in enumerate(legend_labels):
            fig.text(0.03, 0.90 - i * 0.04, label, ha='left', va='top', color=color, fontsize=13, fontweight='bold')

        outdir = os.path.join(self.out_path, 'IndividualSurfaces', f'sub-{self.subject}')
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fname = f'sub-{self.subject}_{dyad_contrast}_vs_{face_contrast}_top{int(percent * 100)}pct_surface.png'
        fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def plot_all_surfaces(self, sts_mask, affine, header, percent=SURFACE_PERCENT):
        fsaverage, fsaverage_meshes = self.load_fsaverage()
        for dyad_contrast, face_contrast in tqdm(list(product(DYAD_CONTRASTS, FACE_CONTRASTS)),
                                                   desc=f'sub-{self.subject} surfaces'):
            self.plot_subject_overlap(dyad_contrast, face_contrast, sts_mask, affine, header,
                                       fsaverage, fsaverage_meshes, percent)

    def run(self):
        sts_img = self.load_sts_parcel()
        sts_mask = sts_img.get_fdata().astype(bool)

        self.compute_dice_table(sts_mask)
        self.plot_all_surfaces(sts_mask, sts_img.affine, sts_img.header)


def main():
    parser = argparse.ArgumentParser(
        description='Voxel-level overlap (Dice + individual surfaces) between dyad and '
                     'face-perception contrasts within the STS parcel, for one subject')
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--task_label', '-t', type=str, default='communicate')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym')
    args = parser.parse_args()
    VoxelOverlap(args).run()


if __name__ == '__main__':
    main()
