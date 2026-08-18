import argparse
import os
from pathlib import Path
from itertools import product, combinations

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from utils.overlap import (DEFAULT_PERCENT_THRESHOLDS, load_hemi_parcel_masks, top_percent_mask,
                            top_percent_mask_bilateral, dice, plot_bilateral_overlap_surface,
                            load_fsaverage_meshes)

DYAD_CONTRASTS = ['com_ind-ind', 'com_phy-phy']
FACE_CONTRASTS = ['face_first-face_noncom', 'face_third-face_noncom']
ALL_CONTRASTS = DYAD_CONTRASTS + FACE_CONTRASTS
# All pairwise combinations among the four contrasts (dyad-dyad, face-face, and dyad-face)
CONTRAST_PAIRS = list(combinations(ALL_CONTRASTS, 2))
PERCENT_THRESHOLDS = DEFAULT_PERCENT_THRESHOLDS
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

    The left and right STS parcels are kept separate throughout: voxels are
    ranked by t-value (positive values only, no significance threshold)
    independently within each hemisphere's parcel, and the top X% of that
    hemisphere's parcel is kept per contrast. Overlap between two such binary
    masks is summarized with the Dice coefficient, computed separately per
    hemisphere and swept across several percent levels, for every pairwise
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

    def compute_dice_table(self, hemi_masks):
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
            for hemi, parcel_mask in hemi_masks.items():
                for percent in PERCENT_THRESHOLDS:
                    mask_a = top_percent_mask(vals_a, parcel_mask, percent)
                    mask_b = top_percent_mask(vals_b, parcel_mask, percent)
                    rows.append({
                        'subject': self.subject,
                        'hemisphere': hemi,
                        'contrast_a': contrast_a,
                        'contrast_b': contrast_b,
                        'pair_type': pair_type(contrast_a, contrast_b),
                        'percent_threshold': percent,
                        'n_a_voxels': int(mask_a.sum()),
                        'n_b_voxels': int(mask_b.sum()),
                        'n_overlap_voxels': int(np.logical_and(mask_a, mask_b).sum()),
                        'dice': dice(mask_a, mask_b),
                    })
        df = pd.DataFrame(rows)
        outfile = os.path.join(self.subject_out_path, f'sub-{self.subject}_dice_coefficients.csv')
        df.to_csv(outfile, index=False)
        print(f'Saved {outfile}')
        return df

    def plot_all_surfaces(self, hemi_masks, affine, header, percent=SURFACE_PERCENT):
        fsaverage, fsaverage_meshes = load_fsaverage_meshes()
        outdir = os.path.join(self.out_path, 'IndividualSurfaces', f'sub-{self.subject}')

        for dyad_contrast, face_contrast in tqdm(list(product(DYAD_CONTRASTS, FACE_CONTRASTS)),
                                                   desc=f'sub-{self.subject} surfaces'):
            dyad_file = self.contrast_file(dyad_contrast)
            face_file = self.contrast_file(face_contrast)
            if not (os.path.exists(dyad_file) and os.path.exists(face_file)):
                continue
            dyad_mask = top_percent_mask_bilateral(nib.load(dyad_file).get_fdata(), hemi_masks, percent)
            face_mask = top_percent_mask_bilateral(nib.load(face_file).get_fdata(), hemi_masks, percent)
            out_file = os.path.join(outdir, f'sub-{self.subject}_{dyad_contrast}_vs_{face_contrast}_'
                                             f'top{int(percent * 100)}pct_surface.png')
            plot_bilateral_overlap_surface(
                dyad_mask, face_mask, affine, header, fsaverage, fsaverage_meshes,
                label_a=dyad_contrast, label_b=face_contrast,
                title=f'sub-{self.subject}: {dyad_contrast}  vs  {face_contrast}  '
                      f'(top {int(percent * 100)}% per hemisphere)',
                out_file=out_file)

    def run(self):
        hemi_masks, affine, header = load_hemi_parcel_masks(self.parcel_path)

        self.compute_dice_table(hemi_masks)
        self.plot_all_surfaces(hemi_masks, affine, header)


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
