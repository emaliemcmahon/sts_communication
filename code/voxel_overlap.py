import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

from utils.overlap import (DEFAULT_PERCENT_THRESHOLDS, load_hemi_parcel_masks, top_percent_mask,
                            top_percent_mask_bilateral, dice, plot_bilateral_overlap_surface,
                            load_fsaverage_meshes)

COMMUNICATE_CONTRASTS = ['com_ind-ind', 'com_phy-phy', 'face_first-face_noncom', 'face_third-face_noncom']
POINTLIGHT_CONTRAST = 'interact-noninteract'
PERCENT_THRESHOLDS = DEFAULT_PERCENT_THRESHOLDS
SURFACE_PERCENT = 0.5  # threshold used for individual-subject surface overlap plots


class VoxelOverlap:
    """
    Voxel-level overlap between each of the four communicate contrasts
    (com_ind-ind, com_phy-phy, face_first-face_noncom, face_third-face_noncom)
    and the pointlight interact-noninteract contrast, within a single subject's
    left/right STS parcel. Both sides come from the whole-brain first-level
    NilearnGLM t-maps (nilearn_glm.py output; not the cross-validated
    NilearnGLMRunwise outputs used by voxel_overlap_pointlight_splithalf.py).

    Voxels are ranked by t-value (positive values only, no significance
    threshold) independently within each hemisphere's parcel, and the top X%
    of that hemisphere's parcel is kept per contrast. Overlap between two such
    binary masks is summarized with the Dice coefficient, computed separately
    per hemisphere and swept across several percent levels.

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
        self.out_path = os.path.join(self.derivatives_path, self.process)
        self.subject_out_path = os.path.join(self.out_path, f'sub-{self.subject}')
        Path(self.subject_out_path).mkdir(parents=True, exist_ok=True)

    def contrast_file(self, task_label, contrast):
        return os.path.join(self.glm_path, f'sub-{self.subject}', f'task-{task_label}',
                             f'contrast-{contrast}_stat-tmap.nii.gz')

    def compute_dice_table(self, hemi_masks):
        rows = []
        pointlight_file = self.contrast_file('pointlight', POINTLIGHT_CONTRAST)
        if not os.path.exists(pointlight_file):
            print(f'WARNING: missing pointlight contrast for sub-{self.subject}, skipping subject')
            df = pd.DataFrame(rows)
        else:
            pointlight_vals = nib.load(pointlight_file).get_fdata()
            for contrast in COMMUNICATE_CONTRASTS:
                communicate_file = self.contrast_file('communicate', contrast)
                if not os.path.exists(communicate_file):
                    print(f'WARNING: missing {contrast} for sub-{self.subject}, skipping contrast')
                    continue
                communicate_vals = nib.load(communicate_file).get_fdata()
                for hemi, parcel_mask in hemi_masks.items():
                    for percent in PERCENT_THRESHOLDS:
                        mask_c = top_percent_mask(communicate_vals, parcel_mask, percent)
                        mask_p = top_percent_mask(pointlight_vals, parcel_mask, percent)
                        rows.append({
                            'subject': self.subject,
                            'hemisphere': hemi,
                            'communicate_contrast': contrast,
                            'percent_threshold': percent,
                            'n_communicate_voxels': int(mask_c.sum()),
                            'n_pointlight_voxels': int(mask_p.sum()),
                            'n_overlap_voxels': int(np.logical_and(mask_c, mask_p).sum()),
                            'dice': dice(mask_c, mask_p),
                        })
            df = pd.DataFrame(rows)
        outfile = os.path.join(self.subject_out_path, f'sub-{self.subject}_dice_coefficients.csv')
        df.to_csv(outfile, index=False)
        print(f'Saved {outfile}')
        return df

    def plot_all_surfaces(self, hemi_masks, affine, header, percent=SURFACE_PERCENT):
        pointlight_file = self.contrast_file('pointlight', POINTLIGHT_CONTRAST)
        if not os.path.exists(pointlight_file):
            return
        pointlight_mask = top_percent_mask_bilateral(nib.load(pointlight_file).get_fdata(), hemi_masks, percent)

        fsaverage, fsaverage_meshes = load_fsaverage_meshes()
        outdir = os.path.join(self.out_path, 'IndividualSurfaces', f'sub-{self.subject}')

        for contrast in tqdm(COMMUNICATE_CONTRASTS, desc=f'sub-{self.subject} surfaces'):
            communicate_file = self.contrast_file('communicate', contrast)
            if not os.path.exists(communicate_file):
                continue
            communicate_mask = top_percent_mask_bilateral(nib.load(communicate_file).get_fdata(),
                                                            hemi_masks, percent)
            out_file = os.path.join(outdir, f'sub-{self.subject}_{contrast}_vs_{POINTLIGHT_CONTRAST}_'
                                             f'top{int(percent * 100)}pct_surface.png')
            plot_bilateral_overlap_surface(
                communicate_mask, pointlight_mask, affine, header, fsaverage, fsaverage_meshes,
                label_a=contrast, label_b=POINTLIGHT_CONTRAST,
                title=f'sub-{self.subject}: {contrast}  vs  {POINTLIGHT_CONTRAST}  '
                      f'(top {int(percent * 100)}% per hemisphere)',
                out_file=out_file)

    def run(self):
        hemi_masks, affine, header = load_hemi_parcel_masks(self.parcel_path)
        self.compute_dice_table(hemi_masks)
        self.plot_all_surfaces(hemi_masks, affine, header)


def main():
    parser = argparse.ArgumentParser(
        description='Voxel-level overlap between communicate contrasts and pointlight '
                     'interact-noninteract within the STS parcel, for one subject')
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym')
    args = parser.parse_args()
    VoxelOverlap(args).run()


if __name__ == '__main__':
    main()
