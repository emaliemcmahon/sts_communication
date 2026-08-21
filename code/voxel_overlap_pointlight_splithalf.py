import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from utils.overlap import (DEFAULT_PERCENT_THRESHOLDS, load_hemi_parcel_masks, top_percent_mask,
                            top_percent_mask_bilateral, dice, plot_bilateral_overlap_surface,
                            load_fsaverage_meshes)

POINTLIGHT_CONTRAST = 'interact-noninteract'
RUN_SPLITS = {'odd': [1, 3], 'even': [2, 4]}
PERCENT_THRESHOLDS = DEFAULT_PERCENT_THRESHOLDS
SURFACE_PERCENT = 0.5  # threshold used for the individual-subject surface overlap plot


class VoxelOverlapPointlightSplitHalf:
    """
    Split-half reliability of the pointlight interact-noninteract contrast within
    a single subject's left/right STS parcel: compares the top-X% voxels (by
    effect size, positive values only, no significance threshold) defined from
    odd runs (1, 3) against those defined from even runs (2, 4).

    Uses the per-run effect-size maps already produced by NilearnGLMRunwise
    (nilearn_glm_runwise.py) for the 'interact' and 'noninteract' conditions
    individually (contrast-interact_run-N.nii.gz / contrast-noninteract_run-N.nii.gz,
    each estimated from that single run alone). The odd/even contrast map is the
    across-run average of (interact - noninteract) for the runs in that half.
    This is distinct from NilearnGLMRunwise's own contrast-interact-noninteract_run-N.nii.gz
    files, which are leave-one-run-out z-score maps (fit on the 3 runs NOT equal
    to N) and do not correspond to an odd/even split.

    The left and right STS parcels are kept separate throughout, matching
    voxel_overlap.py: top-percent selection and the Dice coefficient are both
    computed independently per hemisphere.

    This script processes one subject at a time; run it once per subject
    (see batch_voxel_overlap_pointlight_splithalf.sh / `make voxel_overlap_pointlight_splithalf`).
    Group aggregation happens together with voxel_overlap.py's per-subject output
    in voxel_overlap_group.py, which uses this split-half Dice as a noise ceiling.
    """

    def __init__(self, args):
        self.process = 'VoxelOverlapPointlightSplitHalf'
        self.subject = args.subject
        self.dataset_path = args.dataset_path
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.runwise_path = os.path.join(self.derivatives_path, 'NilearnGLMRunwise', f'sub-{self.subject}')
        self.parcel_path = os.path.join(self.derivatives_path, f'parcels-{args.space_label}')
        self.out_path = os.path.join(self.derivatives_path, self.process)
        self.subject_out_path = os.path.join(self.out_path, f'sub-{self.subject}')
        Path(self.subject_out_path).mkdir(parents=True, exist_ok=True)

    def run_file(self, condition, run):
        return os.path.join(self.runwise_path,
                             f'sub-{self.subject}_task-pointlight_contrast-{condition}_run-{run}.nii.gz')

    def split_contrast(self, runs):
        """Average (interact - noninteract) effect size across the given runs."""
        diffs = []
        for run in runs:
            interact_file = self.run_file('interact', run)
            noninteract_file = self.run_file('noninteract', run)
            if not (os.path.exists(interact_file) and os.path.exists(noninteract_file)):
                print(f'WARNING: missing pointlight run {run} for sub-{self.subject}, skipping that run')
                continue
            interact = nib.load(interact_file).get_fdata()
            noninteract = nib.load(noninteract_file).get_fdata()
            diffs.append(interact - noninteract)
        if not diffs:
            return None
        return np.mean(diffs, axis=0)

    def compute_dice_table(self, hemi_masks):
        odd_vals = self.split_contrast(RUN_SPLITS['odd'])
        even_vals = self.split_contrast(RUN_SPLITS['even'])
        rows = []
        if odd_vals is None or even_vals is None:
            print(f'WARNING: missing odd or even pointlight runs for sub-{self.subject}, skipping subject')
        else:
            for hemi, parcel_mask in hemi_masks.items():
                for percent in PERCENT_THRESHOLDS:
                    mask_odd = top_percent_mask(odd_vals, parcel_mask, percent)
                    mask_even = top_percent_mask(even_vals, parcel_mask, percent)
                    rows.append({
                        'subject': self.subject,
                        'hemisphere': hemi,
                        'percent_threshold': percent,
                        'n_odd_voxels': int(mask_odd.sum()),
                        'n_even_voxels': int(mask_even.sum()),
                        'n_overlap_voxels': int(np.logical_and(mask_odd, mask_even).sum()),
                        'dice': dice(mask_odd, mask_even),
                    })
        df = pd.DataFrame(rows)
        outfile = os.path.join(self.subject_out_path, f'sub-{self.subject}_dice_coefficients.csv')
        df.to_csv(outfile, index=False)
        print(f'Saved {outfile}')
        return df, odd_vals, even_vals

    def plot_surface(self, odd_vals, even_vals, hemi_masks, affine, header, percent=SURFACE_PERCENT):
        if odd_vals is None or even_vals is None:
            return
        fsaverage, fsaverage_meshes = load_fsaverage_meshes()
        odd_mask = top_percent_mask_bilateral(odd_vals, hemi_masks, percent)
        even_mask = top_percent_mask_bilateral(even_vals, hemi_masks, percent)

        outdir = os.path.join(self.out_path, 'IndividualSurfaces', f'sub-{self.subject}')
        out_file = os.path.join(outdir, f'sub-{self.subject}_odd_vs_even_top{int(percent * 100)}pct_surface.png')
        plot_bilateral_overlap_surface(
            odd_mask, even_mask, affine, header, fsaverage, fsaverage_meshes,
            label_a='odd runs', label_b='even runs',
            title=f'sub-{self.subject}: pointlight {POINTLIGHT_CONTRAST}, odd vs even runs '
                  f'(top {int(percent * 100)}% per hemisphere)',
            out_file=out_file)

    def run(self):
        hemi_masks, affine, header = load_hemi_parcel_masks(self.parcel_path)
        df, odd_vals, even_vals = self.compute_dice_table(hemi_masks)
        self.plot_surface(odd_vals, even_vals, hemi_masks, affine, header)


def main():
    parser = argparse.ArgumentParser(
        description='Split-half (odd vs even runs) overlap of the pointlight interact-noninteract '
                     'contrast within the STS parcel, for one subject')
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym')
    args = parser.parse_args()
    VoxelOverlapPointlightSplitHalf(args).run()


if __name__ == '__main__':
    main()
