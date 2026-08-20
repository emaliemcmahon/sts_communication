import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mvpa import ROI_DEFINING_CONTRAST, roi_masks_resolved

# Runs 1-5 and 6-9 use disjoint sets of video exemplars per condition
# (generate_block.py's assign_videos_to_halves/GenerateBlock.run: each
# condition's 10 exemplars are split into two halves of 5, and
# `half = 1 if irun <= n_runs/2 else 2` for irun 0-indexed over the 9 runs
# means runs 1-5 always draw half 1's exemplars and runs 6-9 always draw
# half 2's -- confirmed empirically against the source run files). Splitting
# the CV folds here means cross-fold pattern correlations are estimated from
# both independent GLM noise (fixing the estimation-noise-correlation bias
# described in methods.md) *and* independent video exemplars.
FOLD_RUNS = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9]}
CONDITIONS = ('com_ind', 'ind', 'face_first', 'face_noncom')


class DecodingIndex:
    """
    Cross-validated Haxby-style within/between decoding index for com_ind,
    ind, face_first, face_noncom, testing whether "communicative" patterns
    (com_ind, face_first) are more similar to each other than to
    "independent" patterns (ind, face_noncom), generalizing across whether
    the stimulus depicts a dyad or a single face.

    Each pairwise pattern correlation is computed between condition betas
    estimated from *disjoint* sets of runs (fold A = runs 1-5, fold B =
    runs 6-9; NilearnGLMRunwise per-run betas) instead of from a single
    all-runs-pooled GLM.

    Rationale: cov(beta_A_from_fold_1, beta_B_from_fold_2) = 0 by
    construction whenever the two folds share no runs, regardless of how
    collinear the GLM design is. An earlier all-runs-pooled version of this
    analysis found the opposite: imperfect regressor orthogonality
    (drift/motion interacting with block timing) correlates the
    *estimation noise* of different conditions' betas and inflates
    same-fold pattern correlations even absent any true signal (see the
    whole-brain background-bias investigation in methods.md). Any true
    pattern similarity, in contrast, is stable across independent runs and
    survives the split.

    Each of the four terms is the average of both cross-fold directions
    (e.g. corr(com_ind_A, face_first_B) and corr(com_ind_B, face_first_A))
    to use all the data rather than picking one arbitrary direction.
    """

    def __init__(self, args):
        self.process = 'DecodingIndex'
        self.subject = args.subject
        self.dataset_path = args.dataset_path
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.glm_path = os.path.join(self.derivatives_path, 'NilearnGLM')
        self.runwise_glm_path = os.path.join(self.derivatives_path, 'NilearnGLMRunwise', f'sub-{self.subject}')
        self.parcel_path = os.path.join(self.derivatives_path, f'parcels-{args.space_label}')
        self.out_path = os.path.join(self.derivatives_path, self.process, f'sub-{self.subject}')
        self.hemis = ['l', 'r']
        self.rois = list(ROI_DEFINING_CONTRAST.keys())
        self.overlap_method = args.overlap_method
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def fold_pattern(self, condition, fold, voxels):
        """Average per-run beta pattern across one fold's runs, restricted to `voxels`."""
        run_patterns = []
        for run in FOLD_RUNS[fold]:
            beta_file = os.path.join(self.runwise_glm_path,
                                      f'sub-{self.subject}_task-communicate_contrast-{condition}_run-{run}.nii.gz')
            if not os.path.exists(beta_file):
                return None
            run_patterns.append(nib.load(beta_file).get_fdata().flatten()[voxels])
        return np.mean(run_patterns, axis=0)

    @staticmethod
    def cv_corr(patterns_a, patterns_b, cond1, cond2):
        """Cross-fold correlation between two conditions, averaged over both fold directions."""
        r1 = np.corrcoef(patterns_a[cond1], patterns_b[cond2])[0, 1]
        r2 = np.corrcoef(patterns_b[cond1], patterns_a[cond2])[0, 1]
        return 0.5 * (r1 + r2)

    def run(self):
        rows = []
        for hemi in tqdm(self.hemis, desc=f'sub-{self.subject} decoding index (CV)'):
            hemi_masks = roi_masks_resolved(self.glm_path, self.parcel_path, self.subject, hemi,
                                            rois=self.rois, method=self.overlap_method)
            for roi in self.rois:
                voxels = hemi_masks.get(roi)
                if voxels is None or voxels.sum() < 2:
                    continue

                patterns_a = {c: self.fold_pattern(c, 'A', voxels) for c in CONDITIONS}
                patterns_b = {c: self.fold_pattern(c, 'B', voxels) for c in CONDITIONS}
                if any(p is None for p in list(patterns_a.values()) + list(patterns_b.values())):
                    continue

                r_w1 = self.cv_corr(patterns_a, patterns_b, 'com_ind', 'face_first')
                r_w2 = self.cv_corr(patterns_a, patterns_b, 'ind', 'face_noncom')
                r_b1 = self.cv_corr(patterns_a, patterns_b, 'com_ind', 'face_noncom')
                r_b2 = self.cv_corr(patterns_a, patterns_b, 'face_first', 'ind')
                decoding_index = 0.5 * (r_w1 + r_w2 - r_b1 - r_b2)

                rows.append({
                    'subject': self.subject, 'roi': roi, 'hemi': hemi, 'n_voxels': int(voxels.sum()),
                    'r_w1_com_ind_face_first': r_w1,
                    'r_w2_ind_face_noncom': r_w2,
                    'r_b1_com_ind_face_noncom': r_b1,
                    'r_b2_face_first_ind': r_b2,
                    'decoding_index': decoding_index,
                })

        df = pd.DataFrame(rows)
        out_file = os.path.join(self.out_path, f'sub-{self.subject}_decoding_index.csv')
        df.to_csv(out_file, index=False)
        print(f'Saved {out_file}')
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Cross-validated Haxby-style within/between decoding index for com_ind, ind, '
                     'face_first, face_noncom, per ROI, for one subject, using NilearnGLMRunwise '
                     'per-run betas (runs 1-5 vs. runs 6-9 folds).')
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym')
    parser.add_argument('--overlap_method', type=str, choices=['winner_take_all', 'drop', 'none'],
                         default='winner_take_all',
                         help='How to resolve voxels independently selected by more than one '
                              "ROI's mask (see utils.mvpa.roi_masks_resolved). 'none' disables "
                              'resolution.')
    args = parser.parse_args()
    args.overlap_method = None if args.overlap_method == 'none' else args.overlap_method
    DecodingIndex(args).run()


if __name__ == '__main__':
    main()
