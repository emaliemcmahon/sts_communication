import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from decoding_index import CONDITIONS, FOLD_RUNS, DecodingIndex
from utils.mvpa import ROI_DEFINING_CONTRAST, roi_masks_resolved

# Each pair is tested independently, without requiring generalization across
# agent count (contrast this with decoding_index.py's cross-condition index,
# which needs both pairs to generalize to one another). 'dyad' asks whether
# com_ind is distinguishable from ind on its own; 'face' asks the same for
# face_first vs. face_noncom.
PAIRS = [('dyad', 'com_ind', 'ind'), ('face', 'face_first', 'face_noncom')]


class WithinDecodingIndex:
    """
    Within-condition-pair decoding index, per participant, hemisphere, and
    ROI: for a pair of conditions (com_ind vs. ind, or face_first vs.
    face_noncom), tests whether each condition's pattern is more similar to
    itself (split-half reliability, across fold A/B) than to the other
    condition's pattern (cross-fold, cross-condition correlation) --
    without requiring the cross-condition-pair generalization that
    decoding_index.py's Haxby-style index tests.

    within decoding index = 0.5 * (r_w1 + r_w2) - r_b
        r_w1 = corr(cond1_A, cond1_B)   [split-half reliability of cond1]
        r_w2 = corr(cond2_A, cond2_B)   [split-half reliability of cond2]
        r_b  = cv_corr(cond1, cond2)    [between: cond1 vs. cond2, cross-fold]

    Uses the same fold A (runs 1-5) / fold B (runs 6-9) split and
    NilearnGLMRunwise per-run betas as decoding_index.py (see that module's
    docstring for the cross-validation rationale), and reuses its
    `DecodingIndex.cv_corr` static method directly -- note that
    `cv_corr(patterns_a, patterns_b, X, X)` collapses to the ordinary
    split-half reliability `corr(X_A, X_B)`, since correlation is symmetric
    and both of `cv_corr`'s cross-fold terms are then identical.

    Comparing this within-pair index (here) against the cross-pair index
    (decoding_index.py) shows whether communicative/independent
    distinguishability requires generalizing across agent count, or is
    already present within dyad-only or face-only stimuli alone.
    """

    def __init__(self, args):
        self.process = 'WithinDecodingIndex'
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

    def run(self):
        rows = []
        for hemi in tqdm(self.hemis, desc=f'sub-{self.subject} within decoding index (CV)'):
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

                for pair_name, cond1, cond2 in PAIRS:
                    r_w1 = DecodingIndex.cv_corr(patterns_a, patterns_b, cond1, cond1)
                    r_w2 = DecodingIndex.cv_corr(patterns_a, patterns_b, cond2, cond2)
                    r_b = DecodingIndex.cv_corr(patterns_a, patterns_b, cond1, cond2)
                    within_decoding_index = 0.5 * (r_w1 + r_w2) - r_b

                    rows.append({
                        'subject': self.subject, 'roi': roi, 'hemi': hemi, 'n_voxels': int(voxels.sum()),
                        'pair': pair_name, 'cond1': cond1, 'cond2': cond2,
                        'r_w1_reliability': r_w1, 'r_w2_reliability': r_w2,
                        'r_b_between': r_b,
                        'within_decoding_index': within_decoding_index,
                    })

        df = pd.DataFrame(rows)
        out_file = os.path.join(self.out_path, f'sub-{self.subject}_within_decoding_index.csv')
        df.to_csv(out_file, index=False)
        print(f'Saved {out_file}')
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Within-condition-pair (com_ind vs. ind; face_first vs. face_noncom) '
                    'cross-validated decoding index, per ROI, for one subject, using '
                    'NilearnGLMRunwise per-run betas (runs 1-5 vs. runs 6-9 folds).')
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
    WithinDecodingIndex(args).run()


if __name__ == '__main__':
    main()
