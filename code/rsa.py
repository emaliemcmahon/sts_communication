import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm

from utils.mvpa import ROI_DEFINING_CONTRAST, roi_masks_resolved, condition_pattern


# Condition order shared by the neural RDM and both model RDMs below. Restricted
# to the communicate task (no pointlight interact/noninteract).
CONDITIONS = ['com_ind', 'ind', 'face_first', 'face_noncom']

# Model RDMs (0 = predicted similar, 1 = predicted dissimilar), condition order as above.
# 3p_interaction is the submatrix of the original hand-drawn 6-condition RDM
# restricted to these four conditions. communication is a clean two-category
# split (communicative: com_ind, face_first; independent: ind, face_noncom) -
# the submatrix of the hand-drawn RDM had one cell (ind-face_first) drawn as
# similar despite crossing that category boundary; it's set to dissimilar here.
MODEL_RDMS = {
    '3p_interaction': np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]),
    'communication': np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]),
}

# Condensed (pdist-ordered) form of each model RDM, for comparison against
# condensed neural RDMs (both individual-level here and group-level in
# group_rsa.py, which loads the per-subject .npy files this script saves).
MODEL_RDMS_CONDENSED = {name: squareform(rdm, checks=False) for name, rdm in MODEL_RDMS.items()}


class RSA:
    """
    Representational similarity analysis comparing two model RDMs (3P
    interaction vs. communication) against the neural RDM in each functional
    ROI, for four communicate-task conditions: com_ind, ind, face_first,
    face_noncom.

    The neural RDM is a condensed correlation-distance vector (1 - Pearson
    correlation) between condition beta patterns (each vs. fixation),
    compared against each model RDM with a Spearman correlation. Each
    subject's condensed neural RDM is also saved to disk (.npy) for the
    group-level Mantel-permutation/noise-ceiling analysis in group_rsa.py.
    """

    def __init__(self, args):
        self.process = 'RSA'
        self.subject = args.subject
        self.dataset_path = args.dataset_path
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.glm_path = os.path.join(self.derivatives_path, 'NilearnGLM')
        self.parcel_path = os.path.join(self.derivatives_path, f'parcels-{args.space_label}')
        self.out_path = os.path.join(self.derivatives_path, self.process, f'sub-{self.subject}')
        self.hemis = ['l', 'r']
        self.rois = list(ROI_DEFINING_CONTRAST.keys())
        self.overlap_method = args.overlap_method
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def neural_rdm(self, patterns):
        """Condensed correlation-distance (1 - Pearson r) RDM from a list of condition pattern vectors."""
        return pdist(np.vstack(patterns), metric='correlation')

    def compare_to_models(self, rdm_condensed):
        """Spearman correlation between the condensed neural RDM and each condensed model RDM."""
        results = {}
        for model_name, model_condensed in MODEL_RDMS_CONDENSED.items():
            rho, p = spearmanr(rdm_condensed, model_condensed)
            results[f'{model_name}_rho'] = rho
            results[f'{model_name}_p'] = p
        return results

    def run(self):
        rows = []
        for hemi in tqdm(self.hemis, desc=f'sub-{self.subject} RSA'):
            hemi_masks = roi_masks_resolved(self.glm_path, self.parcel_path, self.subject, hemi,
                                            rois=self.rois, method=self.overlap_method)
            for roi in self.rois:
                voxels = hemi_masks.get(roi)
                if voxels is None or voxels.sum() < 2:
                    continue

                patterns = [condition_pattern(self.glm_path, self.subject, condition, voxels)
                            for condition in CONDITIONS]
                if any(pattern is None for pattern in patterns):
                    continue

                rdm = self.neural_rdm(patterns)
                np.save(os.path.join(self.out_path,
                                      f'sub-{self.subject}_hemi-{hemi}_roi-{roi}_neural_rdm.npy'), rdm)

                row = {'subject': self.subject, 'roi': roi, 'hemi': hemi, 'n_voxels': int(voxels.sum())}
                row.update(self.compare_to_models(rdm))
                rows.append(row)

        df = pd.DataFrame(rows)
        out_file = os.path.join(self.out_path, f'sub-{self.subject}_rsa.csv')
        df.to_csv(out_file, index=False)
        print(f'Saved {out_file}')
        return df


def main():
    parser = argparse.ArgumentParser(
        description='RSA comparing 3P-interaction and communication model RDMs '
                     'against neural RDMs per ROI, for one subject')
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
    RSA(args).run()


if __name__ == '__main__':
    main()
