import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mvpa import ROI_DEFINING_CONTRAST, roi_masks_resolved, condition_pattern


class DecodingIndex:
    """
    Haxby-style within/between decoding index for com_ind, ind, face_first,
    face_noncom, testing whether "communicative" patterns (com_ind, face_first)
    are more similar to each other than to "independent" patterns (ind,
    face_noncom), generalizing across whether the stimulus depicts a dyad or
    a single face.

    decoding index = 0.5 * (r_w1 + r_w2 - r_b1 - r_b2)
        r_w1 = corr(com_ind, face_first)   [within communicate, dyad vs. face]
        r_w2 = corr(ind, face_noncom)      [within independent, dyad vs. face]
        r_b1 = corr(com_ind, face_noncom)  [between: communicate-dyad vs. independent-face]
        r_b2 = corr(face_first, ind)       [between: communicate-face vs. independent-dyad]
    """

    def __init__(self, args):
        self.process = 'DecodingIndex'
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

    @staticmethod
    def pattern_corr(a, b):
        return np.corrcoef(a, b)[0, 1]

    def run(self):
        rows = []
        for hemi in tqdm(self.hemis, desc=f'sub-{self.subject} decoding index'):
            hemi_masks = roi_masks_resolved(self.glm_path, self.parcel_path, self.subject, hemi,
                                            rois=self.rois, method=self.overlap_method)
            for roi in self.rois:
                voxels = hemi_masks.get(roi)
                if voxels is None or voxels.sum() < 2:
                    continue

                com_ind = condition_pattern(self.glm_path, self.subject, 'com_ind', voxels)
                ind = condition_pattern(self.glm_path, self.subject, 'ind', voxels)
                face_first = condition_pattern(self.glm_path, self.subject, 'face_first', voxels)
                face_noncom = condition_pattern(self.glm_path, self.subject, 'face_noncom', voxels)
                if any(pattern is None for pattern in (com_ind, ind, face_first, face_noncom)):
                    continue

                r_w1 = self.pattern_corr(com_ind, face_first)
                r_w2 = self.pattern_corr(ind, face_noncom)
                r_b1 = self.pattern_corr(com_ind, face_noncom)
                r_b2 = self.pattern_corr(face_first, ind)
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
        description='Haxby-style within/between decoding index for com_ind, ind, '
                     'face_first, face_noncom, per ROI, for one subject')
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
