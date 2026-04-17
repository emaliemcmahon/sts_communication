import argparse
import os
import warnings
from tqdm import tqdm
from glob import glob
from pathlib import Path
import numpy as np
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.glm import threshold_stats_img
import nibabel as nib
from utils.mri import parse_contrast, info2vars
from nilearn.masking import intersect_masks


contrasts = {'communicate': [
                ('face_third', 'object'),
                ('face_third+face_first+com_phy+com_ind', 'face_noncom+phy+ind'),
                ('face_third+face_first', 'face_noncom'),
                ('com_phy+com_ind', 'phy+ind'),
                ('com_phy', 'phy'),
                ('com_ind', 'ind'),
                ('face_third', 'face_noncom'),
                ('face_first', 'face_noncom'),
                ('face_third+face_noncom', 'object'),
                ('face_first', 'face_third'),
                ('body', 'object')
            ],
             'tom': [('belief', 'photo')],
             'pointlight': [('interact', 'noninteract')]
            }

class NilearnGLM:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/NilearnGLM'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.sub_num = args.sub
        self.subj = str(self.sub_num).zfill(2)
        self.frame_threshold = 12  # Threshold for number of removed frames per run
        self.contrasts = contrasts.get(self.task_label)
        assert self.task_label in contrasts.keys(), f"Unknown task {self.task_label}"
        print(vars(self))
        Path(f'{self.out_path}/sub-{self.subj}/task-{self.task_label}').mkdir(parents=True, exist_ok=True)

    def save_map_for_parcel(self, zmap, contrast_name):
        # Create mask: positive contrast with p < 0.001 uncorrected
        thresholded, _ = threshold_stats_img(zmap, alpha=0.001, 
                                                height_control=None, two_sided=False)
        mask_data = np.where(thresholded.get_fdata() > 0, 1, 0).astype(np.int32)
        mask_img = nib.Nifti1Image(mask_data, zmap.affine)
        nib.save(mask_img, f'{self.out_path}/sub-{self.subj}/task-{self.task_label}/contrast-{contrast_name}_mask.nii.gz')

    def load_mask(self):
        mask_files = sorted(glob(f'{self.fmriprep_path}/sub-{self.subj}/ses-01/func/*task-{self.task_label}*{self.space_label}*brain_mask.nii.gz'))
        print(mask_files)
        masks = [nib.load(mask_file) for mask_file in mask_files]
        return intersect_masks(masks)

    def run_glm(self):
        mask = self.load_mask()

        # Load model info from BIDS
        model_info = flfb(self.dataset_path,
                          self.task_label,
                          self.space_label,
                          sub_labels=[self.subj],
                          mask_img=mask,
                          slice_time_ref=None,  # Load from the BIDS data
                          smoothing_fwhm=5.0,
                          img_filters=[("desc", "preproc")],
                          confounds_strategy=("motion",),
                          confounds_motion="basic",
                          derivatives_folder=self.fmriprep_path,
                          minimize_memory=True, 
                          hrf_model='spm',
                          n_jobs=-1)
        model, imgs, events, confounds = info2vars(model_info)

        # Shift the time series because fMRIPrep slice time corrects to the middle volume 
        # TR divided by half is 2s / 2 = 1s, so shift events by 1s to align with the corrected time series
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
        events_shifted = []
        for event in events: 
            event['onset'] = event['onset'] + 1
            events_shifted.append(event)

        # Fit the GLM with shifted events and filtered confounds
        model.fit(imgs, events_shifted, confounds)

        # Compute and save all contrasts
        for c1, c2 in tqdm(self.contrasts, total=len(self.contrasts), desc='Computing contrasts'):
            contrast_name = f'{c1}-{c2}'
            contrast = parse_contrast(model, c1, c2)
            tmap = model.compute_contrast(contrast, stat_type='t', output_type='stat')
            nib.save(tmap, f'{self.out_path}/sub-{self.subj}/task-{self.task_label}/contrast-{contrast_name}_stat-tmap.nii.gz')
            self.save_map_for_parcel(tmap, contrast_name)


def main():
    parser = argparse.ArgumentParser(description='Run first-level GLM for a subject and task')
    parser.add_argument('--sub', '-s', type=int, help='Subject number', default=2)
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--task_label', '-t', type=str, default='pointlight',
                         help='Task to run the GLM on (communicate, tom, pointlight)')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = NilearnGLM(args)
    processor.run_glm()


if __name__ == '__main__':
    main()