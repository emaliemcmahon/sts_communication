import argparse
import os
import warnings
from glob import glob
from pathlib import Path
import numpy as np
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm import threshold_stats_img
import nibabel as nib
from utils.mri import check_motion_filtering, parse_contrast

contrasts = {'communicate': [
                ('face_third+face_first+com_phy+com_ind', 'face_noncom+phy+ind'),
                ('face_third+face_first', 'face_noncom'),
                ('com_phy+com_ind', 'phy+ind'),
                ('com_phy', 'phy'),
                ('com_ind', 'ind'),
                ('face_third', 'face_noncom'),
                ('face_first', 'face_noncom'),
                ('face_third+face_noncom', 'object'),
                ('body', 'object'),
                ('face_first', 'face_third'),
            ],
             'tom': [('belief', 'photo')],
             'pointlight': [('interact', 'noninteract')]
            }

class NilearnGLM:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/NilearnGLMRunwise'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.sub_num = args.sub_num
        self.subj = str(self.sub_num).zfill(2)
        self.frame_threshold = 12  # Threshold for number of removed frames per run
        self.contrasts = contrasts.get(self.task_label)
        assert self.task_label in contrasts.keys(), f"Unknown task {self.task_label}"
        print(vars(self))
        Path(f'{self.out_path}/sub-{self.subj}/task-{self.task_label}').mkdir(parents=True, exist_ok=True)

    def run_glm(self):
        # Load data for this subject
        files = sorted(glob(f'{self.fmriprep_path}/sub-{self.subj}/ses-*/func/*{self.task_label}*{self.space_label}*bold.nii.gz'))

        # Load confounds with motion filtering strategy
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            confounds_filtered, sample_masks = load_confounds(files, strategy=('motion', 'scrub'), 
                                                        fd_threshold=1, 
                                                        std_dvars_threshold=3, 
                                                        scrub=0,
                                                        motion='basic')

        # Check which runs exceed motion threshold
        n_trs = nib.load(files[0]).shape[-1]
        excluded_runs, included_runs = check_motion_filtering(sample_masks, n_trs,
                                                                threshold=self.frame_threshold)
        print(f'{len(excluded_runs)=}')
        print(f'{len(included_runs)=}')
        for i, m in enumerate(sample_masks):
            if m is not None:
                print(f'Run {i}: {len(m)=}/{n_trs=}')

        # Load model info from BIDS
        model_info = flfb(self.dataset_path,
                          self.task_label,
                          self.space_label,
                          sub_labels=[self.subj],
                          slice_time_ref=None,  # Load from the BIDS data
                          smoothing_fwhm=5.0,
                          img_filters=[("desc", "preproc")],
                          derivatives_folder=self.fmriprep_path,
                          minimize_memory=True, 
                          hrf_model='spm',
                          n_jobs=int(os.cpu_count()/2))
        (models, models_run_imgs, models_events, _) = model_info

        # Should be one model for one subject
        model = models[0]
        imgs = models_run_imgs[0]
        events = models_events[0]
        confounds = confounds_filtered

        # Shift the time series because fMRIPrep slice time corrects to the middle volume
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
        events_shifted = []
        for event in events: 
            event['onset'] = event['onset'] + 1
            events_shifted.append(event)

        # Fit the GLM with shifted events and filtered confounds
        model.fit(imgs, events_shifted, confounds)

        # Compute and save all contrasts
        for c1, c2 in self.contrasts:
            contrast_name = f'{c1}-{c2}'
            contrast = parse_contrast(model, c1, c2)
            # Compute t-map and effect size map
            tmap = model.compute_contrast(contrast, stat_type='t', output_type='stat')
            effect_map = model.compute_contrast(contrast, stat_type='t', output_type='effect_size')

            # Create mask: positive contrast with p < 0.001 uncorrected
            thresholded, _ = threshold_stats_img(tmap, alpha=0.001, 
                                                 height_control=None, two_sided=False)
            mask_data = np.where(thresholded.get_fdata() > 0, 1, 0).astype(np.int32)
            mask_img = nib.Nifti1Image(mask_data, tmap.affine)

            # Save effect size and mask
            nib.save(effect_map, f'{self.out_path}/sub-{self.subj}/task-{self.task_label}/contrast-{contrast_name}_effect-size.nii.gz')
            nib.save(mask_img, f'{self.out_path}/sub-{self.subj}/task-{self.task_label}/contrast-{contrast_name}_mask.nii.gz')
            print(f'Saved contrast {contrast_name} for sub-{self.subj}, task-{self.task_label}')


def main():
    parser = argparse.ArgumentParser(description='Run first-level GLM for a subject and task')
    parser.add_argument('sub_num', type=int, help='Subject number')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--task_label', '-t', type=str, required=True,
                         help='Task to run the GLM on (communicate, tom, pointlight)')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = NilearnGLM(args)
    processor.run_glm()


if __name__ == '__main__':
    main()