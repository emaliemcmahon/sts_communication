import os
import shutil
import argparse
from pathlib import Path
from nilearn.glm.first_level import first_level_from_bids as flfb
import nibabel as nib
from tqdm import tqdm


contrast_names = {
    'communicate': ['com_phy', 'com_ind', 'phy', 'ind',
                    'face_third', 'face_first', 'face_noncom',
                    'body', 'object'],
    'pointlight': ['interact', 'noninteract'],
    'eploc': ['emotional', 'physical'],
    'tom': ['belief', 'photo']
}


def info2vars(model_info):
    (models, imgs, events, confounds) = model_info
    return models[0], imgs[0], events[0], confounds[0]


def check_motion_filtering(confounds, frame_threshold=5):
    n_excluded_runs = 0
    included_runs = []
    for irun, confound in enumerate(confounds):
        if confound['rot_x'].isna().sum() > frame_threshold: 
            n_excluded_runs += 1
        else:
            included_runs.append(irun)
    return n_excluded_runs, included_runs


def split_into_groups(items, n_groups=3):
    return [items[i::n_groups] for i in range(n_groups)]


class NilearnGLMRunwise:
    def __init__(self, args):
        self.process = 'NilearnGLMRunwise'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.overwrite = args.overwrite
        self.n_groups = args.n_groups
        self.TR = 2
        self.frame_threshold = 12
        print(vars(self))

    def glm(self):
        model_info = flfb(self.dataset_path,
                          self.task_label,
                          self.space_label,
                          sub_labels=[self.subject_label],
                          slice_time_ref=None, # Load from the BIDS data
                          smoothing_fwhm=5.0,
                          img_filters=[("desc", "preproc")],
                          confounds_strategy=('motion', 'scrub'),
                          confounds_motion='basic',
                          derivatives_folder=self.fmriprep_path,
                          minimize_memory=False, 
                          hrf_model='spm',
                          confounds_fd_threshold=0.5, #FD in mm
                          confounds_scrub=5, #remove segments shorter than the given number after scrubbing
                          n_jobs=-1)
        
        # Print info to ensure correct loading
        model, imgs, events, confounds = info2vars(model_info)

        n_excluded_runs, included_runs = check_motion_filtering(confounds,
                                                                frame_threshold=self.frame_threshold)
        print(f'{n_excluded_runs=}')

        # Shift the time series because fMRIPrep slice time corrects to the middle volume
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
        events_shifted = []
        for event in events: 
            event['onset'] = event['onset'] + 1
            events_shifted.append(event)

        run_groups = split_into_groups(included_runs, n_groups=self.n_groups)
        for igroup, runs in tqdm(enumerate(run_groups),
                                 total=self.n_groups, desc='fitting run groups'):
            model.fit([imgs[run] for run in runs], 
                      [events_shifted[run] for run in runs], 
                      [confounds[run] for run in runs])

            # Compute the contrasts
            for contrast in contrast_names[self.task_label]:
                title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast}_run-{igroup+1}'
                output_file = f'{self.out_path}/sub-{self.subject_label}/{title}.nii.gz'            
                stat_map = model.compute_contrast(contrast, output_type='effect_size')
                nib.save(stat_map, output_file)
    
    def run(self):
        if not os.path.exists(f'{self.out_path}/sub-{self.subject_label}'):
            Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)
            self.glm()
        else:
            if self.overwrite:
                shutil.rmtree(f'{self.out_path}/sub-{self.subject_label}')
                Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)
                self.glm()
            else:
                print('Output already exists. To re-run pass --overwrite')


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=str, default='01',
                         help='Subject for the GLM')
    parser.add_argument('--task_label', '-t', type=str, default='communicate',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    parser.add_argument('--n_groups', '-n', type=int, default=9,
                         help='Number of runs to load')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    processor = NilearnGLMRunwise(args)
    processor.run()

if __name__ == '__main__':
    main()