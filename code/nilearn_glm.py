import argparse
import time
from pathlib import Path
from copy import copy
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.plotting import plot_glass_brain, view_img_on_surf
from nilearn.plotting import plot_design_matrix
from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.glm import threshold_stats_img


contrast_names = {
                    'communicate': ['com_phy-phy', 'com_ind-ind',
                                    'com_phy', 'phy', 'com_ind', 'ind',
                                    'face_first-face_third',
                                    'face_first', 'face_third',
                                    '0.5*face_third+0.5*face_first-face_noncom',
                                    '0.5*face_third+0.5*face_first', 'face_noncom'],
                    'pointlight': ['interact-noninteract'],
                    'eploc': ['emotional-physical'],
                    'tom': ['belief-photo']
                 }


def hyphen_to_camel_case(contrast_name):
    """
    Converts a hyphen-separated contrast name into a camelCase-style contrast name.

    Parameters:
    contrast_name (str): The contrast name with hyphens (e.g., 'emotional-physical').

    Returns:
    str: The contrast name in camelCase (e.g., 'emotionalMinusPhysical').
    """
    def replace_symbol(out, symbol, name=None):
        parts = out.split(symbol)   
        part1 =  parts[0]
        part2 = f'{symbol}'.join(parts[1:])
        if name is not None:
            out = part1 + name + part2[0].upper() + part2[1:]
        else:
            out = part1 + part2[0].upper() + part2[1:]
        return out
    out = contrast_name.replace('.', '').replace('*', '')
    for symbol, name in zip(['+', '-', '_'], ['Plus', 'Minus', None]):
        while symbol in out: 
            out = replace_symbol(out, symbol, name)
    return out


def info2vars(model_info):
    (models, imgs, events, confounds) = model_info
    return models[0], imgs[0], events[0], confounds[0]


def check_motion_filtering(confounds, frame_threshold=5):
    bad_runs = 0
    for confound in confounds: 
        if confound['rot_x'].isna().sum() > frame_threshold: 
            bad_runs += 1
    return bad_runs


class NilearnGLM:
    def __init__(self, args):
        self.process = 'NilearnGLM'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.threshold_p = 0.01
        self.TR = 2
        self.frame_threshold = 12
        print(vars(self))
        Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)

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
                          standardize=True,
                          n_jobs=-1)
        
        # Print info to make ensure correct loading
        model, imgs, events, confounds = info2vars(model_info)

        bad_runs = check_motion_filtering(confounds, frame_threshold=self.frame_threshold)
        print(f'{bad_runs=}')

        # Shift the time series because fMRIPrep slice time corrects to the middle volume
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
        events_shifted = []
        for event in events: 
            event['onset'] = event['onset'] + 1
            events_shifted.append(event)

        # Fit the model 
        print('Starting GLM fitting...')
        start_time = time.time()
        model.fit(imgs, events_shifted, confounds)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Fitting GLM took {elapsed_time/60:.2f} minutes')  

        # Plot design as png for easy viewing
        for run, mat in enumerate(model.design_matrices_):
            output_file = f'{self.out_path}/sub-{self.subject_label}/sub-{self.subject_label}_task-{self.task_label}_run-{run}_design.png'
            plot_design_matrix(mat, output_file=output_file)

        # Compute the contrasts
        for contrast in contrast_names[self.task_label]:
            title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{hyphen_to_camel_case(contrast)}'
            output_file = f'{self.out_path}/sub-{self.subject_label}/{title}'            
            stat_map = model.compute_contrast(contrast,
                                              stat_type='t',
                                              output_type='z_score')
            threshold_map, threshold = threshold_stats_img(stat_map,
                                                           alpha=self.threshold_p)
            plot_glass_brain(threshold_map,
                             colorbar=True,
                             threshold=threshold,
                             title=title,
                             plot_abs=False,
                             display_mode="x",
                             output_file=f'{output_file}.pdf')
            view = view_img_on_surf(threshold_map,
                                    threshold=threshold)
            view.save_as_html(f'{output_file}.html')  

        save_glm_to_bids(model, 
                         contrasts=contrast_names[self.task_label],
                         contrast_types={c: 't' for c in contrast_names[self.task_label]},
                         out_dir=f'{self.out_path}',
                         prefix=f'sub-{self.subject_label}_task-{self.task_label}')


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
    args = parser.parse_args()

    processor = NilearnGLM(args)
    processor.glm()

if __name__ == '__main__':
    main()