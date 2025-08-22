import argparse
import time
from glob import glob
from pathlib import Path
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.plotting import plot_glass_brain, view_img_on_surf
from nilearn.plotting import plot_design_matrix
from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.glm import threshold_stats_img
import nibabel as nib
from nilearn.datasets import load_mni152_brain_mask
from nilearn.masking import apply_mask, unmask, intersect_masks

contrast_names = {
                    'communicate': ['0.5*face_third+0.5*face_noncom-object', 
                                    'body-object'],
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


def check_motion_filtering(imgs, events, confounds, frame_threshold=5):
    imgs_out = []
    events_out = []
    confounds_out = []
    bad_runs = 0
    for img, event, confound in zip(imgs, events, confounds): 
        if confound['rot_x'].isna().sum() > frame_threshold: 
            bad_runs += 1
        else:
            imgs_out.append(img)
            events_out.append(event)
            confounds_out.append(confound)
    return imgs_out, events_out, confounds_out, bad_runs

def compute_snr(img_list, mask, output_file=None):
    mask = nib.load(mask) if type(mask) is str else mask

    affine = None
    counter = 0 
    for img_file in img_list:
        counter += 1

        img = nib.load(img_file)
        img_arr = unmask(apply_mask(img, mask), mask).get_fdata()

        if affine is None: 
            affine = img.affine
            avg = img_arr.mean(axis=-1)
            sd = img_arr.std(axis=-1)
        else:
            avg += img_arr.mean(axis=-1)
            sd += img_arr.std(axis=-1)
    # The the mean across runs
    avg /= counter
    sd /= counter

    # Calculate the SNR
    snr = avg/sd
    snr = nib.Nifti1Image(snr, affine=affine)

    if output_file is not None:
        plot_glass_brain(snr,
                         colorbar=True,
                         output_file=output_file)
    return snr


class NilearnGLM:
    def __init__(self, args):
        self.process = 'NilearnGLM'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = str(args.subject_label).zfill(2)
        if self.space_label == 'MNI152NLin2009cAsym':
            self.mask = load_mni152_brain_mask()
        else: 
            self.mask_file = None
        self.alpha = 0.001
        self.correction = None
        self.TR = 2
        self.frame_threshold = 12
        print(vars(self))
        Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)

    def load_mask(self):
        mask_files = sorted(glob(f'{self.fmriprep_path}/sub-{self.subject_label}/ses-01/func/*{self.task_label}*{self.space_label}*brain_mask.nii.gz'))
        masks = [nib.load(mask_file) for mask_file in mask_files]
        return intersect_masks(masks)

    def glm(self):
        mask = self.load_mask()
        model_info = flfb(self.dataset_path,
                          self.task_label,
                          self.space_label,
                          mask_img=mask,
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
                          confounds_std_dvars_threshold=1.5,
                          n_jobs=-1)
        
        # Print info to make ensure correct loading
        model, imgs, events, confounds = info2vars(model_info)

        # Check SNR
        output_file = f'{self.out_path}/sub-{self.subject_label}/sub-{self.subject_label}_task-{self.task_label}_stat-snr.pdf'
        snr = compute_snr(imgs, mask, output_file=output_file)

        filtered_motion = check_motion_filtering(imgs, events, confounds,
                                                 frame_threshold=self.frame_threshold)
        imgs, events, confounds, bad_runs = filtered_motion
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
            tmap = model.compute_contrast(contrast,
                                              stat_type='t',
                                              output_type='stat')
            tmap_thresholded, threshold = threshold_stats_img(tmap, 
                                                              alpha=self.alpha,
                                                              height_control=self.correction)
            plot_glass_brain(tmap_thresholded,
                             colorbar=True,
                             threshold=threshold,
                             title=title,
                             plot_abs=False,
                            #  display_mode="x",
                             output_file=f'{output_file}.pdf')
            view = view_img_on_surf(tmap_thresholded, 
                                    surf_mesh='fsaverage',
                                    threshold=threshold,
                                    bg_on_data=True,
                                    darkness=0.5,
                                    title=title, 
                                    colorbar_height=0.75)
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
    parser.add_argument('--subject_label', '-s', type=int, default=3,
                         help='Subject for the GLM')
    parser.add_argument('--task_label', '-t', type=str, default='pointlight',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = NilearnGLM(args)
    processor.glm()


if __name__ == '__main__':
    main()