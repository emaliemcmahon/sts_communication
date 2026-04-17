import os
import argparse
from glob import glob
from pathlib import Path
from nilearn.glm.first_level import first_level_from_bids as flfb
import nibabel as nib
from tqdm import tqdm
from nilearn.masking import intersect_masks
from itertools import product
from utils.mri import info2vars, selective_mask_img


n_groups = {'pointlight': 4, 'tom': 2, 'communicate': 9}

response_contrasts = {'pointlight': ['interact', 'noninteract'],
                      'tom': ['belief', 'photo'],
                      'communicate': ['com_phy', 'com_ind', 'phy', 'ind',
                    'face_third', 'face_first', 'face_noncom',
                    'body', 'object']}

froi_contrasts = {'pointlight': {'interact-noninteract': ['SI-STS']}, 
                'tom': {'belief-photo': ['TPJ']},
                  'communicate': {'body-object': ['EBA'],
                                  '0.5*face_third+0.5*face_noncom-object': ['fSTS', 'FFA'],
                                  'phy-ind': ['phy-STS'],
                                  'com_phy-phy': ['comphy-STS'],
                                  'com_ind-ind': ['comind-STS'],
                                  '0.5*face_third+0.5*face_first-face_noncom': ['facecom-STS'],
                                  '0.5*com_phy+0.5*com_ind-0.5*phy-0.5*ind': ['dyadcom-STS'],
                                  '0.5*object+0.5*body': ['EVC', 'MT']}}

roi_size = {'comphy-STS': .05, 'comind-STS': .05, 'TPJ': .1,
            'EBA': .1, 'fSTS': .1, 'FFA': .1, 'SI-STS': .05, ''
            'EVC': 0.05, 'MT': 0.1, 
            'dyadcom-STS': .05, 'facecom-STS': .05,
            'phy-STS': 0.05}

roi_parc = {'comphy-STS': 'anatSTS',
            'comind-STS': 'anatSTS',
            'SI-STS': 'anatSTS', 
            'dyadcom-STS': 'anatSTS',
            'facecom-STS': 'anatSTS',
            'phy-STS': 'anatSTS'}


def roi_switcher(roi):
    if roi in list(roi_parc.keys()):
        return roi_parc[roi]
    else:
        return roi


def split_into_groups(items, n_groups=3):
    return [items[i::n_groups] for i in range(n_groups)]


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


class NilearnGLMRunwise:
    def __init__(self, args):
        self.process = 'NilearnGLMRunwise'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.motion_mode = args.motion_mode
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.parcel_path = f'{self.derivatives_path}/parcels-{self.space_label}'
        if self.motion_mode == 'lenient':
            self.out_path = f'{self.derivatives_path}/{self.process}'
        else:
            self.out_path = f'{self.derivatives_path}/{self.process}_{self.motion_mode}'
        self.TR = 2
        self.frame_threshold = 12
        print(vars(self))

    def load_mask(self):
        mask_files = sorted(glob(f'{self.fmriprep_path}/sub-{self.subject_label}/ses-01/func/*task-{self.task_label}*{self.space_label}*brain_mask.nii.gz'))
        print(mask_files)
        masks = [nib.load(mask_file) for mask_file in mask_files]
        return intersect_masks(masks)

    def glm(self):
        mask = self.load_mask()
        
        # Load model with first_level_from_bids but use our filtered confounds
        model_info = flfb(self.dataset_path,
                            self.task_label,
                            self.space_label,
                            mask_img=mask,
                            sub_labels=[self.subject_label],
                            slice_time_ref=None, # Load from the BIDS data
                            smoothing_fwhm=5.0,
                            img_filters=[("desc", "preproc")],
                            derivatives_folder=self.fmriprep_path,
                            minimize_memory=False, 
                            hrf_model='spm',
                            confounds_strategy=("motion",),
                            confounds_motion="basic",
                            n_jobs=-1)        
        model, imgs, events, confounds = info2vars(model_info)

        # Shift the time series because fMRIPrep slice time corrects to the middle volume
        # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
        events_shifted = []
        for event in events: 
            event['onset'] = event['onset'] + 1
            events_shifted.append(event)

        runs = [i for i in range(len(imgs))]
        n_groups_eff = min(n_groups[self.task_label], len(imgs))
        run_groups = split_into_groups(runs, n_groups=n_groups_eff)
        for igroup, runs in tqdm(enumerate(run_groups),
                                 total=n_groups_eff, desc='fitting run groups'):
            # Compute the model and contrasts to define the fROIs
            froi_imgs = [imgs[r] for r in runs if r not in runs]
            froi_events = [events_shifted[r] for r in runs if r not in runs]
            froi_confounds = [confounds[r] for r in runs if r not in runs]
            if froi_imgs:
                model.fit(froi_imgs, froi_events, froi_confounds)
                for contrast in froi_contrasts[self.task_label].keys():
                    contrast_name = hyphen_to_camel_case(contrast)
                    title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast_name}_run-{igroup+1}'
                    contrast_file = f'{self.out_path}/sub-{self.subject_label}/{title}.nii.gz'            
                    stat_map = model.compute_contrast(contrast, output_type='z_score')
                    nib.save(stat_map, contrast_file)

                    for hemi, roi in product(['l', 'r'], froi_contrasts[self.task_label][contrast]):
                        output_file = f'{self.out_path}/sub-{self.subject_label}/sub-{self.subject_label}_run-{igroup+1}_{hemi}{roi}'      
                        mask_file = f'{self.parcel_path}/{hemi}{roi_switcher(roi)}.nii.gz'
                        new_mask = selective_mask_img(mask_file, contrast_file, 
                                                      keep_prop=roi_size[roi],
                                                      debug_output=f'{output_file}.pdf')
                        nib.save(new_mask, f'{output_file}.nii.gz')

            # Compute the model and contrasts to estimate the responses
            resp_imgs = [imgs[r] for r in runs if r in runs]
            resp_events = [events_shifted[r] for r in runs if r in runs]
            resp_confounds = [confounds[r] for r in runs if r in runs]
            if resp_imgs:
                model.fit(resp_imgs, resp_events, resp_confounds)
                for contrast in response_contrasts[self.task_label]:
                    title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast}_run-{igroup+1}'
                    contrast_file = f'{self.out_path}/sub-{self.subject_label}/{title}.nii.gz' 
                    stat_map = model.compute_contrast(contrast, output_type='effect_size')
                    nib.save(stat_map, contrast_file)
    
    def run(self):
        if not os.path.exists(f'{self.out_path}/sub-{self.subject_label}'):
            Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)
            self.glm()
        else:
            if self.overwrite:
                Path(f'{self.out_path}/sub-{self.subject_label}').mkdir(parents=True, exist_ok=True)
                self.glm()
            else:
                print('Output already exists. To re-run pass --overwrite')


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=str, default='02',
                         help='Subject for the GLM')
    parser.add_argument('--task_label', '-t', type=str, default='pointlight',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    parser.add_argument('--motion_mode', choices=['lenient', 'strict'], default='lenient',
                         help='Motion filtering mode')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    processor = NilearnGLMRunwise(args)
    processor.run()

if __name__ == '__main__':
    main()