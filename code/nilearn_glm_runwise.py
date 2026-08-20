import os
import argparse
from pathlib import Path
from nilearn.glm.first_level import first_level_from_bids as flfb
import nibabel as nib
import numpy as np
from tqdm import tqdm
from utils.mri import (info2vars, selective_mask_img, parse_contrast, roi_size, roi_switcher,
                        load_brain_mask, winner_take_all)
import warnings

# Suppress the specific warning about unexpected columns in events data
warnings.filterwarnings("ignore", message="The following unexpected columns in events data will be ignored: response", category=UserWarning)

# Suppress the warning about the same contrast being used for all runs
warnings.filterwarnings("ignore", message="The same contrast will be used for all * runs.*", category=RuntimeWarning)


n_groups = {'pointlight': 4, 'tom': 2, 'communicate': 9}

response_contrasts = {'pointlight': ['interact', 'noninteract'],
                      'tom': ['belief', 'photo'],
                      'communicate': ['com_phy', 'com_ind', 'phy', 'ind',
                    'face_third', 'face_first', 'face_noncom',
                    'body', 'object']}

froi_contrasts = {'pointlight': {'interact-noninteract': ['SI-STS']},
                'tom': {'belief-photo': ['TPJ']},
                  'communicate': {'body-object': ['EBA'],
                                  'face_third+face_noncom-object': ['fSTS', 'FFA'],
                                  'com_phy+phy+com_ind+ind+face_third+face_first+face_noncom+object+body': ['EVC', 'MT']}}


def split_into_groups(items, n_groups=3):
    return [items[i::n_groups] for i in range(n_groups)]


class NilearnGLMRunwise:
    def __init__(self, args):
        self.process = 'NilearnGLMRunwise'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.parcel_path = f'{self.derivatives_path}/parcels-{self.space_label}'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.TR = 2
        self.overlap_method = args.overlap_method
        print(vars(self))

    def load_mask(self):
        return load_brain_mask(self.fmriprep_path, self.subject_label, self.task_label, self.space_label)

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

        all_runs = [i for i in range(len(imgs))]
        n_groups_eff = min(n_groups[self.task_label], len(imgs))
        run_groups = split_into_groups(all_runs, n_groups=n_groups_eff)
        for igroup, run_group in tqdm(enumerate(run_groups),
                                 total=n_groups_eff, desc='fitting run groups'):
            # Compute the model and contrasts to define the fROIs
            froi_imgs = [imgs[r] for r in all_runs if r not in run_group]
            froi_events = [events_shifted[r] for r in all_runs if r not in run_group]
            froi_confounds = [confounds[r] for r in all_runs if r not in run_group]
            if froi_imgs:
                model.fit(froi_imgs, froi_events, froi_confounds)
                contrast_files = {}
                for contrast_name in froi_contrasts[self.task_label].keys():
                    title = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{contrast_name}_run-{igroup+1}'

                    if '-' in contrast_name:
                        c1, c2 = contrast_name.split('-')
                    else:
                        c1, c2 = contrast_name, None
                    contrast = parse_contrast(model, c1, c2)
                    print(contrast)
                    contrast_file = f'{self.out_path}/sub-{self.subject_label}/{title}.nii.gz'
                    stat_map = model.compute_contrast(contrast, output_type='z_score')
                    print(stat_map.shape)
                    nib.save(stat_map, contrast_file)
                    print('Saved contrast stat map to', contrast_file)
                    contrast_files[contrast_name] = contrast_file

                # ROIs defined from different contrasts within this task/fold can still
                # spatially overlap (e.g. fSTS/FFA/EBA/EVC/MT parcels), so voxels raw
                # top-N%-selected by more than one ROI are resolved here, following
                # video_sentence_analysis's build_froi_masks.py winner-take-all
                # approach: each contested voxel goes to whichever competing ROI has
                # the higher z-stat *in its own defining contrast* at that voxel. This
                # only arbitrates within one task's fold (same held-in run set for every
                # ROI compared here) -- ROIs defined in a different task (SI-STS from
                # pointlight, TPJ from tom) use an unrelated fold structure and are not
                # cross-task arbitrated, to avoid leaking a held-out run's data into
                # another task's fold-specific ROI definition.
                for hemi in ['l', 'r']:
                    masks, stats = {}, {}
                    for contrast_name, rois_for_contrast in froi_contrasts[self.task_label].items():
                        contrast_file = contrast_files[contrast_name]
                        stat_data = nib.load(contrast_file).get_fdata()
                        for roi in rois_for_contrast:
                            mask_file = f'{self.parcel_path}/{hemi}{roi_switcher(roi)}.nii.gz'
                            masks[roi] = selective_mask_img(mask_file, contrast_file,
                                                             keep_prop=roi_size[roi],
                                                             return_nifti=False).astype(bool)
                            stats[roi] = stat_data

                    if len(masks) > 1 and self.overlap_method == 'winner_take_all':
                        masks = winner_take_all(masks, stats)
                    elif len(masks) > 1 and self.overlap_method == 'drop':
                        mask_stack = np.stack(list(masks.values()), axis=0)
                        contested = mask_stack.sum(axis=0) > 1
                        masks = {roi: m & ~contested for roi, m in masks.items()}

                    affine = nib.load(next(iter(contrast_files.values()))).affine
                    for roi, mask in masks.items():
                        output_file = f'{self.out_path}/sub-{self.subject_label}/sub-{self.subject_label}_run-{igroup+1}_{hemi}{roi}'
                        nib.save(nib.Nifti1Image(mask.astype(np.int8), affine), f'{output_file}.nii.gz')

            # Compute the model and contrasts to estimate the responses
            resp_imgs = [imgs[r] for r in all_runs if r in run_group]
            resp_events = [events_shifted[r] for r in all_runs if r in run_group]
            resp_confounds = [confounds[r] for r in all_runs if r in run_group]
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
    parser.add_argument('--task_label', '-t', type=str, default='communicate',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overlap_method', type=str, choices=['winner_take_all', 'drop', 'none'],
                         default='winner_take_all',
                         help='How to resolve fROI voxels independently selected by more than '
                              "one ROI within this task's fold. 'none' disables resolution.")
    args = parser.parse_args()
    args.overlap_method = None if args.overlap_method == 'none' else args.overlap_method

    processor = NilearnGLMRunwise(args)
    processor.run()

if __name__ == '__main__':
    main()