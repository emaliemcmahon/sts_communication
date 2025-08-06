import argparse
import os
from tqdm import tqdm
from pathlib import Path
from nilearn.plotting import plot_glass_brain, view_img_on_surf
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
import matplotlib.pyplot as plt
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_contrast_matrix
import nibabel as nib

class GroupRandomEffects:
    def __init__(self, args):
        self.process = 'GroupRandomEffects'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.condition_one = args.condition_one
        self.condition_two = args.condition_two
        self.contrast_name = f'{self.condition_one}-{self.condition_two}'
        self.contrast = f'{self.condition_one}-{self.condition_two}'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.sub_nums = args.sub_nums
        self.subjs = [str(i).zfill(2) for i in self.sub_nums]
        self.alpha = 0.001
        self.correction = None
        print(vars(self))
        Path(f'{self.out_path}/sub-group').mkdir(parents=True, exist_ok=True)

    def glm(self):
        model_info = flfb(self.dataset_path,
                          self.task_label,
                          self.space_label,
                          sub_labels=self.subjs,
                          slice_time_ref=None, # Load from the BIDS data
                          smoothing_fwhm=5.0,
                          img_filters=[("desc", "preproc")],
                          confounds_strategy=('motion', 'scrub'),
                          confounds_motion='basic',
                          derivatives_folder=self.fmriprep_path,
                          minimize_memory=True, 
                          hrf_model='spm',
                          confounds_fd_threshold=0.5, #FD in mm
                          confounds_scrub=5, #remove segments shorter than the given number after scrubbing
                          n_jobs=int(os.cpu_count()/2))
        (models, models_run_imgs, models_events, models_confounds) = model_info

        ncols = 3
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,10))
        title = f'T-Map {self.condition_one} vs {self.condition_two} (FDR q<{self.alpha})'
        axes = np.atleast_2d(axes)
        model_and_args = zip(self.subjs, models, models_run_imgs, models_events, models_confounds)
        for midx, (subj, model, imgs, events, confounds) in tqdm(enumerate(model_and_args),
                                                           total=len(models),
                                                           leave=True,
                                                           desc='First level models'):            
            Path(f'{self.out_path}/sub-{subj}').mkdir(exist_ok=True, parents=True)
            # fit the GLM
            model.fit(imgs, events, confounds)
            if type(self.contrast) is str: 
                columns = list(model.design_matrices_[0].columns)
                self.contrast = np.zeros(len(columns))

                cond1_averaging = self.condition_one.split('+')
                for c in cond1_averaging:
                    self.contrast[columns.index(c)] = 1/len(cond1_averaging)

                cond2_averaging = self.condition_two.split('+')
                for c in cond2_averaging:
                    self.contrast[columns.index(c)] = -1/len(cond2_averaging)

                # Save a visualization of the contrast matrix
                plot_contrast_matrix(self.contrast, model.design_matrices_[0],
                                 output_file=f'{self.out_path}/{self.contrast_name}_design.png')

            tmap = model.compute_contrast(self.contrast,
                                          stat_type='t',
                                          output_type='stat')
            tmap_thresholded, threshold = threshold_stats_img(tmap, 
                                                              alpha=self.alpha,
                                                              height_control=self.correction)
            plot_glass_brain(tmap_thresholded,
                             colorbar=True,
                             threshold=threshold,
                             title=f"sub-{model.subject_label}",
                             axes=axes[int(midx / ncols), int(midx % ncols)],
                             display_mode="x",
                             cmap="bwr")
            view = view_img_on_surf(tmap_thresholded, 
                                    surf_mesh='fsaverage',
                                    threshold=threshold,
                                    bg_on_data=True,
                                    darkness=0.5,
                                    title=title, 
                                    colorbar_height=0.75)
            nib.save(tmap, f'{self.out_path}/sub-{subj}/sub-{subj}_contrast-{self.contrast_name}_stat-tmap.nii.gz')
            view.save_as_html(f'{self.out_path}/sub-{subj}/sub-{subj}_{self.contrast_name}.html')
        fig.suptitle(title)
        plt.savefig(f'{self.out_path}/sub-group/{self.contrast_name}_individuals.png')
        print('Finished first level analyses')

        second_level_model = SecondLevelModel(smoothing_fwhm=8.0, 
                                              n_jobs=int(os.cpu_count()/2))
        second_level_model = second_level_model.fit(models)

        tmap = second_level_model.compute_contrast(first_level_contrast=self.contrast, 
                                                   output_type='stat',
                                                   second_level_stat_type='t')
        nib.save(tmap, f'{self.out_path}/sub-group/contrast-{self.contrast_name}_stat-tmap.nii.gz')
        tmap_thresholded, threshold = threshold_stats_img(tmap,
                                                          alpha=self.alpha,
                                                          height_control=self.correction)
        title = f"{self.condition_one} vs {self.condition_two} (FDR q<{self.alpha})"
        plot_glass_brain(tmap_thresholded,
                         threshold=threshold,
                         colorbar=True,
                         plot_abs=False,
                         title=title,
                         cmap="bwr",
                         output_file=f'{self.out_path}/sub-group/contrast-{self.contrast_name}_stat-tmap.png')
        view = view_img_on_surf(tmap_thresholded, 
                                surf_mesh='fsaverage',
                                threshold=threshold,
                                bg_on_data=True,
                                darkness=0.5,
                                title=title, 
                                colorbar_height=0.75)
        view.save_as_html(f'{self.out_path}/sub-group/contrast-{self.contrast_name}_stat-tmap.html')
        print('Finished second level analysis')


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('sub_nums', nargs='*', type=int, 
                        help='List of elements', default=[1,2,3])
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='com_ind',
                         help='The first condition for the second level analysis')
    parser.add_argument('--condition_two', '-c2', type=str, default='ind',
                         help='The second condition for the second level analysis')
    parser.add_argument('--task_label', '-t', type=str, default='communicate',
                         help='Task to run the GLM on')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = GroupRandomEffects(args)
    processor.glm()

if __name__ == '__main__':
    main()