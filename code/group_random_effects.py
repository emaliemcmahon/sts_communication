import argparse
import os
from tqdm import tqdm
from pathlib import Path
from nilearn.plotting import plot_glass_brain, view_img_on_surf
from nilearn.glm.first_level import first_level_from_bids as flfb
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from nilearn.image import threshold_img


class GroupRandomEffects:
    def __init__(self, args):
        self.process = 'GroupRandomEffects'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.fmriprep_path = f'{self.derivatives_path}/fmriprep'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.condition_one = args.condition_one
        self.condition_two = args.condition_two
        self.contrast = f'{self.condition_one}-{self.condition_two}'
        self.task_label = args.task_label
        self.space_label = args.space_label
        self.sub_nums = args.sub_nums
        self.subjs = [str(i).zfill(2) for i in self.sub_nums]
        self.threshold = norm.isf(0.001)
        print(vars(self))
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

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
        axes = np.atleast_2d(axes)
        model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
        for midx, (model, imgs, events, confounds) in tqdm(enumerate(model_and_args),
                                                           total=len(models),
                                                           leave=True,
                                                           desc='First level models'):            
            # fit the GLM
            model.fit(imgs, events, confounds)
            tmap = model.compute_contrast(self.contrast,
                                          stat_type='t',
                                          output_type='stat')
            colorbar = True if midx == len(self.subjs)-1 else False
            plot_glass_brain(tmap,
                             colorbar=colorbar,
                             threshold=self.threshold,
                             title=f"sub-{model.subject_label}",
                             axes=axes[int(midx / ncols), int(midx % ncols)],
                             display_mode="x",
                             cmap="bwr")
        fig.suptitle(f"T-Map {self.condition_one} vs {self.condition_two} (unc p<0.001)")
        plt.savefig(f'{self.out_path}/{self.contrast}_individuals.png')
        print('Finished first level analyses')


        second_level_model = SecondLevelModel(smoothing_fwhm=8.0, 
                                              n_jobs=int(os.cpu_count()/2))
        second_level_model = second_level_model.fit(models)

        tmap = second_level_model.compute_contrast(first_level_contrast=self.contrast, 
                                                   output_type='stat',
                                                   second_level_stat_type='t')
        tmap_thresholded = threshold_img(tmap, threshold=self.threshold,
                                         copy_header=True)
        title = f"{self.condition_one} vs {self.condition_two} (unc p<0.001)"
        plot_glass_brain(tmap_thresholded,
                         threshold=self.threshold,
                         colorbar=True,
                         plot_abs=False,
                         title=title,
                         cmap="bwr",
                         output_file=f'{self.out_path}/{self.contrast}_group.png')
        view = view_img_on_surf(tmap_thresholded, 
                                threshold=self.threshold, 
                                bg_on_data=True,
                                darkness=0.5,
                                title=title)
        view.save_as_html(f'{self.out_path}/{self.contrast}_group.html')
        print('Finished second level analysis')


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('sub_nums', nargs='*', type=int, 
                        help='List of elements', default=[1,2])
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='com_phy',
                         help='The first condition for the second level analysis')
    parser.add_argument('--condition_two', '-c2', type=str, default='phy',
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