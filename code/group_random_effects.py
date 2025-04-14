import argparse
import time
from pathlib import Path
from copy import copy
from nilearn.plotting import plot_glass_brain
from nilearn.glm.first_level import first_level_from_bids as flfb
import nibabel as nib
from nilearn.glm.second_level import SecondLevelModel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from nilearn.plotting import plot_design_matrix


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
        self.n_subjs = args.n_subjs
        self.subjs = [str(i+1).zfill(2) for i in range(self.n_subjs)]
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
                          minimize_memory=False, 
                          hrf_model='spm',
                          confounds_fd_threshold=0.5, #FD in mm
                          confounds_scrub=5, #remove segments shorter than the given number after scrubbing
                          n_jobs=-1)
        (models, models_run_imgs, models_events, models_confounds) = model_info

        ncols = 3
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 3))
        axes = np.atleast_2d(axes)
        model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
        for midx, (model, imgs, events, confounds) in enumerate(model_and_args):            
            # fit the GLM
            model.fit(imgs, events, confounds)
            zmap = model.compute_contrast(self.contrast)
            plot_glass_brain(zmap,
                             colorbar=False,
                             threshold=self.threshold,
                             title=f"sub-{model.subject_label}",
                             axes=axes[int(midx / ncols), int(midx % ncols)],
                             display_mode="x",
                             cmap="bwr")
        fig.suptitle(f"{self.condition_one} vs {self.condition_two} (unc p<.001)")
        plt.savefig(f'{self.out_path}/{self.contrast}_individuals.png')

        second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=2)
        second_level_model = second_level_model.fit(models)


        z_score = second_level_model.compute_contrast(first_level_contrast=self.contrast)
        plot_design_matrix(second_level_model.design_matrix_,
                           output_file=f'{self.out_path}/{self.contrast}_design.png')
        
        title = f"{self.condition_one} vs {self.condition_two} (unc p<0.001)"
        plot_glass_brain(z_score,
                         threshold=self.threshold,
                         plot_abs=False,
                         title=title,
                         output_file=f'{self.out_path}/{self.contrast}_group.png')


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='com_phy',
                         help='The first condition for the second level analysis')
    parser.add_argument('--condition_two', '-c2', type=str, default='phy',
                         help='The second condition for the second level analysis')
    parser.add_argument('--task_label', '-t', type=str, default='communicate',
                         help='Task to run the GLM on')
    parser.add_argument('--n_subjs', '-n', type=int, default=5,
                        help='the number of subjects to include')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = GroupRandomEffects(args)
    processor.glm()

if __name__ == '__main__':
    main()