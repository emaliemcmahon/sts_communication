import argparse
import os
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.mass_univariate import permuted_ols
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import norm
import time


class GroupRandomEffects:
    def __init__(self, args):
        self.process = 'GroupRandomEffects'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.glm_path = f'{self.derivatives_path}/NilearnGLM'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.condition_one = args.condition_one
        self.condition_two = args.condition_two
        self.contrast_name = f'{self.condition_one}-{self.condition_two}'
        self.task_label = args.task_label
        
        # Find all subjects that have the contrast file
        self.contrast_files = sorted(glob(f'{self.glm_path}/sub-*/task-{self.task_label}/contrast-{self.contrast_name}_stat-tmap.nii.gz'))
        self.subjs = [Path(f).parent.parent.name.replace('sub-', '') for f in self.contrast_files]
        
        print(f'Found {len(self.contrast_files)} subjects with contrast {self.contrast_name} for task {self.task_label}: {self.subjs}')
        Path(f'{self.out_path}').mkdir(parents=True, exist_ok=True)

    def glm(self):
        """Second-level analysis with non-parametric FWER correction."""
        print('Running non-parametric second-level analysis with FWER correction...')
        n_subjects = len(self.contrast_files)
        design = pd.DataFrame(np.ones((n_subjects, 1)), columns=['intercept'])
        
        # Use non_parametric_inference with proper contrast specification
        start_time = time.time()
        tfce_result = non_parametric_inference(
                                self.contrast_files,
                                design_matrix=design,
                                second_level_contrast='intercept',
                                model_intercept=True,
                                n_perm=10000,  
                                two_sided_test=False,
                                smoothing_fwhm=8,
                                tfce=True,
                                verbose=3,
                                n_jobs=-1
                            )
        end_time = time.time()
        print(f"Non-parametric inference took {end_time - start_time:.2f} seconds")

        # Save the FWER-corrected negative log p-values
        nib.save(tfce_result['t'], f'{self.out_path}/contrast-{self.contrast_name}_stat-t.nii.gz')
        nib.save(tfce_result['logp_max_t'], f'{self.out_path}/contrast-{self.contrast_name}_stat-logp_max_t.nii.gz')
        nib.save(tfce_result['tfce'], f'{self.out_path}/contrast-{self.contrast_name}_stat-tfce.nii.gz')
        nib.save(tfce_result['logp_max_tfce'], f'{self.out_path}/contrast-{self.contrast_name}_stat-logp_max_tfce.nii.gz')


def main():
    parser = argparse.ArgumentParser(description='Run second-level GLM from first-level contrast outputs')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='face_third+face_noncom',#required=True,
                         help='The first condition for the contrast')
    parser.add_argument('--condition_two', '-c2', type=str, default='object',#required=True,
                         help='The second condition for the contrast')
    parser.add_argument('--task_label', '-t', type=str, default='communicate',#required=True,
                         help='Task label (communicate, tom, pointlight)')
    args = parser.parse_args()

    processor = GroupRandomEffects(args)
    processor.glm()

if __name__ == '__main__':
    main()