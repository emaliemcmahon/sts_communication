import argparse
import os
import warnings
from glob import glob
from tqdm import tqdm
from pathlib import Path
from nilearn.glm.second_level import SecondLevelModel
from nilearn.mass_univariate import permuted_ols
import numpy as np
import nibabel as nib


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
        self.effect_files = sorted(glob(f'{self.glm_path}/sub-*/task-{self.task_label}/contrast-{self.contrast_name}_effect-size.nii.gz'))
        self.subjs = [Path(f).parent.parent.name.replace('sub-', '') for f in self.effect_files]
        
        print(f'Found {len(self.effect_files)} subjects with contrast {self.contrast_name} for task {self.task_label}: {self.subjs}')
        Path(f'{self.out_path}/sub-group').mkdir(parents=True, exist_ok=True)

    def glm(self):
        # Second-level analysis using permutation testing
        print('Running permutation-based second-level analysis...')
        n_subjects = len(self.effect_files)
        design = np.ones((n_subjects, 1))  # Test group mean
        
        # Load and stack effect size maps
        effect_imgs = [nib.load(f) for f in self.effect_files]
        data_4d = np.stack([img.get_fdata() for img in effect_imgs], axis=-1)
        affine = effect_imgs[0].affine
        shape_3d = data_4d.shape[:3]
        data_2d = data_4d.reshape(-1, n_subjects)  # (n_voxels, n_subjects)
        
        # Run permutation testing
        t_scores, p_values = permuted_ols(design, data_2d.T,  # permuted_ols expects (n_samples, n_features)
                                          confounding_vars=None, 
                                          n_perm=1000, 
                                          n_jobs=int(os.cpu_count()/2))
        
        # Reshape back to 3D
        t_scores_3d = t_scores.reshape(shape_3d)
        p_values_3d = p_values.reshape(shape_3d)
        
        # Create nibabel images
        tmap = nib.Nifti1Image(t_scores_3d, affine)
        pmap = nib.Nifti1Image(p_values_3d, affine)
        
        # Save results
        nib.save(tmap, f'{self.out_path}/sub-group/contrast-{self.contrast_name}_stat-tmap.nii.gz')
        nib.save(pmap, f'{self.out_path}/sub-group/contrast-{self.contrast_name}_p-values.nii.gz')
        
        # Threshold at p < 0.05 (FWE corrected by permutation)
        thresholded = np.where(p_values_3d < 0.05, t_scores_3d, 0)
        tmap_thresholded = nib.Nifti1Image(thresholded, affine)
        
        # Save thresholded map
        nib.save(tmap_thresholded, f'{self.out_path}/sub-group/contrast-{self.contrast_name}_thresholded.nii.gz')
        print('Finished second level analysis')


def main():
    parser = argparse.ArgumentParser(description='Run second-level GLM from first-level contrast outputs')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, required=True,
                         help='The first condition for the contrast')
    parser.add_argument('--condition_two', '-c2', type=str, required=True,
                         help='The second condition for the contrast')
    parser.add_argument('--task_label', '-t', type=str, required=True,
                         help='Task label (communicate, tom, pointlight)')
    args = parser.parse_args()

    processor = GroupRandomEffects(args)
    processor.glm()

if __name__ == '__main__':
    main()