import argparse
import os
from pathlib import Path
from itertools import product

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import cortex

from utils.mri import roi_switcher, vol2surf_int, selective_mask_img, roi_size
from nilearn.datasets import fetch_surf_fsaverage

task_contrasts = {'communicate': {'com_phy+phy+com_ind+ind+face_third+face_first+face_noncom+object+body': ['EVC', 'MT']}, 
                  'tom': {'belief-photo': ['TPJ']}}


class PlotROISurfaces:
    def __init__(self, args):
        self.process = 'PlotROISurfaces'
        self.subject = args.subject
        self.dataset_path = args.dataset_path
        self.space_label = args.space_label
        self.outpath = os.path.join(self.dataset_path, 'derivatives', self.process)
        Path(self.outpath).mkdir(exist_ok=True, parents=True)
        self.data_dir = os.path.join(self.dataset_path, 'derivatives', 'NilearnGLM', f'sub-{self.subject}')
        self.parcel_path = f'{self.dataset_path}/derivatives/parcels-{self.space_label}'

    def load_mask(self, task, contrast, roi):
        """Load and combine ROI masks from both hemispheres"""
        out = None
        for hemi in ['l', 'r']:
            contrast_file = os.path.join(self.data_dir, f'task-{task}', f'contrast-{contrast}_stat-tmap.nii.gz')
            mask_file = f'{self.parcel_path}/{hemi}{roi_switcher(roi)}.nii.gz'
            new_mask = selective_mask_img(mask_file, contrast_file, 
                                            keep_prop=roi_size[roi],
                                            return_nifti=False)
            if out is None:
                out = new_mask.copy()
            else:
                out += new_mask
        return out

    def plot_surf(self, roi_surfaces, roi2int):
        """Plot ROI surfaces using PyCortex web viewer"""
        print('Creating vertex data for PyCortex...')
        
        # Collect data across all ROIs
        left_data = np.zeros_like(roi_surfaces[0].data.parts['left'])
        right_data = np.zeros_like(roi_surfaces[0].data.parts['right'])
        
        for surf in roi_surfaces:
            left_data += surf.data.parts['left']
            right_data += surf.data.parts['right']
        
        # Create PyCortex Vertex object
        data = np.concatenate([left_data, right_data])
        vertex = cortex.Vertex(data, subject='fsaverage')
        
        # Create flatmap figure
        fig = cortex.quickflat.make_figure(vertex, with_colorbar=False, with_labels=False)
        
        # Save the figure
        fig.savefig(os.path.join(self.outpath, f'{self.subject}_surface_flat.png'), transparent=True, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def get_nifti_info(self):
        """Load a reference NIfTI file to get affine and header"""
        ref_file = f'{self.parcel_path}/lEVC.nii.gz'
        img = nib.load(ref_file)
        return img.affine, img.header

    def load_fsaverage_meshes(self):
        """Load fsaverage meshes - white, pial, and sulcal data"""
        print('Loading fsaverage meshes...')
        fsaverage = fetch_surf_fsaverage(mesh='fsaverage7')
        
        # Return both white (inner) and pial (outer) meshes for depth sampling
        meshes = {
            "white_left": fsaverage.white_left,
            "pial_left": fsaverage.pial_left,
            "white_right": fsaverage.white_right,
            "pial_right": fsaverage.pial_right,
        }
        sulcal = fsaverage.sulc_left
        
        return meshes, sulcal

    def load_and_project_rois(self):
        """Load ROI masks, project to surface, and binarize"""
        affine, header = self.get_nifti_info()
        
        # Load fsaverage meshes ONCE at the start
        fsaverage_meshes, fsaverage_sulcal = self.load_fsaverage_meshes()

        roi_surfaces = []
        roi2int = dict()
        roi_counter = 0 
        
        for task, contrasts_dict in task_contrasts.items():
            for contrast, rois in contrasts_dict.items():
                for roi in rois:
                    roi_counter += 1
                    print(f'Loading mask for task {task}, contrast {contrast}, roi {roi}')
                    mask_img = self.load_mask(task, contrast, roi)
                    mask_img = nib.Nifti1Image(mask_img, affine, header)
                    
                    # Project volume to surface
                    img = vol2surf_int(mask_img, output_int=roi_counter, 
                                    fsaverage_meshes=fsaverage_meshes,
                                    fsaverage_sulcal=fsaverage_sulcal)

                    roi2int[roi] = roi_counter
                    roi_surfaces.append(img)

        return roi_surfaces, roi2int
      

    def plot(self):
        """Main plotting pipeline"""
        roi_surfaces, roi2int = self.load_and_project_rois()
        self.plot_surf(roi_surfaces, roi2int)


def main():
    parser = argparse.ArgumentParser(description='Plot ROI masks on surfaces for a given subject')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, default='01', help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym', 
                        help='Space label (e.g., fsaverage)')
    args = parser.parse_args()
    PlotROISurfaces(args).plot()


if __name__ == '__main__':
    main()