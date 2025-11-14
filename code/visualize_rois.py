import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn.plotting import plot_roi, plot_surf_roi
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import vol_to_surf
from itertools import product


class VisualizeROIs:
    def __init__(self, args):
        self.process = 'VisualizeROIs'
        self.subject_label = str(args.subject_label).zfill(2)
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.roi_path = f'{self.dataset_path}/derivatives/NilearnGLMRunwise/sub-{self.subject_label}'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.out_file = f'{self.out_path}/sub-{self.subject_label}_fROIs'
        self.hemis = ['l', 'r']
        self.rois = ['EVC', 'MT', 'FFA', 'EBA', 'fSTS',
                     'SI-STS', 'TPJ', 'facecom-STS', 'dyadcom-STS']
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def load_rois(self):
        affine = None
        for hemi in self.hemis: 
            for iroi, roi in tqdm(enumerate(self.rois), 
                                  desc=f'Loading {hemi} hemisphere',
                                  total=len(self.rois)):
                roi_file_name = f'{self.roi_path}/sub-{self.subject_label}_run-1_{hemi}{roi}.nii.gz'
                if affine is None:
                    roi_img = nib.load(roi_file_name)
                    affine = roi_img.affine
                    roi_arr = roi_img.get_fdata().astype('bool')
                    rois_img = np.zeros_like(roi_arr, dtype='float')
                else:
                    roi_arr = nib.load(roi_file_name).get_fdata().astype('bool')
                rois_img[roi_arr] = iroi + 1
        return nib.Nifti1Image(rois_img, affine=affine)
    
    def plot_rois(self, rois_img):
        title = f'fROIs in sub-{self.subject_label}'
        plot_roi(rois_img,
                 colorbar=True,
                 title=title,
                 display_mode='mosaic',
                 cmap='Set1',
                 draw_cross=False,
                 output_file=f'{self.out_file}.png')
        

        fsaverage_meshes = load_fsaverage()
        fsaverage_sulcal = load_fsaverage_data(data_type='sulcal')
        for hemi, view in product(['left', 'right'], ['lateral', 'ventral', 'medial']):
            colorbar = True if view == 'lateral' else False
            surf_mesh = fsaverage_meshes['pial'].parts[hemi]
            surf_rois = vol_to_surf(rois_img,
                                    surf_mesh,
                                    n_samples=2,
                                    radius=0.0,
                                    interpolation='nearest')
            fig = plot_surf_roi(surf_mesh=fsaverage_meshes['inflated'],
                                roi_map=surf_rois,
                                hemi=hemi,
                                view=view,
                                engine='plotly',
                                bg_map=fsaverage_sulcal,
                                bg_on_data=True,
                                colorbar=colorbar,
                                threshold=0.5,
                                cmap='Set1',
                                title=title)
            
            # Customize colorbar with ROI labels
            if colorbar:
                # Directly modify the colorbar of the mesh trace
                fig.figure.data[0].colorbar = dict(
                    tickmode='array',
                    tickvals=list(range(1, len(self.rois) + 1)),
                    ticktext=self.rois,
                    title='ROI'
                )
                # Set color range to start at 1
                fig.figure.data[0].cmin = 1
                fig.figure.data[0].cmax = len(self.rois)
            
            fig.figure.write_image(f'{self.out_file}_view-{view}_hemi-{hemi}.png')

    def run(self):
        rois_img = self.load_rois()
        self.plot_rois(rois_img)


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=int, default=2,
                         help='Subject for the GLM')
    args = parser.parse_args()

    processor = VisualizeROIs(args)
    processor.run()


if __name__ == '__main__':
    main()
