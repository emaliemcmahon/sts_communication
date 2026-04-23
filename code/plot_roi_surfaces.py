import argparse
import os
from pathlib import Path
from itertools import product

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cortex
from nilearn import plotting as nplot

from utils.mri import roi_switcher, vol2surf_int, selective_mask_img, roi_size
from nilearn.datasets import fetch_surf_fsaverage

task_contrasts = {'communicate': {'com_phy+phy+com_ind+ind+face_third+face_first+face_noncom+object+body': ['MT'],
                                  'body-object': ['EBA'],
                                  'face_third+face_noncom-object': ['fSTS'],
                                  'face_third+face_first+com_phy+com_ind-face_noncom+phy+ind': ['com-STS']}, 
                    'pointlight': {'interact-noninteract': ['SI-STS']},
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
        """Load and combine ROI masks from both hemispheres.

        Returns None when required inputs are missing so callers can skip safely.
        """
        contrast_file = os.path.join(self.data_dir, f'task-{task}', f'contrast-{contrast}_stat-tmap.nii.gz')
        if not os.path.exists(contrast_file):
            print(f'WARNING: Missing contrast file, skipping ROI {roi}: {contrast_file}')
            return None

        out = None
        for hemi in ['l', 'r']:
            mask_file = f'{self.parcel_path}/{hemi}{roi_switcher(roi)}.nii.gz'
            if not os.path.exists(mask_file):
                print(f'WARNING: Missing parcel mask for ROI {roi} ({hemi} hemisphere), skipping hemi: {mask_file}')
                continue

            new_mask = selective_mask_img(mask_file, contrast_file, 
                                            keep_prop=roi_size[roi],
                                            return_nifti=False)
            if out is None:
                out = new_mask.copy()
            else:
                out += new_mask

        if out is None:
            print(f'WARNING: No valid hemispheres found for ROI {roi}; skipping.')
            return None

        return out

    def _build_plot_data_and_colors(self, roi_vertex_masks):
        """Build integer-coded ROI surface data and consistent colors."""
        roi_names = list(roi_vertex_masks.keys())
        if len(roi_names) == 0:
            raise RuntimeError('No ROI masks to plot.')

        n_vertices = next(iter(roi_vertex_masks.values())).shape[0]
        # Keep non-ROI vertices transparent so curvature remains visible.
        plot_data = np.full(n_vertices, np.nan, dtype=float)

        cmap_base = plt.get_cmap('tab20b', len(roi_names))
        roi_colors_text = [cmap_base(i) for i in range(len(roi_names))]
        roi_colors_plot = [(r, g, b, 0.6) for (r, g, b, _) in roi_colors_text]

        # Assign each ROI a unique integer ID.
        for idx, roi in enumerate(roi_names, start=1):
            roi_mask = roi_vertex_masks[roi] > 0
            plot_data[roi_mask & np.isnan(plot_data)] = idx

        return roi_names, plot_data, roi_colors_text, roi_colors_plot

    def _add_legend_text(self, fig, roi_names, roi_colors_text):
        """Add top-left legend text with current styling."""
        fig.text(
            0.03,
            0.95,
            'ROIs',
            ha='left',
            va='top',
            fontsize=24,
            fontweight='bold',
            color='black',
        )

        y_start = 0.92
        y_step = 0.037
        for idx, roi in enumerate(roi_names):
            y_pos = y_start - (idx * y_step)
            fig.text(
                0.03,
                y_pos,
                roi,
                ha='left',
                va='top',
                color=roi_colors_text[idx],
                fontsize=20,
                fontweight='bold',
            )

    def plot_flatmap(self, roi_vertex_masks):
        """Plot ROI masks on fsaverage flatmap with unique colors and legend labels."""
        print('Creating vertex data for PyCortex...')
        roi_names, plot_data, roi_colors_text, roi_colors_plot = self._build_plot_data_and_colors(roi_vertex_masks)

        vertex_data = cortex.Vertex(
            plot_data,
            'fsaverage',
            cmap=ListedColormap(roi_colors_plot),
            vmin=1,
            vmax=len(roi_names),
        )

        fig = cortex.quickflat.make_figure(
            vertex_data,
            with_colorbar=False,
            with_labels=False,
            with_sulci=False,
            with_curvature=True,
            with_rois=False,
            curvature_contrast=0.15,
            curvature_brightness=0.75,
        )
        self._add_legend_text(fig, roi_names, roi_colors_text)

        fig.savefig(
            os.path.join(self.outpath, f'{self.subject}_surface_flat.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
        )
        plt.close(fig)

    def plot_inflated_lateral(self, roi_vertex_masks, fsaverage):
        """Plot ROI masks on inflated lateral surfaces using nilearn."""
        print('Creating inflated lateral plots with nilearn...')
        roi_names, plot_data, roi_colors_text, roi_colors_plot = self._build_plot_data_and_colors(roi_vertex_masks)
        n_vertices = plot_data.shape[0]
        n_left = n_vertices // 2

        left_data = np.nan_to_num(plot_data[:n_left], nan=0.0)
        right_data = np.nan_to_num(plot_data[n_left:], nan=0.0)

        fig = plt.figure(figsize=(15, 7))
        ax_left = fig.add_subplot(1, 2, 1, projection='3d')
        ax_right = fig.add_subplot(1, 2, 2, projection='3d')

        nplot.plot_surf_stat_map(
            fsaverage.infl_left,
            left_data,
            hemi='left',
            view='lateral',
            bg_map=fsaverage.sulc_left,
            cmap=ListedColormap(roi_colors_plot),
            threshold=0.5,
            vmin=1,
            vmax=len(roi_names),
            colorbar=False,
            bg_on_data=True,
            axes=ax_left,
            figure=fig,
        )
        ax_left.set_title('Left Lateral', fontsize=12)

        nplot.plot_surf_stat_map(
            fsaverage.infl_right,
            right_data,
            hemi='right',
            view='lateral',
            bg_map=fsaverage.sulc_right,
            cmap=ListedColormap(roi_colors_plot),
            threshold=0.5,
            vmin=1,
            vmax=len(roi_names),
            colorbar=False,
            bg_on_data=True,
            axes=ax_right,
            figure=fig,
        )
        ax_right.set_title('Right Lateral', fontsize=12)

        self._add_legend_text(fig, roi_names, roi_colors_text)

        fig.savefig(
            os.path.join(self.outpath, f'{self.subject}_surface_inflated_lateral.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
        )
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
        
        return meshes, sulcal, fsaverage

    def load_and_project_rois(self):
        """Load ROI masks, project to surface, and return per-ROI vertex masks."""
        affine, header = self.get_nifti_info()
        
        # Load fsaverage meshes ONCE at the start
        fsaverage_meshes, fsaverage_sulcal, fsaverage = self.load_fsaverage_meshes()

        roi_vertex_masks = dict()
        
        for task, contrasts_dict in task_contrasts.items():
            for contrast, rois in contrasts_dict.items():
                for roi in rois:
                    print(f'Loading mask for task {task}, contrast {contrast}, roi {roi}')
                    mask_img = self.load_mask(task, contrast, roi)
                    if mask_img is None:
                        continue
                    mask_img = nib.Nifti1Image(mask_img, affine, header)
                    
                    # Project volume to surface
                    img = vol2surf_int(mask_img, 
                                    fsaverage_meshes=fsaverage_meshes,
                                    fsaverage_sulcal=fsaverage_sulcal)

                    # Store projected ROI vertices by ROI name.
                    mask_left = img.data.parts['left'] > 0
                    mask_right = img.data.parts['right'] > 0
                    mask = np.concatenate([mask_left.astype(float), mask_right.astype(float)])

                    if roi in roi_vertex_masks:
                        roi_vertex_masks[roi] = np.maximum(roi_vertex_masks[roi], mask)
                    else:
                        roi_vertex_masks[roi] = mask

        if len(roi_vertex_masks) == 0:
            raise RuntimeError('No ROIs were projected to surface; check task_contrasts and input files.')

        return roi_vertex_masks, fsaverage
      

    def plot(self):
        """Main plotting pipeline"""
        roi_vertex_masks, fsaverage = self.load_and_project_rois()
        self.plot_flatmap(roi_vertex_masks)
        self.plot_inflated_lateral(roi_vertex_masks, fsaverage)


def main():
    parser = argparse.ArgumentParser(description='Plot ROI masks on surfaces for a given subject')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/Users/emaliem/Dropbox/mit_projects/communication_in_sts/fmri_block_experiment/sts_communication')
    parser.add_argument('--subject', '-s', type=str, default='01', help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym', 
                        help='Space label (e.g., fsaverage)')
    args = parser.parse_args()
    PlotROISurfaces(args).plot()


if __name__ == '__main__':
    main()