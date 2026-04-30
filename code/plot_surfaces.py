import argparse
import os
from pathlib import Path
from itertools import product

import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.mri import vol2surf
from nilearn.plotting import plot_surf_stat_map, plot_glass_brain


class PlotSurfaces:
    def __init__(self, args):
        self.condition_one = args.condition_one
        self.condition_two = args.condition_two
        self.task = args.task # Nonfunctional but added for consistency with other classes
        self.palette = sns.color_palette(args.palette_name, as_cmap=True)
        self.threshold = None
        self.file_prefix = f'contrast-{self.condition_one}-{self.condition_two}.nii.gz'
        self.dataset_path = args.dataset_path
        self.alpha = 0.05
        self.logp = -1*np.log10(self.alpha)  # Threshold for p < 0.05 in -log10 space
        self.threshold = None
        self.data_top_dir = os.path.join(self.dataset_path, 'derivatives')
        self.random_effects_dir = os.path.join(self.data_top_dir, 'GroupRandomEffects')
        self.outpath = os.path.join(self.data_top_dir, 'SurfacePlots')
        Path(self.outpath).mkdir(exist_ok=True, parents=True)

    def load_stat_img(self):
        tval_file = os.path.join(self.random_effects_dir, self.file_prefix.replace('.nii.gz', '_stat-t.nii.gz'))
        pval_file = os.path.join(self.random_effects_dir, self.file_prefix.replace('.nii.gz', '_stat-logp_max_tfce.nii.gz'))
        stat_img = nib.load(tval_file)
        p_img = nib.load(pval_file)
        stat_img_data = stat_img.get_fdata()
        p_img_data = p_img.get_fdata()
        # Determine threshold based on p-values
        self.threshold = np.min(stat_img_data[p_img_data > self.logp]) if np.any(p_img_data > self.logp) else None
        return stat_img, p_img

    def plot_stat_map(self, stat_img):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_glass_brain(
            stat_img,
            threshold=self.threshold,
            vmin=self.threshold,
            cmap=self.palette,
            colorbar=True,
            plot_abs=False,
            axes=ax
            )

        # Apply tight layout and save
        plt.savefig(os.path.join(self.outpath, f'{self.condition_one}_vs_{self.condition_two}_volume.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_surfs(self, img, fsaverage_meshes, fsaverage_sulcal):
        for hemi, view in product(['left', 'right'], ['lateral', 'ventral']):
            fig = plot_surf_stat_map(
                stat_map=img,
                surf_mesh=fsaverage_meshes["inflated"],
                hemi=hemi,
                view=view,
                threshold=self.threshold,
                vmin=self.threshold,
                bg_map=fsaverage_sulcal,
                darkness=None,
                cmap=self.palette
            )

            # Add colorbar label
            cbar = fig.axes[-1]  # Get the colorbar axis
            cbar.set_ylabel('t-value', rotation=270, labelpad=20)

            # Make background transparent
            fig.patch.set_alpha(0)
            for ax in fig.axes:
                ax.patch.set_alpha(0)

            # Apply tight layout and save
            fig.savefig(os.path.join(self.outpath, f'{self.condition_one}_vs_{self.condition_two}_surface_{hemi}_{view}.png'), 
                        transparent=True, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot(self):
        stat_img, p_img = self.load_stat_img()
        self.plot_stat_map(stat_img)
        img, fsaverage_meshes, fsaverage_sulcal = vol2surf(stat_img)
        self.plot_surfs(img, fsaverage_meshes, fsaverage_sulcal)


def main():
    parser = argparse.ArgumentParser(description='Plot pretty surface maps for a given contrast')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='interact',
                         help='The first condition for the second level analysis')
    parser.add_argument('--condition_two', '-c2', type=str, default='noninteract',
                         help='The second condition for the second level analysis')
    parser.add_argument('--palette_name', type=str, default='Reds',
                         help='Seaborn color palette name')
    parser.add_argument('--task', '-t', type=str, default='pointlight',
                         help='Task label for the analysis')
    args = parser.parse_args()
    PlotSurfaces(args).plot()

if __name__ == '__main__':
    main()