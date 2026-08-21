import argparse
import os
from pathlib import Path
from itertools import product

import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

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
        random_effects_dir = f'GroupRandomEffects_{args.out_tag}' if args.out_tag else 'GroupRandomEffects'
        out_dir = f'SurfacePlots_{args.out_tag}' if args.out_tag else 'SurfacePlots'
        self.random_effects_dir = os.path.join(self.data_top_dir, random_effects_dir)
        self.outpath = os.path.join(self.data_top_dir, out_dir)
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

    def plot_surfs(self, img, fsaverage_meshes, fsaverage_sulcal, vmin, vmax):
        items = list(product(['left', 'right'], ['lateral', 'ventral']))
        for hemi, view in tqdm(items, desc='Plotting surfaces', unit='plot'):
            fig = plot_surf_stat_map(
                stat_map=img,
                surf_mesh=fsaverage_meshes["inflated"],
                hemi=hemi,
                view=view,
                threshold=self.threshold,
                vmin=vmin,
                vmax=vmax,
                bg_map=fsaverage_sulcal,
                darkness=None,
                cmap=self.palette
            )

            # Add colorbar label
            cbar = fig.axes[-1]  # Get the colorbar axis
            cbar.set_ylabel('t-value', rotation=270, labelpad=10, fontsize=22,
                            va='bottom')
            cbar.yaxis.label.set_rotation(270)

            # Adjust colorbar height to make it less tall
            pos = cbar.get_position()
            cbar.set_position([pos.x0, pos.y0 + pos.height * 0.2, pos.width, pos.height * 0.8])

            # Set ticks to be equally distributed across the shared vmin/vmax range
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ylim(vmin, vmax)
            cbar.set_yticks(ticks)
            cbar.set_yticklabels([f'{t:.1f}' for t in ticks])

            # Make tick labels bigger
            cbar.tick_params(axis='y', labelsize=18)

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
        # Use vmin/vmax from the volume so left and right hemi colorbars match
        vol_data = stat_img.get_fdata()
        vmax = 0.9 * float(np.nanmax(vol_data))
        vmin = self.threshold if self.threshold is not None else float(np.nanmin(vol_data))
        img, fsaverage_meshes, fsaverage_sulcal = vol2surf(stat_img)
        self.plot_surfs(img, fsaverage_meshes, fsaverage_sulcal, vmin, vmax)


def main():
    parser = argparse.ArgumentParser(description='Plot pretty surface maps for a given contrast')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='face_third+face_first+com_phy+com_ind',
                         help='The first condition for the second level analysis')
    parser.add_argument('--condition_two', '-c2', type=str, default='face_noncom+phy+ind',
                         help='The second condition for the second level analysis')
    parser.add_argument('--palette_name', type=str, default='magma',
                         help='Seaborn color palette name')
    parser.add_argument('--task', '-t', type=str, default='communicate',
                         help='Task label for the analysis')
    parser.add_argument('--out_tag', type=str, default='',
                         help='If set, read from derivatives/GroupRandomEffects_<out_tag>/ and write to '
                              'derivatives/SurfacePlots_<out_tag>/ instead of the unsuffixed directories.')
    args = parser.parse_args()
    PlotSurfaces(args).plot()

if __name__ == '__main__':
    main()