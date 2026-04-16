import argparse
import os
from pathlib import Path

import nibabel as nib
import nilearn.plotting as nlp
import matplotlib.pyplot as plt
import seaborn as sns

from utils.mri import vol2surf
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_surf_stat_map


class PlotSurfaces:
    def __init__(self, args):
        self.condition_one = args.condition_one
        self.condition_two = args.condition_two
        self.alpha = args.alpha
        self.correction = args.correction
        self.two_sided = args.two_sided
        self.palette = sns.color_palette(args.palette_name, as_cmap=True)
        self.threshold = None
        self.file_name = f'contrast-{self.condition_one}-{self.condition_two}_stat-tmap.nii.gz'
        self.dataset_path = args.dataset_path
        self.data_top_dir = os.path.join(self.dataset_path, 'derivatives')
        self.data_path = os.path.join(self.data_top_dir, 'GroupRandomEffects', 'sub-group')
        self.outpath = os.path.join(self.data_top_dir, 'SurfacePlots')
        Path(self.outpath).mkdir(exist_ok=True, parents=True)

    def load_and_threshold_stat_img(self, file):
        stat_img = nib.load(os.path.join(self.data_path, file))
        tmap = nib.load(os.path.join(self.data_path, file))
        stat_img, threshold = threshold_stats_img(tmap, alpha=self.alpha, 
                                                height_control=self.correction, 
                                                two_sided=self.two_sided)
        self.threshold = threshold
        return stat_img

    def plot_stat_map(self, stat_img):
        fig = nlp.plot_stat_map(stat_img, threshold=self.threshold, 
                                display_mode='z', cut_coords=5, 
                                black_bg=False, cmap=self.palette)

        # Apply tight layout and save
        plt.savefig(os.path.join(self.outpath, f'{self.condition_one}_vs_{self.condition_two}_volume.png'), 
                    transparent=True, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_surfs(self, img, fsaverage_meshes, fsaverage_sulcal):
        for hemi in ['left', 'right']:
            fig = plot_surf_stat_map(
                stat_map=img,
                surf_mesh=fsaverage_meshes["inflated"],
                hemi=hemi,
                threshold=self.threshold,
                bg_map=fsaverage_sulcal,
                darkness=None,
                cmap=self.palette
            )

            # Add colorbar label
            cbar = fig.axes[-1]  # Get the colorbar axis
            cbar.set_ylabel('t-statistic', rotation=270, labelpad=20)

            # Make background transparent
            fig.patch.set_alpha(0)
            for ax in fig.axes:
                ax.patch.set_alpha(0)

            # Apply tight layout and save
            fig.savefig(os.path.join(self.outpath, f'{self.condition_one}_vs_{self.condition_two}_surface_{hemi}.png'), 
                        transparent=True, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot(self):
        stat_img = self.load_and_threshold_stat_img(self.file_name)
        self.plot_stat_map(stat_img)
        img, fsaverage_meshes, fsaverage_sulcal = vol2surf(stat_img)
        self.plot_surfs(img, fsaverage_meshes, fsaverage_sulcal)


def main():
    parser = argparse.ArgumentParser(description='Plot pretty surface maps for a given contrast')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--condition_one', '-c1', type=str, default='com_ind',
                         help='The first condition for the second level analysis')
    parser.add_argument('--condition_two', '-c2', type=str, default='ind',
                         help='The second condition for the second level analysis')
    parser.add_argument('--alpha', type=float, default=0.05,
                         help='Alpha level for thresholding')
    parser.add_argument('--correction', type=str, default='bonferroni',
                         help='Multiple comparison correction method')
    parser.add_argument('--two_sided', action=argparse.BooleanOptionalAction, default=False,
                         help='Whether to perform a two-sided test')
    parser.add_argument('--palette_name', type=str, default='Reds',
                         help='Seaborn color palette name')
    args = parser.parse_args()
    PlotSurfaces(args).plot()

if __name__ == '__main__':
    main()