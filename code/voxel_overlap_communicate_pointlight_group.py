import argparse
import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HEMISPHERE_ORDER = ['left', 'right']
CONTRAST_ORDER = ['com_ind-ind', 'com_phy-phy', 'face_first-face_noncom', 'face_third-face_noncom']
COMPARISON_LABEL = 'communicate vs pointlight'
CEILING_LABEL = 'pointlight split-half (ceiling)'
PALETTE = {COMPARISON_LABEL: 'darkorange', CEILING_LABEL: 'gray'}


class VoxelOverlapCommunicatePointlightGroup:
    """
    Aggregate per-subject Dice tables from voxel_overlap_communicate_pointlight.py
    (each communicate contrast vs. pointlight interact-noninteract) and from
    voxel_overlap_pointlight_splithalf.py (pointlight interact-noninteract,
    odd vs even runs) into group tables, plus a combined summary plot that
    overlays the split-half Dice as a noise-ceiling reference curve on every
    communicate-vs-pointlight panel. Split-half reliability is how much the
    pointlight contrast overlaps with *itself* across independent halves of
    the data, which upper-bounds how much overlap you could plausibly see
    with any other, independent contrast.
    """

    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.communicate_pointlight_path = os.path.join(
            self.dataset_path, 'derivatives', 'VoxelOverlapCommunicatePointlight')
        self.splithalf_path = os.path.join(
            self.dataset_path, 'derivatives', 'VoxelOverlapPointlightSplitHalf')

    @staticmethod
    def _load_tables(path, script_name):
        files = sorted(glob(os.path.join(path, 'sub-*', 'sub-*_dice_coefficients.csv')))
        if not files:
            raise RuntimeError(
                f'No per-subject Dice CSVs found under {path}. Run {script_name} for each subject first.'
            )
        print(f'Found {len(files)} per-subject Dice tables in {path}')
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    def load_subject_tables(self):
        communicate_df = self._load_tables(self.communicate_pointlight_path,
                                            'voxel_overlap_communicate_pointlight.py')
        splithalf_df = self._load_tables(self.splithalf_path,
                                          'voxel_overlap_pointlight_splithalf.py')
        return communicate_df, splithalf_df

    def save_group_tables(self, communicate_df, splithalf_df):
        communicate_outfile = os.path.join(self.communicate_pointlight_path, 'group_dice_coefficients.csv')
        communicate_df.to_csv(communicate_outfile, index=False)
        print(f'Saved {communicate_outfile}')

        splithalf_outfile = os.path.join(self.splithalf_path, 'group_dice_coefficients.csv')
        splithalf_df.to_csv(splithalf_outfile, index=False)
        print(f'Saved {splithalf_outfile}')

    def plot_splithalf_summary(self, splithalf_df):
        """Standalone pointlight split-half reliability plot (the ceiling on its own)."""
        plot_df = splithalf_df.copy()
        plot_df['percent_label'] = (plot_df['percent_threshold'] * 100).round().astype(int)
        hemispheres = [h for h in HEMISPHERE_ORDER if h in plot_df['hemisphere'].unique()]

        fig, axes = plt.subplots(1, len(hemispheres), figsize=(6.5 * len(hemispheres), 6),
                                  sharey=True, squeeze=False)
        axes = axes[0]
        for ax, hemi in zip(axes, hemispheres):
            sub_df = plot_df[plot_df['hemisphere'] == hemi]
            sns.pointplot(data=sub_df, x='percent_label', y='dice', errorbar='se', ax=ax, color='gray')
            sns.stripplot(data=sub_df, x='percent_label', y='dice', alpha=0.25, ax=ax, color='gray')
            ax.set_title(f'{hemi} hemisphere')
            ax.set_xlabel('Top % of STS parcel (by effect size)')

        axes[0].set_ylabel('Dice coefficient (odd vs even runs)')
        fig.suptitle('Pointlight interact-noninteract: split-half reliability within STS')
        fig.tight_layout()
        outfile = os.path.join(self.splithalf_path, 'dice_group_summary.png')
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {outfile}')

    def plot_comparison_summary(self, communicate_df, splithalf_df):
        """Communicate-vs-pointlight Dice with the split-half ceiling overlaid on every panel."""
        plot_df = communicate_df.copy()
        plot_df['percent_label'] = (plot_df['percent_threshold'] * 100).round().astype(int)

        ceiling_df = splithalf_df.copy()
        ceiling_df['percent_label'] = (ceiling_df['percent_threshold'] * 100).round().astype(int)

        hemispheres = [h for h in HEMISPHERE_ORDER if h in plot_df['hemisphere'].unique()]
        contrasts = [c for c in CONTRAST_ORDER if c in plot_df['communicate_contrast'].unique()]

        fig, axes = plt.subplots(len(hemispheres), len(contrasts),
                                  figsize=(5 * len(contrasts), 5 * len(hemispheres)),
                                  sharey=True, sharex=True, squeeze=False)

        for i, hemi in enumerate(hemispheres):
            ceiling_hemi_df = ceiling_df[ceiling_df['hemisphere'] == hemi].copy()
            ceiling_hemi_df['series'] = CEILING_LABEL

            for j, contrast in enumerate(contrasts):
                ax = axes[i][j]
                sub_df = plot_df[(plot_df['hemisphere'] == hemi)
                                  & (plot_df['communicate_contrast'] == contrast)].copy()
                sub_df['series'] = COMPARISON_LABEL
                panel_df = pd.concat([sub_df, ceiling_hemi_df], ignore_index=True)

                sns.pointplot(data=panel_df, x='percent_label', y='dice', hue='series',
                               palette=PALETTE, dodge=0.2, errorbar='se', ax=ax)
                ax.set_title(f'{hemi}: {contrast}' if i == 0 else contrast, fontsize=10)
                ax.set_xlabel('Top % of STS parcel (by t-value)' if i == len(hemispheres) - 1 else '')
                ax.set_ylabel(f'{hemi} hemisphere\nDice coefficient' if j == 0 else '')
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc='upper left', title=None)
                elif ax.get_legend() is not None:
                    ax.get_legend().remove()

        fig.suptitle('Voxel-level overlap within STS: communicate contrasts vs. pointlight interact-noninteract\n'
                      '(gray = pointlight odd/even split-half reliability, as a noise ceiling)')
        fig.tight_layout()
        outfile = os.path.join(self.communicate_pointlight_path, 'dice_group_summary.png')
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {outfile}')

    def run(self):
        communicate_df, splithalf_df = self.load_subject_tables()
        self.save_group_tables(communicate_df, splithalf_df)
        self.plot_splithalf_summary(splithalf_df)
        self.plot_comparison_summary(communicate_df, splithalf_df)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-subject communicate-vs-pointlight and pointlight split-half '
                     'Dice tables into group summaries, with split-half as a noise-ceiling reference')
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    args = parser.parse_args()
    VoxelOverlapCommunicatePointlightGroup(args).run()


if __name__ == '__main__':
    main()
