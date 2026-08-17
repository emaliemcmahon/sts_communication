import argparse
import os
from glob import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PAIR_TYPE_ORDER = ['dyad-face', 'dyad-dyad', 'face-face']


class VoxelOverlapGroup:
    """
    Aggregate per-subject Dice tables produced by voxel_overlap.py
    (derivatives/VoxelOverlap/sub-*/sub-*_dice_coefficients.csv) into a single
    group table and summary plot, faceted by pair type (cross-domain
    dyad-vs-face pairs vs. the two within-domain baseline pairs).
    """

    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.out_path = os.path.join(self.dataset_path, 'derivatives', 'VoxelOverlap')

    def load_subject_tables(self):
        files = sorted(glob(os.path.join(self.out_path, 'sub-*', 'sub-*_dice_coefficients.csv')))
        if not files:
            raise RuntimeError(
                f'No per-subject Dice CSVs found under {self.out_path}. '
                'Run voxel_overlap.py for each subject first (see `make voxel_overlap`).'
            )
        print(f'Found {len(files)} per-subject Dice tables')
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    def save_group_table(self, df):
        outfile = os.path.join(self.out_path, 'group_dice_coefficients.csv')
        df.to_csv(outfile, index=False)
        print(f'Saved {outfile}')

    def plot_group_summary(self, df):
        plot_df = df.copy()
        plot_df['contrast_pair'] = plot_df['contrast_a'] + '  vs  ' + plot_df['contrast_b']
        plot_df['percent_label'] = (plot_df['percent_threshold'] * 100).round().astype(int)

        pair_types = [p for p in PAIR_TYPE_ORDER if p in plot_df['pair_type'].unique()]
        fig, axes = plt.subplots(1, len(pair_types), figsize=(6.5 * len(pair_types), 6), sharey=True)
        if len(pair_types) == 1:
            axes = [axes]

        for ax, pt in zip(axes, pair_types):
            sub_df = plot_df[plot_df['pair_type'] == pt]
            # dodge must be False (or plain True) when there's only one hue level,
            # otherwise seaborn divides by (n_hue_levels - 1) == 0
            n_hue = sub_df['contrast_pair'].nunique()
            point_dodge = 0.3 if n_hue > 1 else False
            strip_dodge = n_hue > 1
            sns.pointplot(data=sub_df, x='percent_label', y='dice', hue='contrast_pair',
                           dodge=point_dodge, errorbar='se', ax=ax)
            sns.stripplot(data=sub_df, x='percent_label', y='dice', hue='contrast_pair',
                           dodge=strip_dodge, alpha=0.2, ax=ax, legend=False)
            ax.set_title(pt)
            ax.set_xlabel('Top % of STS parcel (by t-value)')
            ax.legend(title=None, fontsize=8, loc='upper left')

        axes[0].set_ylabel('Dice coefficient')
        fig.suptitle('Voxel-level overlap within STS: dyad vs. face-perception contrasts')
        fig.tight_layout()
        outfile = os.path.join(self.out_path, 'dice_group_summary.png')
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {outfile}')

    def run(self):
        df = self.load_subject_tables()
        self.save_group_table(df)
        self.plot_group_summary(df)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-subject voxel-overlap Dice tables into a group summary')
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    args = parser.parse_args()
    VoxelOverlapGroup(args).run()


if __name__ == '__main__':
    main()
