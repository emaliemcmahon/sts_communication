import argparse
import os
from glob import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PAIR_TYPE_ORDER = ['dyad-face', 'dyad-dyad', 'face-face']
HEMISPHERE_ORDER = ['left', 'right']


class VoxelOverlapGroup:
    """
    Aggregate per-subject Dice tables produced by voxel_overlap.py
    (derivatives/VoxelOverlap/sub-*/sub-*_dice_coefficients.csv) into a single
    group table and summary plot, faceted by hemisphere (rows) and pair type
    (columns: cross-domain dyad-vs-face pairs vs. the two within-domain
    baseline pairs) — the top-percent selection and Dice coefficient are
    computed independently per hemisphere, so hemispheres are never pooled.
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
        hemispheres = [h for h in HEMISPHERE_ORDER if h in plot_df['hemisphere'].unique()]

        fig, axes = plt.subplots(len(hemispheres), len(pair_types),
                                  figsize=(6.5 * len(pair_types), 5.5 * len(hemispheres)),
                                  sharey=True, sharex=True, squeeze=False)

        for i, hemi in enumerate(hemispheres):
            for j, pt in enumerate(pair_types):
                ax = axes[i, j]
                sub_df = plot_df[(plot_df['hemisphere'] == hemi) & (plot_df['pair_type'] == pt)]
                # dodge must be False (or plain True) when there's only one hue level,
                # otherwise seaborn divides by (n_hue_levels - 1) == 0
                n_hue = sub_df['contrast_pair'].nunique()
                point_dodge = 0.3 if n_hue > 1 else False
                strip_dodge = n_hue > 1
                sns.pointplot(data=sub_df, x='percent_label', y='dice', hue='contrast_pair',
                               dodge=point_dodge, errorbar='se', ax=ax)
                sns.stripplot(data=sub_df, x='percent_label', y='dice', hue='contrast_pair',
                               dodge=strip_dodge, alpha=0.2, ax=ax, legend=False)
                ax.set_title(f'{hemi}: {pt}' if i == 0 else pt)
                ax.set_xlabel('Top % of STS parcel (by t-value)' if i == len(hemispheres) - 1 else '')
                ax.set_ylabel(f'{hemi} hemisphere\nDice coefficient' if j == 0 else '')
                ax.legend(title=None, fontsize=8, loc='upper left')

        fig.suptitle('Voxel-level overlap within STS (computed separately per hemisphere): '
                      'dyad vs. face-perception contrasts')
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
