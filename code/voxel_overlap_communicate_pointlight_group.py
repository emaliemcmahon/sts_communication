import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HEMISPHERE_ORDER = ['left', 'right']
CONTRAST_ORDER = ['com_ind-ind', 'com_phy-phy', 'face_first-face_noncom', 'face_third-face_noncom']
CONTRAST_COLORS = {
    'com_ind-ind': '#1b9e77',
    'com_phy-phy': '#d95f02',
    'face_first-face_noncom': '#7570b3',
    'face_third-face_noncom': '#e7298a',
}
CEILING_COLOR = '0.5'
N_BOOT = 2000
CI = 95
BOOT_SEED = 0


def pivot_wide(df, value_col='dice', index_col='subject', columns_col='percent_threshold'):
    """subjects (rows) x condition (columns, e.g. percent_threshold), values = statistic."""
    return df.pivot(index=index_col, columns=columns_col, values=value_col).sort_index(axis=1)


def bootstrap_mean_ci(wide_df, n_boot=N_BOOT, ci=CI, seed=BOOT_SEED):
    """
    Percentile-method bootstrap CI, resampling participants (rows of wide_df)
    with replacement, jointly across all columns within each draw.

    Returns (observed_mean, ci_lower, ci_upper), each a pandas Series indexed
    like wide_df.columns.
    """
    rng = np.random.default_rng(seed)
    values = wide_df.to_numpy()
    n_subjects = values.shape[0]
    observed_mean = values.mean(axis=0)

    boot_idx = rng.integers(0, n_subjects, size=(n_boot, n_subjects))
    boot_means = values[boot_idx].mean(axis=1)  # (n_boot, n_conditions)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha, axis=0)
    upper = np.percentile(boot_means, 100 - alpha, axis=0)

    return (pd.Series(observed_mean, index=wide_df.columns),
            pd.Series(lower, index=wide_df.columns),
            pd.Series(upper, index=wide_df.columns))


class VoxelOverlapCommunicatePointlightGroup:
    """
    Aggregate per-subject Dice tables from voxel_overlap_communicate_pointlight.py
    (each communicate contrast vs. pointlight interact-noninteract) and from
    voxel_overlap_pointlight_splithalf.py (pointlight interact-noninteract,
    odd vs even runs) into group tables, plus a combined summary plot: one
    panel per hemisphere, all four communicate contrasts overlaid in
    different colors, with the split-half Dice shown as a shaded 95% CI band
    (a noise ceiling — how much the pointlight contrast overlaps with
    *itself* across independent halves of the data, which upper-bounds how
    much overlap any other, independent contrast could plausibly show).
    All uncertainty (the ceiling band and the per-contrast error bars alike)
    is a 95% CI from bootstrapping participants.
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
        """Standalone pointlight split-half reliability plot (the ceiling on its own),
        one panel per hemisphere, with a bootstrap 95% CI error bar at each percent level."""
        hemispheres = [h for h in HEMISPHERE_ORDER if h in splithalf_df['hemisphere'].unique()]

        fig, axes = plt.subplots(1, len(hemispheres), figsize=(6.5 * len(hemispheres), 6),
                                  sharey=True, squeeze=False)
        axes = axes[0]

        for ax, hemi in zip(axes, hemispheres):
            wide = pivot_wide(splithalf_df[splithalf_df['hemisphere'] == hemi])
            percents = wide.columns.to_numpy() * 100
            mean, lower, upper = bootstrap_mean_ci(wide)
            yerr = np.vstack([mean.to_numpy() - lower.to_numpy(), upper.to_numpy() - mean.to_numpy()])

            for _, row in wide.iterrows():
                ax.plot(percents, row.to_numpy(), color=CEILING_COLOR, alpha=0.15, linewidth=0.8, zorder=1)
            ax.errorbar(percents, mean.to_numpy(), yerr=yerr, marker='o', markersize=5, capsize=3,
                        color=CEILING_COLOR, zorder=2)
            ax.set_title(f'{hemi} hemisphere')
            ax.set_xlabel('Top % of STS parcel (by effect size)')

        axes[0].set_ylabel('Dice coefficient (odd vs even runs)')
        fig.suptitle('Pointlight interact-noninteract: split-half reliability within STS\n'
                      '(error bars = 95% CI, bootstrapped across participants)')
        fig.tight_layout()
        outfile = os.path.join(self.splithalf_path, 'dice_group_summary.png')
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {outfile}')

    def plot_comparison_summary(self, communicate_df, splithalf_df):
        """One panel per hemisphere: all communicate contrasts overlaid (bootstrap 95% CI
        error bars), with the pointlight split-half Dice shown as a shaded 95% CI band."""
        hemispheres = [h for h in HEMISPHERE_ORDER if h in communicate_df['hemisphere'].unique()]
        contrasts = [c for c in CONTRAST_ORDER if c in communicate_df['communicate_contrast'].unique()]

        fig, axes = plt.subplots(1, len(hemispheres), figsize=(7 * len(hemispheres), 6),
                                  sharey=True, squeeze=False)
        axes = axes[0]

        # small x-offsets so overlapping contrasts' error bars stay legible
        offsets = np.linspace(-1.5, 1.5, len(contrasts))

        for ax, hemi in zip(axes, hemispheres):
            # Noise ceiling: shaded 95% CI band (bootstrap over participants)
            ceiling_wide = pivot_wide(splithalf_df[splithalf_df['hemisphere'] == hemi])
            ceiling_percents = ceiling_wide.columns.to_numpy() * 100
            _, ceiling_lower, ceiling_upper = bootstrap_mean_ci(ceiling_wide)
            ax.fill_between(ceiling_percents, ceiling_lower.to_numpy(), ceiling_upper.to_numpy(),
                             color=CEILING_COLOR, alpha=0.25, label='pointlight split-half (ceiling)', zorder=1)

            # Communicate-vs-pointlight overlap: one line per contrast, bootstrap 95% CI error bars
            for contrast, offset in zip(contrasts, offsets):
                contrast_wide = pivot_wide(
                    communicate_df[(communicate_df['hemisphere'] == hemi)
                                   & (communicate_df['communicate_contrast'] == contrast)])
                c_percents = contrast_wide.columns.to_numpy() * 100
                c_mean, c_lower, c_upper = bootstrap_mean_ci(contrast_wide)
                yerr = np.vstack([c_mean.to_numpy() - c_lower.to_numpy(), c_upper.to_numpy() - c_mean.to_numpy()])
                ax.errorbar(c_percents + offset, c_mean.to_numpy(), yerr=yerr, marker='o', markersize=5,
                            capsize=3, linestyle='-', color=CONTRAST_COLORS[contrast], label=contrast, zorder=2)

            ax.set_title(f'{hemi} hemisphere')
            ax.set_xlabel('Top % of STS parcel (by t-value)')

        axes[0].set_ylabel('Dice coefficient')
        axes[0].legend(fontsize=9, loc='upper left')
        fig.suptitle('Voxel-level overlap within STS: communicate contrasts vs. pointlight interact-noninteract\n'
                      '(shaded band = pointlight split-half 95% CI, as a noise ceiling; '
                      'error bars = 95% CI, both bootstrapped across participants)')
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
