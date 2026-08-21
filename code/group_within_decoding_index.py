import argparse
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.mvpa import ROI_DEFINING_CONTRAST
from utils.stats import bootstrap_mean_ci, p2star
from within_decoding_index import PAIRS


class GroupWithinDecodingIndex:
    """
    Group-level significance testing for the per-subject within-condition-pair
    decoding index (com_ind vs. ind; face_first vs. face_noncom; see
    within_decoding_index.py). For each (hemisphere, ROI, pair), tests the
    subject-wise within decoding index against 0 with a sign-flip permutation
    test and reports a subject-bootstrap confidence interval on the mean --
    mirroring group_decoding_index.py's test, but per pair.

    Also plots the within-pair indices alongside the existing cross-pair
    decoding index (decoding_index.py / group_decoding_index.py, loaded from
    `--cross_stats_file` if present) so the two can be compared directly.
    """

    def __init__(self, args):
        self.process = 'GroupWithinDecodingIndex'
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.n_permutations = args.n_permutations
        self.n_bootstrap = args.n_bootstrap
        self.conf_level = args.conf_level
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.individual_path = os.path.join(self.derivatives_path, 'WithinDecodingIndex')
        out_dir = f'{self.process}_{args.out_tag}' if args.out_tag else self.process
        self.out_path = os.path.join(self.derivatives_path, out_dir)
        self.out_file = os.path.join(self.out_path, 'subject_within_decoding_index.csv')
        self.stats_file = os.path.join(self.out_path, 'stats.csv')
        self.cross_stats_file = args.cross_stats_file or os.path.join(
            self.derivatives_path, 'GroupDecodingIndex', 'stats.csv')
        self.sub_nums = args.sub_nums
        self.subjs = [str(i).zfill(2) for i in self.sub_nums]
        self.hemis = ['l', 'r']
        self.rois = list(ROI_DEFINING_CONTRAST.keys())
        self.pairs = [p[0] for p in PAIRS]
        print(vars(self))
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        rows = []
        for sub in self.subjs:
            f = os.path.join(self.individual_path, f'sub-{sub}', f'sub-{sub}_within_decoding_index.csv')
            if not os.path.exists(f):
                print(f'Warning: missing {f}')
                continue
            rows.append(pd.read_csv(f))
        if not rows:
            raise FileNotFoundError('No subject within-decoding-index CSVs found.')
        long = pd.concat(rows, ignore_index=True)
        long.to_csv(self.out_file, index=False)
        return long

    def compute_stats(self, long):
        rows = []
        for hemi, roi, pair in product(self.hemis, self.rois, self.pairs):
            sub = long.loc[(long['hemi'] == hemi) & (long['roi'] == roi) & (long['pair'] == pair)]
            vals = sub['within_decoding_index'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                rows.append({'hemi': hemi, 'roi': roi, 'pair': pair, 'n': int(vals.size),
                             'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                             'p': np.nan, 'star': None})
                continue

            observed, ci_low, ci_high, p_val, _ = bootstrap_mean_ci(
                vals, n_bootstrap=self.n_bootstrap, n_permutations=self.n_permutations,
                conf_level=self.conf_level, alternative='greater')
            rows.append({'hemi': hemi, 'roi': roi, 'pair': pair, 'n': int(vals.size),
                         'mean': observed, 'ci_low': ci_low, 'ci_high': ci_high,
                         'p': p_val, 'star': p2star(p_val)})

        stats = pd.DataFrame(rows)
        stats.to_csv(self.stats_file, index=False)
        return stats

    def plot_summary(self, stats):
        sns.set_context('talk', font_scale=0.8)
        fig, axes = plt.subplots(1, len(self.pairs), figsize=(12 * len(self.pairs), 5), sharey=True)
        hemi_colors = {'l': '#4C72B0', 'r': '#DD8452'}
        bar_width = 0.8 / len(self.hemis)

        for ax, pair in zip(np.atleast_1d(axes), self.pairs):
            for roi_idx, roi in enumerate(self.rois):
                for hemi_idx, hemi in enumerate(self.hemis):
                    row = stats.loc[(stats['roi'] == roi) & (stats['hemi'] == hemi) & (stats['pair'] == pair)]
                    if row.empty or not np.isfinite(row.iloc[0]['mean']):
                        continue
                    row = row.iloc[0]
                    bar_x = roi_idx + (hemi_idx - (len(self.hemis) - 1) / 2) * bar_width
                    lo = row['ci_low'] if np.isfinite(row['ci_low']) else row['mean']
                    hi = row['ci_high'] if np.isfinite(row['ci_high']) else row['mean']
                    ax.bar(bar_x, row['mean'], width=bar_width, color=hemi_colors[hemi],
                           edgecolor='black', linewidth=0.8,
                           yerr=[[max(row['mean'] - lo, 0)], [max(hi - row['mean'], 0)]],
                           capsize=3, label=hemi if roi_idx == 0 else None)
                    if pd.notna(row['star']) and row['star']:
                        ax.text(bar_x, hi + 0.005, row['star'], ha='center', fontsize=14)

            ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
            ax.set_xticks(np.arange(len(self.rois)))
            ax.set_xticklabels(self.rois, rotation=45, ha='right')
            ax.set_title(f'pair = {pair}')
            sns.despine(ax=ax)

        axes[0].set_ylabel('Within decoding index')
        fig.suptitle('Group within decoding index: 0.5*(r_w1+r_w2) - r_b, per condition pair')
        axes[-1].legend(title='hemi', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_path, 'within_decoding_index_summary.pdf'))
        plt.close(fig)

    def plot_comparison(self, stats):
        """Cross-pair (decoding_index.py) vs. within-pair (this script)
        decoding index, side by side per ROI, one panel per hemisphere."""
        if not os.path.exists(self.cross_stats_file):
            print(f'Cross-pair stats file not found ({self.cross_stats_file}); '
                 'skipping comparison plot. Run group_decoding_index.py first.')
            return
        cross = pd.read_csv(self.cross_stats_file)
        cross = cross.assign(index_type='cross')[['hemi', 'roi', 'mean', 'ci_low', 'ci_high', 'star', 'index_type']]

        within = stats.copy()
        within['index_type'] = 'within-' + within['pair']
        within = within[['hemi', 'roi', 'mean', 'ci_low', 'ci_high', 'star', 'index_type']]

        combined = pd.concat([cross, within], ignore_index=True)
        index_types = ['cross'] + [f'within-{p}' for p in self.pairs]
        type_colors = {'cross': '#55A868', 'within-dyad': '#4C72B0', 'within-face': '#DD8452'}

        sns.set_context('talk', font_scale=0.8)
        fig, axes = plt.subplots(1, len(self.hemis), figsize=(12 * len(self.hemis), 5), sharey=True)
        bar_width = 0.8 / len(index_types)

        for ax, hemi in zip(np.atleast_1d(axes), self.hemis):
            for roi_idx, roi in enumerate(self.rois):
                for type_idx, index_type in enumerate(index_types):
                    row = combined.loc[(combined['hemi'] == hemi) & (combined['roi'] == roi)
                                       & (combined['index_type'] == index_type)]
                    if row.empty or not np.isfinite(row.iloc[0]['mean']):
                        continue
                    row = row.iloc[0]
                    bar_x = roi_idx + (type_idx - (len(index_types) - 1) / 2) * bar_width
                    lo = row['ci_low'] if np.isfinite(row['ci_low']) else row['mean']
                    hi = row['ci_high'] if np.isfinite(row['ci_high']) else row['mean']
                    ax.bar(bar_x, row['mean'], width=bar_width, color=type_colors[index_type],
                           edgecolor='black', linewidth=0.8,
                           yerr=[[max(row['mean'] - lo, 0)], [max(hi - row['mean'], 0)]],
                           capsize=3, label=index_type if roi_idx == 0 else None)
                    if pd.notna(row['star']) and row['star']:
                        ax.text(bar_x, hi + 0.005, row['star'], ha='center', fontsize=12)

            ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
            ax.set_xticks(np.arange(len(self.rois)))
            ax.set_xticklabels(self.rois, rotation=45, ha='right')
            ax.set_title(f'hemi = {hemi}')
            sns.despine(ax=ax)

        axes[0].set_ylabel('Decoding index')
        fig.suptitle('Cross-pair vs. within-pair decoding index')
        axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_path, 'within_vs_cross_decoding_index.pdf'))
        plt.close(fig)

    def run(self):
        if self.overwrite or not os.path.exists(self.out_file):
            long = self.load_data()
        else:
            long = pd.read_csv(self.out_file)

        stats = self.compute_stats(long)
        self.plot_summary(stats)
        self.plot_comparison(stats)


def main():
    parser = argparse.ArgumentParser(
        description='Group-level significance testing for the per-subject within-condition-pair '
                    'decoding index, plus a comparison against the cross-pair decoding index.')
    parser.add_argument('sub_nums', nargs='*', type=int, help='List of subject numbers',
                        default=[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23])
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--n_permutations', type=int, default=10000,
                        help='Monte Carlo sign patterns for the group sign-flip test '
                             '(exact enumeration is used when n <= 15).')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Subject bootstrap resamples for the CI.')
    parser.add_argument('--conf_level', type=float, default=0.95)
    parser.add_argument('--cross_stats_file', type=str, default='',
                        help='Path to the cross-pair decoding index stats.csv '
                             '(group_decoding_index.py output) to plot alongside the '
                             'within-pair indices. Defaults to '
                             'derivatives/GroupDecodingIndex/stats.csv; the comparison '
                             'plot is skipped if not found.')
    parser.add_argument('--out_tag', type=str, default='',
                        help='If set, write to derivatives/GroupWithinDecodingIndex_<out_tag>/ '
                             'instead of derivatives/GroupWithinDecodingIndex/, so a run on a '
                             'subject subset does not overwrite the full-sample results.')
    args = parser.parse_args()
    GroupWithinDecodingIndex(args).run()


if __name__ == '__main__':
    main()
