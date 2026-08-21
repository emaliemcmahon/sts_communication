import argparse
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from utils.mvpa import ROI_DEFINING_CONTRAST
from utils.stats import bootstrap_mean_ci, p2star


class GroupDecodingIndex:
    """
    Group-level significance testing for the per-subject Haxby-style
    within/between decoding index (com_ind, ind, face_first, face_noncom;
    see decoding_index.py). For each (hemisphere, ROI), tests the subject-wise
    decoding index against 0 with a sign-flip permutation test and reports a
    subject-bootstrap confidence interval on the mean.
    """

    def __init__(self, args):
        self.process = 'GroupDecodingIndex'
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.n_permutations = args.n_permutations
        self.n_bootstrap = args.n_bootstrap
        self.conf_level = args.conf_level
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.individual_path = os.path.join(self.derivatives_path, 'DecodingIndex')
        out_dir = f'{self.process}_{args.out_tag}' if args.out_tag else self.process
        self.out_path = os.path.join(self.derivatives_path, out_dir)
        self.out_file = os.path.join(self.out_path, 'subject_decoding_index.csv')
        self.stats_file = os.path.join(self.out_path, 'stats.csv')
        self.corr_file = os.path.join(self.out_path, f'{args.corr_roi_x}_{args.corr_roi_y}_correlation.csv')
        self.sub_nums = args.sub_nums
        self.subjs = [str(i).zfill(2) for i in self.sub_nums]
        self.hemis = ['l', 'r']
        self.rois = list(ROI_DEFINING_CONTRAST.keys())
        self.corr_roi_x = args.corr_roi_x
        self.corr_roi_y = args.corr_roi_y
        print(vars(self))
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        rows = []
        for sub in self.subjs:
            f = os.path.join(self.individual_path, f'sub-{sub}', f'sub-{sub}_decoding_index.csv')
            if not os.path.exists(f):
                print(f'Warning: missing {f}')
                continue
            rows.append(pd.read_csv(f))
        if not rows:
            raise FileNotFoundError('No subject decoding-index CSVs found.')
        long = pd.concat(rows, ignore_index=True)
        long.to_csv(self.out_file, index=False)
        return long

    def compute_stats(self, long):
        rows = []
        for hemi, roi in product(self.hemis, self.rois):
            sub = long.loc[(long['hemi'] == hemi) & (long['roi'] == roi)]
            vals = sub['decoding_index'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                rows.append({'hemi': hemi, 'roi': roi, 'n': int(vals.size),
                             'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                             'p': np.nan, 'star': None})
                continue

            observed, ci_low, ci_high, p_val, _ = bootstrap_mean_ci(
                vals, n_bootstrap=self.n_bootstrap, n_permutations=self.n_permutations,
                conf_level=self.conf_level, alternative='greater')
            rows.append({'hemi': hemi, 'roi': roi, 'n': int(vals.size),
                         'mean': observed, 'ci_low': ci_low, 'ci_high': ci_high,
                         'p': p_val, 'star': p2star(p_val)})

        stats = pd.DataFrame(rows)
        stats.to_csv(self.stats_file, index=False)
        return stats

    def plot_summary(self, stats):
        sns.set_context('talk', font_scale=0.8)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        hemi_colors = {'l': '#4C72B0', 'r': '#DD8452'}
        bar_width = 0.8 / len(self.hemis)

        max_top = 0.0
        for roi_idx, roi in enumerate(self.rois):
            for hemi_idx, hemi in enumerate(self.hemis):
                row = stats.loc[(stats['roi'] == roi) & (stats['hemi'] == hemi)]
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
                max_top = max(max_top, hi)
                if row['star']:
                    ax.text(bar_x, hi + 0.005, row['star'], ha='center', fontsize=14)

        ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
        ax.set_xticks(np.arange(len(self.rois)))
        ax.set_xticklabels(self.rois, rotation=45, ha='right')
        ax.set_ylabel('Decoding index')
        ax.set_title('Group decoding index: 0.5*(r_w1+r_w2-r_b1-r_b2)')
        sns.despine(ax=ax)
        ax.legend(title='hemi', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_path, 'decoding_index_summary.pdf'))
        plt.close(fig)

    # (hemi_x, hemi_y) pairings: same-hemisphere first, then both
    # cross-hemisphere directions (left corr_roi_x vs. right corr_roi_y and
    # vice versa), since a confound shared across homotopic regions need not
    # respect hemisphere boundaries.
    HEMI_PAIRS = [('l', 'l'), ('r', 'r'), ('l', 'r'), ('r', 'l')]

    def compute_roi_correlation(self, long):
        """Across-subject Pearson correlation between the decoding index in
        ``self.corr_roi_x`` and ``self.corr_roi_y``, computed for each
        same- and cross-hemisphere pairing (HEMI_PAIRS). A weak/non-significant
        correlation supports the argument that a ``corr_roi_y`` effect (e.g.
        SI-STS) isn't inherited from a low-level-confound-driven effect in
        ``corr_roi_x`` (e.g. EVC).
        """
        wide = long.pivot_table(index='subject', columns=['roi', 'hemi'], values='decoding_index')
        wide.columns = [f'{roi}_{hemi}' for roi, hemi in wide.columns]
        wide = wide.reset_index()

        rows = []
        for hemi_x, hemi_y in self.HEMI_PAIRS:
            x_col, y_col = f'{self.corr_roi_x}_{hemi_x}', f'{self.corr_roi_y}_{hemi_y}'
            if x_col not in wide.columns or y_col not in wide.columns:
                rows.append({'hemi_x': hemi_x, 'hemi_y': hemi_y, 'n': 0, 'r': np.nan, 'p': np.nan})
                continue
            sub = wide[[x_col, y_col]].dropna()
            if len(sub) < 3:
                rows.append({'hemi_x': hemi_x, 'hemi_y': hemi_y, 'n': len(sub), 'r': np.nan, 'p': np.nan})
                continue
            r, p = pearsonr(sub[x_col], sub[y_col])
            rows.append({'hemi_x': hemi_x, 'hemi_y': hemi_y, 'n': len(sub), 'r': r, 'p': p})
        corr = pd.DataFrame(rows)
        corr.to_csv(self.corr_file, index=False)
        return wide, corr

    def plot_roi_correlation(self, wide, corr):
        sns.set_context('talk', font_scale=0.8)
        pair_colors = {('l', 'l'): '#4C72B0', ('r', 'r'): '#DD8452',
                       ('l', 'r'): '#55A868', ('r', 'l'): '#8172B2'}
        hemi_names = {'l': 'left', 'r': 'right'}
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)

        for ax, (hemi_x, hemi_y) in zip(axes.flat, self.HEMI_PAIRS):
            x_col, y_col = f'{self.corr_roi_x}_{hemi_x}', f'{self.corr_roi_y}_{hemi_y}'
            row = corr.loc[(corr['hemi_x'] == hemi_x) & (corr['hemi_y'] == hemi_y)].iloc[0]
            if x_col not in wide.columns or y_col not in wide.columns or pd.isna(row['r']):
                ax.set_title(f'{hemi_names[hemi_x]} {self.corr_roi_x} vs. {hemi_names[hemi_y]} '
                             f'{self.corr_roi_y}\n(insufficient data)')
                continue
            sub = wide[[x_col, y_col]].dropna()
            sns.regplot(data=sub, x=x_col, y=y_col, ax=ax,
                        color=pair_colors[(hemi_x, hemi_y)], scatter_kws={'edgecolor': 'black', 's': 40})
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
            star = p2star(row['p']) or 'n.s.'
            ax.set_title(f"{hemi_names[hemi_x]} {self.corr_roi_x} vs. {hemi_names[hemi_y]} {self.corr_roi_y} "
                         f"(n={int(row['n'])})\nr={row['r']:.2f}, p={row['p']:.3f} {star}")
            ax.set_xlabel(f'{hemi_names[hemi_x]} {self.corr_roi_x} decoding index')
            ax.set_ylabel(f'{hemi_names[hemi_y]} {self.corr_roi_y} decoding index')

        fig.suptitle(f'Across-subject correlation: {self.corr_roi_x} vs. {self.corr_roi_y} decoding index '
                     '(same- and cross-hemisphere)')
        sns.despine(fig=fig)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_path, f'{self.corr_roi_x}_{self.corr_roi_y}_correlation.pdf'))
        plt.close(fig)

    def run(self):
        if self.overwrite or not os.path.exists(self.out_file):
            long = self.load_data()
        else:
            long = pd.read_csv(self.out_file)

        stats = self.compute_stats(long)
        self.plot_summary(stats)

        wide, corr = self.compute_roi_correlation(long)
        self.plot_roi_correlation(wide, corr)


def main():
    parser = argparse.ArgumentParser(
        description='Group-level significance testing for the per-subject decoding index.')
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
    parser.add_argument('--corr_roi_x', type=str, default='EVC',
                         help='ROI plotted on the x-axis of the across-subject '
                              'decoding-index correlation scatter.')
    parser.add_argument('--corr_roi_y', type=str, default='SI-STS',
                         help='ROI plotted on the y-axis of the across-subject '
                              'decoding-index correlation scatter.')
    parser.add_argument('--out_tag', type=str, default='',
                         help='If set, write to derivatives/GroupDecodingIndex_<out_tag>/ instead '
                              'of derivatives/GroupDecodingIndex/, so a run on a subject subset '
                              'does not overwrite the full-sample results.')
    args = parser.parse_args()
    GroupDecodingIndex(args).run()


if __name__ == '__main__':
    main()
