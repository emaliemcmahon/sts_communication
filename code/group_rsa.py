import argparse
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.spatial.distance import squareform
from tqdm import tqdm

from rsa import CONDITIONS, MODEL_RDMS, MODEL_RDMS_CONDENSED
from utils.mvpa import ROI_DEFINING_CONTRAST
from utils.stats import loso_noise_ceiling, mantel_permutation_pvalue, p2star

MODEL_COLORS = {'3p_interaction': '#4C72B0', 'communication': '#DD8452'}


class GroupRSA:
    """
    Group-level RSA: averages per-subject condensed neural RDMs (saved by
    rsa.py) per ROI/hemisphere, correlates the group-mean neural RDM against
    each model RDM (3P interaction, communication) with a Mantel permutation
    test (shuffles condition labels, preserving the RDM's dependence
    structure), and reports a leave-one-subject-out (LOSO) noise ceiling for
    the neural RDM's reliability across subjects.

    No confidence interval is reported on the group correlation itself: it
    is a single scalar functional of the group-mean RDM, not a mean of
    subject-level values, so there is no natural sample whose spread it
    summarizes. The LOSO noise ceiling gives a sense of subject-level
    variability instead.
    """

    def __init__(self, args):
        self.process = 'GroupRSA'
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.individual_path = os.path.join(self.derivatives_path, 'RSA')
        self.out_path = os.path.join(self.derivatives_path, self.process)
        self.out_file = os.path.join(self.out_path, 'summary.nc')
        self.stats_file = os.path.join(self.out_path, 'stats.csv')
        self.sub_nums = args.sub_nums
        self.subjs = [str(i).zfill(2) for i in self.sub_nums]
        self.n_permutations = args.n_permutations
        self.hemis = ['l', 'r']
        self.rois = list(ROI_DEFINING_CONTRAST.keys())
        self.models = list(MODEL_RDMS_CONDENSED.keys())
        self.n_pairs = len(MODEL_RDMS_CONDENSED[self.models[0]])
        print(vars(self))
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        data = xr.DataArray(
            np.full((len(self.hemis), len(self.rois), len(self.subjs), self.n_pairs), np.nan),
            dims=['hemi', 'roi', 'subject', 'pattern'],
            coords={'hemi': self.hemis, 'roi': self.rois,
                    'subject': self.subjs, 'pattern': np.arange(self.n_pairs)},
        )
        for hemi, roi, sub in tqdm(list(product(self.hemis, self.rois, self.subjs)),
                                    desc='Loading subject data'):
            file = os.path.join(self.individual_path, f'sub-{sub}',
                                 f'sub-{sub}_hemi-{hemi}_roi-{roi}_neural_rdm.npy')
            if not os.path.exists(file):
                continue
            arr = np.load(file)
            if arr.shape[0] != self.n_pairs:
                print(f'Warning: unexpected pattern length for sub-{sub} {hemi} {roi}: '
                      f'{arr.shape[0]} (expected {self.n_pairs})')
                continue
            data.loc[hemi, roi, sub, :] = arr
        return data

    def compute_summary(self, data):
        """Group-mean RSA (Mantel p-value) and LOSO noise ceiling per (hemi, roi, model)."""
        rows = []
        for hemi, roi in tqdm(list(product(self.hemis, self.rois)), desc='Mantel + LOSO'):
            sub_rdms = data.loc[hemi, roi].values  # (n_subj, n_pairs)
            valid = ~np.isnan(sub_rdms).all(axis=1)
            sub_rdms_v = sub_rdms[valid]
            n = int(valid.sum())

            if sub_rdms_v.shape[0] < 2:
                for model in self.models:
                    rows.append({'hemi': hemi, 'roi': roi, 'model': model, 'n': n,
                                 'r': np.nan, 'p': np.nan, 'star': None,
                                 'noise_ceiling': np.nan})
                continue

            noise_ceiling, _ = loso_noise_ceiling(sub_rdms_v)
            mean_rdm = np.nanmean(sub_rdms_v, axis=0)
            for model in self.models:
                r, p_val = mantel_permutation_pvalue(
                    mean_rdm, MODEL_RDMS_CONDENSED[model],
                    n_permutations=self.n_permutations, alternative='greater')
                rows.append({'hemi': hemi, 'roi': roi, 'model': model, 'n': n,
                             'r': r, 'p': p_val, 'star': p2star(p_val),
                             'noise_ceiling': noise_ceiling})

        summary = pd.DataFrame(rows)
        summary.to_csv(self.stats_file, index=False)
        return summary

    def plot_rdms(self, data, summary):
        for hemi, roi in tqdm(list(product(self.hemis, self.rois)), desc='Plotting RDMs'):
            mean_rdm = np.nanmean(data.loc[hemi, roi].values, axis=0)
            if np.isnan(mean_rdm).all():
                continue
            mean_rdm_sq = squareform(mean_rdm)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
            panels = [('3p_interaction', MODEL_RDMS['3p_interaction']),
                      ('neural', mean_rdm_sq),
                      ('communication', MODEL_RDMS['communication'])]
            for ax, (name, mat) in zip(axes, panels):
                mask = np.triu(np.ones_like(mat, dtype=bool))
                sns.heatmap(mat, ax=ax, cmap='viridis', cbar=True, mask=mask,
                            xticklabels=CONDITIONS, yticklabels=CONDITIONS, square=True)
                ax.set_title(name)
                ax.tick_params(axis='x', rotation=45)

            for ax, model in zip((axes[0], axes[2]), ('3p_interaction', 'communication')):
                row = summary.loc[(summary['hemi'] == hemi) & (summary['roi'] == roi)
                                   & (summary['model'] == model)]
                if row.empty:
                    continue
                r_val, p_val = row.iloc[0]['r'], row.iloc[0]['p']
                star = p2star(p_val) or ''
                ax.set_xlabel(f'rho={r_val:.2f} {star}')

            fig.suptitle(f'{hemi} {roi}')
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_path, f'rdm_hemi-{hemi}_roi-{roi}.pdf'))
            plt.close(fig)

    def plot_rsa_summary(self, summary):
        sns.set_context('talk', font_scale=0.7)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        bar_width = 0.8 / len(self.models)

        for ax, hemi in zip(axes, self.hemis):
            s = summary.loc[summary['hemi'] == hemi]
            max_top = 0.0
            for roi_idx, roi in enumerate(self.rois):
                for model_idx, model in enumerate(self.models):
                    row = s.loc[(s['roi'] == roi) & (s['model'] == model)]
                    if row.empty or not np.isfinite(row.iloc[0]['r']):
                        continue
                    row = row.iloc[0]
                    bar_x = roi_idx + (model_idx - (len(self.models) - 1) / 2) * bar_width
                    ax.bar(bar_x, row['r'], width=bar_width, color=MODEL_COLORS[model],
                           edgecolor='black', linewidth=0.8, label=model if roi_idx == 0 else None)
                    max_top = max(max_top, row['r'])
                    if np.isfinite(row['noise_ceiling']):
                        ax.plot(bar_x, row['noise_ceiling'], marker='D', color='black',
                                markersize=5, linestyle='None', zorder=3)
                        max_top = max(max_top, row['noise_ceiling'])
                    if row['star']:
                        ax.text(bar_x, max(row['r'], row['noise_ceiling'] if np.isfinite(row['noise_ceiling']) else row['r']) + 0.02,
                                row['star'], ha='center', fontsize=14)

            ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
            ax.set_xticks(np.arange(len(self.rois)))
            ax.set_xticklabels(self.rois, rotation=45, ha='right')
            ax.set_title(f'{hemi} hemisphere')
            sns.despine(ax=ax)

        axes[0].set_ylabel('Spearman rho (group-mean RDM vs. model)')
        axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_path, 'rsa_summary.pdf'))
        plt.close(fig)

    def run(self):
        if self.overwrite or not os.path.exists(self.out_file):
            data = self.load_data()
            data.to_netcdf(self.out_file)
        else:
            data = xr.open_dataarray(self.out_file)

        summary = self.compute_summary(data)
        self.plot_rdms(data, summary)
        self.plot_rsa_summary(summary)


def main():
    parser = argparse.ArgumentParser(
        description='Group-level RSA: Mantel-permutation test of group-mean neural RDMs '
                     'against the 3P-interaction and communication model RDMs, per ROI.')
    parser.add_argument('sub_nums', nargs='*', type=int, help='List of subject numbers',
                         default=[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23])
    parser.add_argument('--dataset_path', '-d', type=str,
                         default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--n_permutations', type=int, default=5000,
                         help='Number of Mantel permutations for the p-value.')
    args = parser.parse_args()
    GroupRSA(args).run()


if __name__ == '__main__':
    main()
