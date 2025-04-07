import os
import argparse
from glob import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np
from itertools import permutations
from statsmodels.stats.power import tt_ind_solve_power
from scipy.stats import ttest_1samp


class GroupRunwiseResults:
    def __init__(self, args):
        self.process = 'GroupRunwiseResults'
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.plot_indiv = args.plot_indiv
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.individual_path = f'{self.derivatives_path}/RunwiseResponse'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.out_file = f'{self.out_path}/summary.csv'
        self.subjs = ['CP03', 'CP04']
        self.subj_colors = ['black', 'dimgray']
        self.palette = [
                "#E57373",  # Soft muted red
                "#3949AB",  # Deep muted navy (Dark Blue 1)
                "#5C6BC0",  # Dusty periwinkle (Dark Blue 2)
                "#90CAF9",  # Pale sky blue (Light Blue)
                "#673AB7",  # Rich muted violet (Dark Purple 1)
                "#9575CD",  # Soft lavender (Light Purple 1)
                "#B39DDB",  # Pale lilac (Light Purple 2)
            ]
        
        self.conditions = ['object', 'body', 
                           'face_first', 'face_third', 'face_noncom',
                           'com_phy', 'phy', 'com_ind', 'ind']
        self.plotting_conditions = ['object',
                                    'face_first', 'face_third', 'face_noncom',
                                    'com', 'phy', 'ind']
        self.dyad_com_conditions = ['com_phy', 'com_ind']
        self.hemis = ['l', 'r']
        self.rois = ['EVC', 'MT', 'FFA', 'EBA', 'fSTS',
                     'SI-STS', 'face-comSTS', 'dyad-comSTS',
                     'TPJ']
        Path(self.out_path).mkdir(exist_ok=True, parents=True)

    def plot_rois(self, roi_response):
        sns.set_context('talk')
        fig, axes = plt.subplots(len(self.rois), 2,
                               sharex=True, sharey='row',
                               figsize=(10, 24))
        axes = axes.flatten()
        for ax, ((roi, hemi), df) in zip(axes, roi_response.groupby(['roi', 'hemi'], observed=True)):
            sns.barplot(x='trial_type', y='response',
                        hue='trial_type', legend=False,
                        ax=ax, data=df, palette=self.palette)
            
            if self.plot_indiv:
                sns.stripplot(x='trial_type', y='response', 
                              hue='subject_label', ax=ax,
                              legend=False, palette='gray',
                              data=df, size=10)

            ax.set_xticks(range(len(self.plotting_conditions)))
            ax.set_xticklabels(self.plotting_conditions, rotation=45, ha='right')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('')
            ax.set_ylabel(r'$\beta$')
            ax.set_title(f'{hemi} {roi}')
        fig.tight_layout()
        fig.savefig(f'{self.out_path}/summary.pdf')

    def load_data(self):
        files = glob(f'{self.individual_path}/*/roi_response.csv')
        files = [file for subj, file in product(self.subjs, files) if subj in file]
        df = []
        for file in tqdm(files, desc='Loading subject data'):
            sub = file.split('/')[-2].split('-')[-1]
            sub_df = pd.read_csv(file)
            sub_df['subject_label'] = sub
            df.append(sub_df)
        return pd.concat(df, ignore_index=True).reset_index(drop=True)
    
    def power_analysis(self, df):
        summary = []
        for (hemi, roi, subj), roi_df in df.groupby(['hemi', 'roi', 'subject_label'], observed=True):
            roi_df.set_index('trial_type', inplace=True)
            for (c1, c2) in permutations(self.plotting_conditions, 2):
                diff = roi_df.loc[c1, 'response'] - roi_df.loc[c2, 'response']
                summary.append({'hemi': hemi, 'subj': subj, 'roi': roi, 'c1': c1, 'c2': c2, 'diff': diff})
        summary = pd.DataFrame(summary)
        summary = summary.set_index(['hemi', 'roi', 'c1', 'c2']).sort_index()
        
        for roi in ['SI-STS', 'fSTS', 'dyad-comSTS', 'face-comSTS']:
            for c1, c2 in [('face_first', 'face_noncom'),
                           ('face_third', 'face_noncom'),
                           ('com', 'ind'),
                           ('com', 'phy')]:
                try: 
                    vals = summary.loc[('r', roi, c1, c2), 'diff'].to_numpy()
                    effect_size = np.mean(vals) / np.std(vals)
                    stats = ttest_1samp(vals, popmean=0, alternative='greater')
                    print(f'{roi} {c1} {c2}')
                    print(f'effect_size = {effect_size}')
                    print(f't({int(stats.df)}) = {stats.statistic:.3f}, p = {stats.pvalue:.3f}')
                    if stats.pvalue > 0.05: 
                        n_obs = tt_ind_solve_power(effect_size=effect_size, alpha=0.05,
                                                power=0.95, alternative='larger')
                        print(f'n_obs = {n_obs:.1f}')
                    else:
                        print(f'already significant')
                except:
                    print('failed to converge')
        return n_obs

    def run(self):
        if self.overwrite or not os.path.exists(self.out_file):
            df = self.load_data()
            df.to_csv(self.out_file, index=False)
        else:
            df = pd.read_csv(self.out_file)
        avg_df = df.groupby(['roi', 'hemi',
                             'trial_type', 'subject_label']).mean(numeric_only=True).reset_index()
        com_df = avg_df.loc[avg_df['trial_type'].isin(self.dyad_com_conditions)]
        com_df = com_df.groupby(['roi', 'hemi', 'subject_label']).mean(numeric_only=True).reset_index()
        com_df['trial_type'] = 'com'

        avg_df = avg_df.loc[~avg_df['trial_type'].isin(self.dyad_com_conditions)].reset_index(drop=True)
        mean_df = pd.concat([avg_df, com_df[avg_df.columns]], ignore_index=True).reset_index(drop=True)

        mean_df = mean_df.loc[mean_df['trial_type'].isin(self.plotting_conditions)].reset_index(drop=True)

        mean_df['trial_type'] = pd.Categorical(mean_df['trial_type'],
                                            ordered=True,
                                            categories=self.plotting_conditions)
        mean_df['roi'] = pd.Categorical(mean_df['roi'],
                                            ordered=True,
                                            categories=self.rois)
        mean_df['hemi'] = pd.Categorical(mean_df['hemi'],
                                            ordered=True,
                                            categories=self.hemis)
        mean_df['subject_label'] = pd.Categorical(mean_df['subject_label'],
                                            ordered=True,
                                            categories=self.subjs)
        self.plot_rois(mean_df)
        self.power_analysis(mean_df)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/communicate_pilot')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--plot_indiv', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    GroupRunwiseResults(args).run()


if __name__ == '__main__':
    main()
