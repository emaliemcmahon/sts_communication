import os
import argparse
from glob import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from itertools import product
import numpy as np
from matplotlib.lines import Line2D
from itertools import permutations
from statsmodels.stats.power import tt_ind_solve_power
from scipy.stats import ttest_1samp


cond_renames = {'ind': {'comInd': 'communication', 
                        'ind': 'independent'},
                'phy': {'comPhy': 'communication',
                        'phy': 'joint action'},
                'faces': {'faceFirst': 'communicate to screen',
                          'faceThird': 'communicate off screen',
                          'faceNoncom': 'non-communicative'}}

roi_renames = {'rightEVC': 'EVC',
               'rightMT': 'MT',
               'rightFFA': 'FFA',
               'rightEBA': 'EBA',
               'rightfSTS': 'fSTS',
               'rightsiSTS': 'SI-STS',
               'rightfacecomSTS': 'face-com-STS',
               'rightindSTS': 'com-ind-STS',
               'rightphySTS': 'com-phy-STS', 
               'rightTPJ': 'TPJ'}


class GroupROIResults:
    def __init__(self, args):
        self.process = 'GroupROIResults'
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.plot_indiv = args.plot_indiv
        self.condition = args.condition
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.individual_path = f'{self.derivatives_path}/CondROIAnalysis'
        self.out_path = f'{self.derivatives_path}/{self.process}'
        self.out_file = f'{self.out_path}/{self.condition}_group-rois.csv'
        self.fig_out = f'{self.out_path}/{self.condition}_group-rois.png'
        self.subjs = ['CP03', 'CP04']
        self.filtered_rois = ['rightMT', 'rightFFA', 'rightEBA', 'rightfSTS', 'rightsiSTS']
        self.conditions = []
        self.rois = []
        if 'ind' in self.condition:
            self.palette = 'magma'
            self.filtered_rois += ['rightfacecomSTS', 'rightphySTS', 'rightTPJ']
            self.c1 = 'communication'
            self.c2 = 'independent'
        elif 'phy' in self.condition:
            self.palette = 'magma'
            self.filtered_rois += ['rightfacecomSTS', 'rightindSTS', 'rightTPJ']
            self.c1 = 'communication'
            self.c2 = 'joint action'
        else:
            self.palette = 'mako'
            self.filtered_rois += ['rightindSTS', 'rightphySTS', 'rightTPJ']
            self.c1 = 'communicate off screen'
            self.c2 = 'non-communicative'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)

    def plotting_df(self, df):
        og_conds = list(cond_renames[self.condition].keys())
        self.conditions = list(cond_renames[self.condition].values())
        self.rois = [roi_renames[roi] for roi in self.filtered_rois]
        df_ = df.copy()
        df_ = df_.loc[df_.full_roi.isin(self.filtered_rois)].reset_index(drop=True)
        df_ = df_.loc[df_.condition.isin(og_conds)].reset_index(drop=True)
        df_['condition'] = df_['condition'].replace(cond_renames[self.condition])
        df_['full_roi'] = df_['full_roi'].replace(roi_renames)
        df_['condition'] = pd.Categorical(df_['condition'],
                                        categories=self.conditions,
                                        ordered=True)
        df_['full_roi'] = pd.Categorical(df_['full_roi'],
                                        categories=self.rois,
                                        ordered=True)
        if 'subject_label' in df_.columns:
            df_['subject_label'] = pd.Categorical(df_['subject_label'],
                                                categories=self.subjs,
                                                ordered=True)
        return df_
    
    def plot_responses(self, df, mean_df):
        if len(self.conditions) == 3:
            xs_og = [-.3, 0, .3]
        elif len(self.conditions) == 2:
            xs_og = [-.25, 0.25]
        else:
            print('not implementd')
        sns.set_context('talk')
        fig, ax = plt.subplots()
        sns.barplot(x='full_roi', y='response',
                    hue='condition', data=mean_df,
                    palette=self.palette, ax=ax,
                    legend=False)
        
        handles = []
        if self.plot_indiv:
            offset = 0 
            for roi, roi_df in df.groupby('full_roi', observed=True):
                for (subj, subj_df), color in zip(roi_df.groupby('subject_label', observed=True),
                                                ['black', 'dimgray']):
                    xs = [offset+x for x in xs_og]
                    ys = [cond_df.response for _, cond_df in subj_df.groupby('condition', observed=True)]
                    plt.plot(xs, ys, '-', color=color,
                            marker='o', alpha=.8, markersize=8)
                    if offset == 0: 
                        handles.append(Line2D([0], [0], color=color, label=subj))
                offset += 1

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_xlabel('ROIs')
        ax.set_ylabel('Beta weight')
        palette = sns.color_palette(self.palette, len(self.conditions))
        handles += [Patch(color=color, label=label) for color, label in zip(palette, self.conditions)]

        ax.set_xticks(np.arange(len(self.rois)))
        ax.set_xticklabels(self.rois, rotation=45, ha='right')
        ax.set_xlabel('')

        fig.legend(handles, self.subjs + self.conditions, 
                   loc='center left', bbox_to_anchor=(1.05, 0.5),
                   ncol=1, prop={'size': 12})
        plt.tight_layout()
        plt.savefig(self.fig_out, bbox_inches='tight')

    def load_data(self):
        sns.set_context(context='talk')
        files = glob(f'{self.individual_path}/*/*communicate.csv')
        files = [file for subj, file in product(self.subjs, files) if subj in file]
        df = []
        for file in tqdm(files, desc='Loading subject data'):
            sub = Path(file).name.split('_')[0].split('-')[-1]
            sub_df = pd.read_csv(file)
            sub_df['subject_label'] = sub
            sub_df['full_roi'] = sub_df['hemi'] + sub_df['roi']
            df.append(sub_df)
        return pd.concat(df, ignore_index=True).reset_index(drop=True)
    
    def power_analysis(self, df):
        summary = []
        for (roi, subj), roi_df in df.groupby(['full_roi', 'subject_label'], observed=True):
            roi_df.set_index('condition', inplace=True)
            for (c1, c2) in permutations(self.conditions, 2):
                diff = roi_df.loc[c1, 'response'] - roi_df.loc[c2, 'response']
                summary.append({'subj': subj, 'roi': roi, 'c1': c1, 'c2': c2, 'diff': diff})
        summary = pd.DataFrame(summary)
        summary = summary.set_index(['roi', 'c1', 'c2']).sort_index()
        
        for roi in ['SI-STS', 'fSTS', 'com-ind-STS', 'com-phy-STS', 'face-com-STS']:
            try: 
                vals = summary.loc[(roi, self.c1, self.c2), 'diff'].to_numpy()
                effect_size = np.mean(vals) / np.std(vals)
                n_obs = tt_ind_solve_power(effect_size=effect_size, alpha=0.01,
                                        power=0.8, alternative='larger')
                stats = ttest_1samp(vals, popmean=0, alternative='greater')
                print(f'{roi} {self.c1} {self.c2}')
                print(f't({int(stats.df)}) = {stats.statistic:.3f}, p = {stats.pvalue:.3f}')
                print(f'n_obs = {n_obs:.1f}')
            except: 
                print(f'{roi} not in data')
        return n_obs

    def run(self):
        if self.overwrite or not os.path.exists(self.out_file):
            df = self.load_data()
            df.to_csv(self.out_file, index=False)
        else:
            df = pd.read_csv(self.out_file)
        mean_df = df.groupby(['condition', 
                             'full_roi']).mean(numeric_only=True).reset_index()
        mean_df = self.plotting_df(mean_df)
        df = self.plotting_df(df)
        self.power_analysis(df)
        self.plot_responses(self.plotting_df(df), mean_df)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/communicate_pilot')
    parser.add_argument('--condition', '-c', type=str, default='faces')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--plot_indiv', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    GroupROIResults(args).run()


if __name__ == '__main__':
    main()
