import os
import argparse
from glob import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import permutations
from scipy.stats import ttest_rel
import numpy as np
from matplotlib.collections import  PathCollection

contrasts = [('face-first', 'face-noncom'),
             ('face-third', 'face-noncom'),
             ('com-ind', 'ind'),
             ('com-joint', 'joint')]

condition_rename = {'com_phy': 'com-joint', 'phy': 'joint',
                    'com_ind': 'com-ind',
                    'face_first': 'face-first', 'face_third': 'face-third',
                    'face_noncom': 'face-noncom'}

def p2star(p):
    if p < 0.05:
        if 0.001 > p: 
            star = '***'
        elif 0.01 > p >= 0.001:
            star = '**' 
        elif 0.05 > p >= 0.01:
            star = '*'
    else: 
        star = None
    return star

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
        self.stats_file = f'{self.out_path}/stats.csv'
        self.sub_nums = args.sub_nums
        self.subjs = [f'sub-{str(i).zfill(2)}' for i in self.sub_nums]
        print(vars(self))
        self.subj_colors = ['black', 'dimgray']
        self.palette = [
                "#E57373",  # Soft muted red
                "#3949AB",  # Deep muted navy (Dark Blue 1)
                "#5C6BC0",  # Dusty periwinkle (Dark Blue 2)
                "#90CAF9",  # Pale sky blue (Light Blue)
                "#673AB7",  # Rich muted violet (Dark Purple 1)
                "#9575CD",  # Soft lavender (Light Purple 1)
                "#7E57C2",  # Dusty plum (Dark Purple 2)
                "#B39DDB",  # Pale lilac (Light Purple 2)
            ]
        
        self.conditions = ['object', 'body', 
                           'face_first', 'face_third', 'face_noncom',
                           'com_phy', 'phy', 'com_ind', 'ind']
        self.plotting_conditions = ['object',
                                    'face-first', 'face-third', 'face-noncom',
                                    'com-joint', 'joint', 'com-ind', 'ind']
        self.hemis = ['l', 'r']
        self.rois = ['EVC', 'MT', 'FFA', 'EBA', 'fSTS',
                     'SI-STS', 'face-comSTS', 
                      'com-indSTS', 'com-phySTS', 
                     'TPJ']
        Path(self.out_path).mkdir(exist_ok=True, parents=True)

    def plot_rois(self, roi_response):
        sns.set_context('talk')
        fig, axes = plt.subplots(len(self.rois), 2,
                               sharex=True, sharey='row',
                               figsize=(10, 24))
        axes = axes.flatten()
        for ax, ((roi, hemi), df) in zip(axes, roi_response.groupby(['roi', 'hemi'], observed=True)):
            if self.plot_indiv:
                sns.barplot(x='trial_type', y='response',
                        hue='trial_type', legend=False,
                        ax=ax, data=df, palette=self.palette,
                        errorbar=None)
                swarm = sns.stripplot(x='trial_type', y='response', 
                                    #   hue='subject_label', ax=ax,
                                      ax=ax,
                                      legend=False, color='black',
                                      data=df, dodge=True, size=8)
                
                # Get swarmplot point positions
                swarm_positions = []
                for collection in ax.collections:
                    if isinstance(collection, PathCollection):  # This identifies the swarmplot points
                        swarm_positions.append(collection.get_offsets())

                # Create a dictionary to map (trial_type, subject_label) to coordinates
                coord_dict = {}
                for offsets in swarm_positions:
                    for x, y in offsets:
                        # Find which trial_type and subject_label this point corresponds to
                        closest = df.iloc[(df['response'] - y).abs().argsort()[:1]]
                        key = (closest['trial_type'].values[0], closest['subject_label'].values[0])
                        coord_dict[key] = (x, y)

                # Connect points for each subject in the correct order
                for subject in df['subject_label'].unique():
                    x_coords = []
                    y_coords = []
                    
                    # Get points in the order of plotting_conditions
                    for trial in self.plotting_conditions:
                        key = (trial, subject)
                        if key in coord_dict:
                            x, y = coord_dict[key]
                            x_coords.append(x)
                            y_coords.append(y)
                    
                    # Only draw lines if we have points to connect
                    if len(x_coords) > 1:
                        ax.plot(x_coords, y_coords, color='black', alpha=0.3)
            else:
                sns.barplot(x='trial_type', y='response',
                            hue='trial_type', legend=False,
                            ax=ax, data=df, palette=self.palette,
                            errorbar='se')

            ax.set_xticks(range(len(self.plotting_conditions)))
            ax.set_xticklabels(self.plotting_conditions, rotation=45, ha='right')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('')
            ax.set_ylabel(r'$\beta$')
            ax.set_title(f'{hemi} {roi}')
        fig.tight_layout()
        fig.savefig(f'{self.out_path}/summary.pdf')

    def plot_individual_roi(self, df, stats, hemi='r', 
                            rois=['EVC', 'MT', 'FFA', 'EBA', 'fSTS', 'SI-STS', 'TPJ']):
        df = df.loc[df.roi.isin(rois) & (df.hemi == hemi)].set_index('roi')
        stats = stats.loc[stats.roi.isin(rois) & (stats.hemi == hemi)].set_index('roi')
        sns.set_context('paper')
        fig, axes = plt.subplots(1, len(rois),
                                 figsize=(12.5, 3))
        axes = axes.flatten()
        for iroi, (ax, roi) in enumerate(zip(axes, rois)):
            roi_stats = stats.loc[roi]
            sns.barplot(x='trial_type', y='response',
                        hue='trial_type', legend=False,
                        ax=ax, data=df.loc[roi], 
                        palette=self.palette, errorbar='se')
            error_max = [line.get_ydata()[1] for line in ax.lines]

            face_pos = None
            stats_pos = []
            for _, row in roi_stats.iterrows():
                # Get the indicies of the conditions for the axis
                c1_ind = self.plotting_conditions.index(row['c1'])
                c2_ind = self.plotting_conditions.index(row['c2'])

                star = p2star(row['p'])
                if star is not None:
                    # Find the y positition to draw the line
                    if 'face' not in row['c1']:
                        pair_max = max([error_max[c1_ind], error_max[c2_ind]]) #top of error bar between pair
                        y_pos = pair_max + (max(error_max)*0.05) # add 5% of the tallest error bar to the pos
                    else:
                        if face_pos is None:
                            face_max = max([error_max[self.plotting_conditions.index(cond)] for cond in self.plotting_conditions if 'face' in cond])
                            face_pos = face_max + (max(error_max)*0.05)
                            y_pos = face_pos
                        else:
                            face_pos += max(error_max)*0.1
                            y_pos = face_pos
                            
                    ax.hlines(xmin=c1_ind, xmax=c2_ind, y=y_pos, color='k')
                    ax.text(x=c1_ind+((c2_ind-c1_ind)/2),
                            y=y_pos, s=star, ha='center', 
                            fontsize=8)
                    stats_pos.append(y_pos)
            
            ax.set_xticks(range(len(self.plotting_conditions)))
            ax.set_xticklabels(self.plotting_conditions, rotation=45, 
                               ha='right', fontsize=8)
            if stats_pos:
                ax.set_ylim([-.1, max(stats_pos)+(max(error_max)*0.1)])
            else:
                ax.set_ylim([-.1, ax.get_ylim()[-1]])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('')
            if iroi == 0: 
                ax.set_ylabel(r'$\beta$ values')
            else:
                ax.set_ylabel('')
            ax.set_title(f'{roi}')
        fig.tight_layout()
        fig.savefig(f'{self.out_path}/right_small_summary.pdf')

    def load_data(self):
        df = []
        for sub in tqdm(self.subjs, desc='Loading subject data'):
            file = f'{self.individual_path}/{sub}/roi_response.csv'
            sub_df = pd.read_csv(file)
            sub_df['subject_label'] = sub
            df.append(sub_df)
        return pd.concat(df, ignore_index=True).reset_index(drop=True)
    
    def statistical_analysis(self, mean_df):
        summary = []
        for (hemi, roi), roi_df in mean_df.groupby(['hemi', 'roi'], observed=True):
            roi_df.set_index('trial_type', inplace=True)
            for (c1, c2) in contrasts:
                a = roi_df.loc[c1].sort_values(by='subject_label')['response'].to_numpy()
                b = roi_df.loc[c2].sort_values(by='subject_label')['response'].to_numpy()
                stats = ttest_rel(a, b, alternative='greater')
                summary.append({'hemi': hemi, 'roi': roi,
                                 'c1': c1, 'c2': c2, 'diff': np.mean(a-b),
                                 't': stats.statistic, 'dof': stats.df, 
                                 'p': stats.pvalue})
        summary = pd.DataFrame(summary)
        summary.to_csv(self.stats_file, index=False)
        return summary

    def run(self):
        if self.overwrite or not os.path.exists(self.out_file):
            df = self.load_data()
            df.to_csv(self.out_file, index=False)
        else:
            df = pd.read_csv(self.out_file)
        mean_df = df.groupby(['roi', 'hemi',
                             'trial_type', 'subject_label']).mean(numeric_only=True).reset_index()
        mean_df['trial_type'] = mean_df['trial_type'].replace(condition_rename)
        mean_df = mean_df.loc[mean_df['trial_type'].isin(self.plotting_conditions)].reset_index(drop=True)
        mean_df.drop(columns='run', inplace=True)
        summary = self.statistical_analysis(mean_df)

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
        self.plot_individual_roi(mean_df, summary)  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_nums', nargs='*', type=int, help='List of elements', default=[1,2,3])
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--plot_indiv', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    GroupRunwiseResults(args).run()


if __name__ == '__main__':
    main()
