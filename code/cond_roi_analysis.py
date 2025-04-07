import os
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.patches import Patch


roi_list = ['EVC', 'MT', 
            'FFA', 'fSTS', 'EBA',
            'indSTS', 'phySTS', 
            'facecomSTS', 'siSTS', 'TPJ']

task_exclude = {'dysoc': ['EVC', 'MT', 'FFA', 'fSTS'],
                'communicate': [],#['ciSTS', 'cpSTS', 'fSTS'],
                'pointlight': ['siSTS']}

hemi_abbr = {'l': 'left', 'r': 'right'} #


def mask_img(mask, img):
    # Load file if not loaded already
    mask = nib.load(mask) if type(mask) is str else mask
    img = nib.load(img) if type(img) is str else img

    # Make numpy is not already
    mask = mask.get_fdata() if type(mask) is not np.ndarray else mask
    img = img.get_fdata() if type(img) is not np.ndarray else img

    img_flat = img.flatten()
    mask_flat = mask.flatten().astype(bool)

    return img_flat[mask_flat]


class CondROIAnalysis:
    def __init__(self, args):
        self.process = 'CondROIAnalysis'
        self.subject_label = args.subject_label
        self.task_label = args.task_label
        self.dataset_path = args.dataset_path
        self.overwrite = args.overwrite
        self.reduced_plot = args.reduced_plot
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.glm_path = f'{self.derivatives_path}/NilearnGLM/sub-{self.subject_label}'
        self.roi_path = f'{self.derivatives_path}/FunctionalROIs/sub-{self.subject_label}'
        self.out_path = f'{self.derivatives_path}/{self.process}/sub-{self.subject_label}'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

        if self.reduced_plot:
            print('to do')
            # self.df_out = f'{self.out_path}/sub-{self.subject_label}_task-{self.task_label}_reduced.csv'
            # self.fig_out = f'{self.out_path}/sub-{self.subject_label}_task-{self.task_label}_reduced.png'
            # self.task_conditions = {'dysoc': ['faces', 'bodies', 'objects'],
            #        'communicate': ['05unblurComPhyPlus05UnblurComInd', 'unblurInd', 'unblurPhy',
            #                        '05blurComPhyPlus05BlurComInd', 'blurInd', 'blurPhy',
            #                        'faceFirst', 'faceThird', 'faceNoncom'],
            #        'pointlight': ['interact', 'noninteract']}
            # self.conditions = ['communication', 'independent', 'joint action',
            #             'blurred communication', 'blurred independent', 'blurred joint action',
            #             'face to', 'face away', 'face independent']
            # self.custom_palette = [
            #     "#4E1F6F",  # sat purple
            #     "#A74779",  # sat red
            #     "#E89276",  # sat orange
            #     "#6730AA",  # bright purple
            #     "#D661CB",  # bright red
            #     "#EEACA2",  # red pink
            #     "#3E346B",  # Darker green
            #     "#347AA2",  # Dark green
            #     "#4AC2AD"   # Light green
            # ]
            # self.cond_renames = {task_cond: cond for task_cond, cond in zip(self.task_conditions['communicate'], self.conditions)}
        else:
            self.df_out = f'{self.out_path}/sub-{self.subject_label}_task-{self.task_label}.csv'
            self.fig_out = f'{self.out_path}/sub-{self.subject_label}_task-{self.task_label}.png'
            self.task_conditions = {'dysoc': ['faces', 'bodies', 'objects'],
                   'communicate': ['comInd', 'comPhy', 'ind', 'phy',
                                   'faceFirst', 'faceThird', 'faceNoncom'],
                   'pointlight': ['interact', 'noninteract']}
            self.conditions = ['com_ind', 'com_joint', 'independent', 'joint action',
                           'face first', 'face third', 'face independent']
            self.custom_palette = [
                "#003366",  # Darker blue
                "#004080",  # Dark blue
                "#66B2FF",  # Light blue
                "#B3D9FF",  # Lighter blue
                "#006600",  # Darker green
                "#008000",  # Dark green
                "#66FF66"   # Light green
            ]
            self.cond_renames = {task_cond: cond for task_cond, cond in zip(self.task_conditions['communicate'], self.conditions)}


    def get_responses(self):
        print(f'{self.task_conditions[self.task_label]=}')
        rois = [item for item in roi_list if item not in task_exclude[self.task_label]]
        print(f'{rois=}')
        df = []
        for cond in self.task_conditions[self.task_label]:
            file = f'sub-{self.subject_label}_task-{self.task_label}_contrast-{cond}_stat-effect_statmap.nii.gz'
            betas = nib.load(f'{self.glm_path}/{file}')
            for hemi, roi in product(['l', 'r'], rois): 
                mask = f'{self.roi_path}/sub-{self.subject_label}_{hemi}{roi}.nii.gz'
                response = mask_img(mask, betas)
                df.append({'condition': cond, 'hemi': hemi_abbr[hemi],
                           'roi': roi, 'full_roi': f'{hemi}{roi}',
                           'response': response.mean()})
        return pd.DataFrame(df)
    
    def plot_responses(self, df):
        df_ = df.copy()
        if self.task_label == 'communicate': 
            df_['condition'] = df_['condition'].replace(self.cond_renames)
            df_['condition'] = pd.Categorical(df_['condition'], 
                                              categories=self.conditions, ordered=True)
            palette = self.custom_palette
        else:
            palette = 'rocket'
        sns.set_context('talk')
        fig, axes = plt.subplots(1, len(hemi_abbr), sharey=True, figsize=(20,6))
        for ax, (hemi_name, hemi_df) in zip(axes, df_.groupby('hemi')):
            sns.barplot(x='roi', y='response',
                        hue='condition', data=hemi_df,
                        palette=palette, ax=ax,
                        legend=False, saturation=1)
            ax.set_xlabel(hemi_name.capitalize())
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if hemi_name == 'right':
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', which='both', length=0)
            else:
                ax.set_ylabel('Beta weight')
                palette = sns.color_palette(palette, len(self.conditions))
                handles = [Patch(color=color, label=label) for color, label in zip(palette, self.conditions)]
        fig.legend(handles, self.conditions, loc='center left', bbox_to_anchor=(1.05, 0.5),
                ncol=1, title='Trial Type', prop={'size': 12})
        plt.tight_layout()
        plt.savefig(self.fig_out, bbox_inches='tight')
        
    def run(self):
        if not os.path.exists(self.df_out) or self.overwrite:
            df = self.get_responses()
            df.to_csv(self.df_out, index=False)
        else:
            df = pd.read_csv(self.df_out)
        self.plot_responses(df)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_label', '-s', type=str, default='CP01',
                        help='Subject for the GLM')
    parser.add_argument('--task_label', '-t', type=str, default='communicate',
                        help='Subject for the GLM')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/communicate_pilot')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reduced_plot', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    CondROIAnalysis(args).run()


if __name__ == '__main__':
    main()