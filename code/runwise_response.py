import os
import argparse
from pathlib import Path
import numpy as np
from itertools import product
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import view_img_on_surf


roi_size = {'EVC': 0.05, 'MT': 0.1,
            'dyad-comSTS': .05, 
            'phySTS': .05, 'indSTS': 0.05,
            'face-comSTS': 0.05,
            'FFA': .1, 'fSTS': .1, 
            'EBA': .1}


roi_parc = {'dyad-comSTS': 'anatSTS',
            'phySTS': 'anatSTS',
            'indSTS': 'anatSTS',
            'face-comSTS': 'anatSTS'}


def roi_switcher(roi):
    if roi in list(roi_parc.keys()):
        return roi_parc[roi]
    else:
        return roi


def list_of_dict_mean(list_of_dicts):
    # Initialize a dictionary to hold the sum and count for each key
    sum_dict = {}
    count_dict = {}

    # Iterate through each dictionary in the list
    for d in list_of_dicts:
        for key, val in d.items():
            if key not in sum_dict:
                sum_dict[key] = np.zeros_like(val)
                count_dict[key] = 0
            sum_dict[key] += val
            count_dict[key] += 1

    # Compute the mean for each key
    return {key: sum_dict[key] / count_dict[key] for key in sum_dict}


class RunwiseResponse:
    def __init__(self, args):
        self.process = 'RunwiseResponse'
        self.overwrite = args.overwrite
        self.task_label = 'communicate'
        self.space_label = args.space_label
        self.subject_label = args.subject_label
        self.dataset_path = args.dataset_path
        self.visualize_rois = args.visualize_rois
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.out_path = f'{self.derivatives_path}/{self.process}/sub-{self.subject_label}'
        self.out_file = f'{self.out_path}/roi_response.csv'
        self.glm_path = f'{self.derivatives_path}/NilearnGLMRunwise/sub-{self.subject_label}'
        self.parcel_path = f'{self.derivatives_path}/parcels-{self.space_label}'
        self.froi_path = f'{self.derivatives_path}/FunctionalROIs/sub-{self.subject_label}/'
        self.nifti_path = f'{self.dataset_path}/sub-{self.subject_label}/*/func'
        self.conditions = ['object', 'body', 
                           'face_first', 'face_third', 'face_noncom',
                           'com_phy', 'phy', 'com_ind', 'ind']
        self.n_runs = 9
        self.hemis = ['l', 'r']
        self.rois = ['EVC', 'MT', 'FFA', 'EBA', 'fSTS',
                     'SI-STS', 'face-comSTS', 'dyad-comSTS',
                     'TPJ']
        self.loc_rois = ['SI-STS', 'TPJ']
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def get_contrast(self, response_dict, roi):
        if roi in ['MT', 'EVC']:
            out = np.mean([response_dict[cond] for cond in self.conditions], axis=0)
        elif roi in ['FFA', 'fSTS']:
            a = np.mean([response_dict[cond] for cond in ['face_first', 'face_third', 'face_noncom']], axis=0)
            b = response_dict['object']
            out = a - b
        elif roi == 'face-comSTS':
            a = np.mean([response_dict[cond] for cond in ['face_first', 'face_third']], axis=0)
            b = response_dict['face_noncom']
            out = a - b
        elif roi == 'dyad-comSTS':
            a = np.mean([response_dict[cond] for cond in ['com_ind', 'com_phy']], axis=0)
            b = np.mean([response_dict[cond] for cond in ['ind', 'phy']], axis=0)
            out = a - b
        elif roi == 'phySTS':
            out = response_dict['com_phy'] - response_dict['phy']
        elif roi == 'indSTS':
            out = response_dict['com_ind'] - response_dict['ind']
        elif roi == 'EBA':
            out = response_dict['body'] - response_dict['object']
        return out.flatten()
    
    def define_roi(self, img_arr, run=None, hemi='r', roi='EBA'):
        # Get mask
        mask = nib.load(f'{self.parcel_path}/{hemi}{roi_switcher(roi)}.nii.gz')
        mask_arr = mask.get_fdata().flatten()
        anti_mask = np.invert(mask_arr.astype(bool))
        n_voxels_to_keep = int(mask_arr.sum() * roi_size[roi])

        # Get the indices of the highest response
        roi_response = np.nan_to_num(img_arr, -1000.)
        roi_response[anti_mask] = -1000.
        voxel_sorted_indices = np.argsort(roi_response) #sort smallest to largest
        voxel_sorted_indices = voxel_sorted_indices[::-1] #sort largest to smallest
        voxels_to_keep = voxel_sorted_indices[:n_voxels_to_keep]

        if self.visualize_rois: 
            # Visualize the ROI
            out_arr = np.zeros_like(mask_arr)
            out_arr[voxels_to_keep] = 1
            out = nib.Nifti1Image(out_arr.reshape(mask.shape), affine=mask.affine)
            if run is not None: 
                out_name = f'{self.out_path}/{hemi}{roi}_run-{run}'
            else:
                out_name = f'{self.out_path}/{hemi}{roi}'
            # nib.save(out, f'{out_name}.nii.gz')
            view = view_img_on_surf(out, threshold=0.1)
            view.save_as_html(f'{out_name}.html')  
        return voxels_to_keep
    
    def load_responses(self):
        response_dict = []
        for run in range(self.n_runs):
            response_dict.append(dict())
            for condition in self.conditions:
                img_file = f'{self.glm_path}/sub-{self.subject_label}_task-{self.task_label}_contrast-{condition}_run-{run+1}.nii.gz'
                response_dict[-1][condition] = nib.load(img_file).get_fdata()
        return response_dict
    
    def get_roi_response(self, responses):
        out = []
        iterator = tqdm(range(self.n_runs),
                        total=self.n_runs, leave=True,
                        desc='getting runwise response in the ROIs')
        for run in iterator:
            # Make the data frame for defining ROIs
            roi_def_response = [resp for i, resp in enumerate(responses) if i not in ([run] if isinstance(run, int) else run)]
            roi_def_response = list_of_dict_mean(roi_def_response) #compute the mean across runs
            
            for hemi, roi in product(self.hemis, self.rois):
                if roi not in self.loc_rois: 
                    contrast = self.get_contrast(roi_def_response, roi)
                    roi_indices = self.define_roi(contrast, run=run, roi=roi)
                else:
                    # If the ROI is defined from other data, load the mask
                    roi_mask = nib.load(f'{self.froi_path}/sub-{self.subject_label}_{hemi}{roi}.nii.gz')
                    roi_indices = roi_mask.get_fdata().astype(bool).flatten()

                # Estimate the response in the ROI
                roi_response = [responses[i] for i in ([run] if isinstance(run, int) else run)]
                if len(roi_response) > 1:
                    roi_response = list_of_dict_mean(roi_response)
                else:
                    roi_response = roi_response[0]

                for condition, response in roi_response.items():
                    out.append({'roi': roi, 'hemi': hemi,
                                'run': run, 'trial_type': condition,
                                'response': response.flatten()[roi_indices].mean()})
        return pd.DataFrame(out)

    def plot_rois(self, roi_response):
        sns.set_context('talk')
        colors = [
                    "#E57373",  # Soft muted red
                    "#FFB74D",  # Warm peachy-orange
                    "#3949AB",  # Deep muted navy (Dark Blue 1)
                    "#5C6BC0",  # Dusty periwinkle (Dark Blue 2)
                    "#90CAF9",  # Pale sky blue (Light Blue)
                    "#673AB7",  # Rich muted violet (Dark Purple 1)
                    "#9575CD",  # Soft lavender (Light Purple 1)
                    "#7E57C2",  # Dusty plum (Dark Purple 2)
                    "#B39DDB",  # Pale lilac (Light Purple 2)
                ]

        fig, axes = plt.subplots(len(self.rois), 2,
                               sharex=True, sharey='row',
                               figsize=(10, 24))
        axes = axes.flatten()
        for ax, ((roi, hemi), df) in zip(axes, roi_response.groupby(['roi', 'hemi'], observed=True)):
            sns.barplot(x='trial_type', y='response',
                        hue='trial_type', legend=False,
                        ax=ax, data=df, palette=colors)
            ax.set_xticks(range(len(self.conditions)))
            ax.set_xticklabels(self.conditions, rotation=45, ha='right')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('')
            ax.set_ylabel(r'$\beta$')
            ax.set_title(f'{hemi} {roi}')

            ind_fig, ind_ax = plt.subplots()
            sns.barplot(x='trial_type', y='response',
                        hue='trial_type', legend=False,
                        ax=ind_ax, data=df, palette=colors)
            ind_ax.set_xticks(range(len(self.conditions)))
            ind_ax.set_xticklabels(self.conditions, rotation=45, ha='right')
            ind_ax.spines['right'].set_visible(False)
            ind_ax.spines['top'].set_visible(False)
            ind_ax.set_xlabel('')
            ind_fig.tight_layout()
            ind_fig.savefig(f'{self.out_path}/{hemi}{roi}.pdf')
        fig.tight_layout()
        fig.savefig(f'{self.out_path}/../sub-{self.subject_label}_summary.pdf')

    def run(self):
        if not os.path.exists(self.out_file) or self.overwrite: 
            responses = self.load_responses()
            roi_response = self.get_roi_response(responses)
            roi_response.to_csv(self.out_file, index=False)
        else:
            roi_response = pd.read_csv(self.out_file)
        roi_response['trial_type'] = pd.Categorical(roi_response['trial_type'],
                                            ordered=True,
                                            categories=self.conditions)
        roi_response['roi'] = pd.Categorical(roi_response['roi'],
                                            ordered=True,
                                            categories=self.rois)
        roi_response['hemi'] = pd.Categorical(roi_response['hemi'],
                                            ordered=True,
                                            categories=self.hemis)
        self.plot_rois(roi_response)


def main():
    parser = argparse.ArgumentParser(description='Run a standard first-level GLM on the localizer tasks')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/mindhive/nklab3/users/emaliem/communicate_pilot')
    parser.add_argument('--subject_label', '-s', type=str, default='CP03',
                         help='Subject for the GLM')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--visualize_rois', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    processor = RunwiseResponse(args)
    processor.run()


if __name__ == '__main__':
    main()
