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
from nilearn.plotting import plot_glass_brain, view_img_on_surf
from itertools import combinations


roi_size = {'EVC': 0.05, 'MT': 0.1,
            'com-indSTS': .05,
            'com-phySTS': 0.05,
            'face-comSTS': 0.05,
            'FFA': .1, 'fSTS': .1, 
            'EBA': .1}


roi_parc = {'dyad-comSTS': 'anatSTS',
            'com-indSTS': 'anatSTS',
            'com-phySTS': 'anatSTS',
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


def count_overlaps(dictionary):
    # Convert numpy arrays to sets of elements for each key
    sets = {key: set(arr) for key, arr in dictionary.items()}
    
    # Get all unique pairs of keys
    key_pairs = combinations(dictionary.keys(), 2)
    
    # Compute overlaps for each pair
    long_data = []
    for key1, key2 in key_pairs:
        intersection = sets[key1] & sets[key2]
        long_data.append({'roi1': key1, 'roi2': key2, 'overlap': len(intersection)})
    
    return pd.DataFrame(long_data)


def remove_overlapping_values(original_dict, regions_to_filter):
    """
    Removes values from the arrays of specified regions that overlap with any other region in the dictionary.
    
    Parameters:
    - original_dict: dict where keys are regions and values are numpy arrays
    - regions_to_filter: list of regions to filter overlaps from
    
    Returns:
    - A new dictionary with overlapping values removed from the specified regions
    """
    # Create a copy of the original dictionary to avoid modifying it directly
    filtered_dict = {k: np.copy(v) for k, v in original_dict.items()}
    
    for region in regions_to_filter:
        if region not in filtered_dict:
            continue  # Skip if region not in dictionary
        
        # Collect all values from other regions (including those in regions_to_filter except current region)
        other_values = set()
        for other_region, arr in original_dict.items():
            if other_region != region:
                other_values.update(arr)
        
        # Convert to numpy array for efficient operations
        other_values_arr = np.array(list(other_values)) if other_values else np.array([])
        
        # Remove overlapping values from the current region
        original_arr = filtered_dict[region]
        mask = ~np.isin(original_arr, other_values_arr)
        filtered_dict[region] = original_arr[mask]
    
    return filtered_dict


class RunwiseResponse:
    def __init__(self, args):
        self.process = 'RunwiseResponse'
        self.task_label = 'communicate'
        self.space_label = args.space_label
        self.subject_label = str(args.subject_label).zfill(2)
        self.dataset_path = args.dataset_path
        self.n_runs = args.n_runs
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.out_path = f'{self.derivatives_path}/{self.process}/sub-{self.subject_label}'
        self.out_file = f'{self.out_path}/roi_response.csv'
        self.overlap_out_file = f'{self.out_path}/roi_overlap.csv'
        self.glm_path = f'{self.derivatives_path}/NilearnGLMRunwise/sub-{self.subject_label}'
        self.parcel_path = f'{self.derivatives_path}/parcels-{self.space_label}'
        self.froi_path = f'{self.derivatives_path}/FunctionalROIs/sub-{self.subject_label}/'
        self.nifti_path = f'{self.dataset_path}/sub-{self.subject_label}/*/func'
        self.conditions = ['object', 'body', 
                           'face_first', 'face_third', 'face_noncom',
                           'com_phy', 'phy', 'com_ind', 'ind']
        self.plotting_conditions = ['object', 'body',
                                    'face_first', 'face_third', 'face_noncom',
                                    'com_phy', 'phy', 'com_ind', 'ind']
        self.hemis = ['l', 'r']
        self.rois = ['EVC', 'MT', 'FFA', 'EBA', 'fSTS',
                     'SI-STS', 'TPJ']
                    #  'face-comSTS',  'com-indSTS', 'com-phySTS',
        self.overlap_rois = ['EVC', 'FFA', 'EBA', 'MT', 'TPJ']
        self.loc_rois = ['SI-STS', 'TPJ']
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def get_contrast(self, response_dict, roi):
        if roi in ['MT', 'EVC']:
            out = np.mean([response_dict[cond] for cond in ['object', 'phy']], axis=0)
        elif roi in ['FFA', 'fSTS']:
            a = np.mean([response_dict[cond] for cond in ['face_first', 'face_noncom']], axis=0)
            b = response_dict['object']
            out = a - b
        elif roi == 'face-comSTS':
            a = np.mean([response_dict[cond] for cond in ['face_first', 'face_third']], axis=0)
            b = response_dict['face_noncom']
            out = a - b
        elif roi == 'com-indSTS':
            out = response_dict['com_ind'] - response_dict['ind']
        elif roi == 'com-phySTS':
            out = response_dict['com_phy'] - response_dict['phy']
        elif roi == 'EBA':
            out = response_dict['body'] - response_dict['object']
        return out.flatten()
    
    def define_roi(self, img_arr, run, hemi, roi):
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

        return voxels_to_keep
    
    def load_responses(self):
        response_dict = []
        affine = None
        for run in range(self.n_runs):
            response_dict.append(dict())
            for condition in self.conditions:
                img_file = f'{self.glm_path}/sub-{self.subject_label}_task-{self.task_label}_contrast-{condition}_run-{run+1}.nii.gz'
                response_dict[-1][condition] = nib.load(img_file).get_fdata()
                if affine is None:
                    affine = nib.load(img_file).affine
        return response_dict, affine
    
    def visualize_rois(self, out, out_name):
        nib.save(out, f'{out_name}.nii.gz')
        plot_glass_brain(out, f'{out_name}.pdf')
        view = view_img_on_surf(out, threshold=0.1)
        view.save_as_html(f'{out_name}.html')  
        plt.close()

    def get_roi_response(self, responses, affine=np.eye(4)):
        out = []
        iterator = tqdm(range(self.n_runs),
                        total=self.n_runs, leave=True,
                        desc='getting runwise response in the ROIs')
        overlap_df = []
        roi_images = dict()
        for run in iterator:
            # Make the data frame for defining ROIs
            roi_def_response = [resp for i, resp in enumerate(responses) if i not in ([run] if isinstance(run, int) else run)]
            roi_def_response = list_of_dict_mean(roi_def_response) #compute the mean across runs
            
            for hemi in self.hemis:
                overlap_dict = {}
                for roi in self.rois:
                    if roi not in self.loc_rois: 
                        contrast = self.get_contrast(roi_def_response, roi)
                        roi_indices = self.define_roi(contrast, run=run, roi=roi, hemi=hemi)
                    else:
                        # If the ROI is defined from other data, load the mask
                        roi_mask = nib.load(f'{self.froi_path}/sub-{self.subject_label}_{hemi}{roi}.nii.gz')
                        roi_indices = np.where(roi_mask.get_fdata().astype(bool).flatten())[0]
                    overlap_dict[roi] = roi_indices
            
                # Save overlap
                df = count_overlaps(overlap_dict)
                df['hemi'], df['run'] = hemi, run
                overlap_df.append(df)

                filtered_roi_indices = remove_overlapping_values(overlap_dict, self.overlap_rois)
                for roi in self.rois: 
                    # Estimate the response in the ROI
                    roi_response = [responses[i] for i in ([run] if isinstance(run, int) else run)]
                    if len(roi_response) > 1:
                        roi_response = list_of_dict_mean(roi_response)
                    else:
                        roi_response = roi_response[0]

                    for condition, response in roi_response.items():
                        out.append({'roi': roi, 'hemi': hemi,
                                    'run': run, 'trial_type': condition,
                                    'response': response.flatten()[filtered_roi_indices[roi]].mean()})
                        
                    # Add ROI image to the array by adding a True value to an existing array or creating new boolean array
                    roi_array = np.zeros_like(response.flatten(), dtype='bool')
                    roi_array[filtered_roi_indices[roi]] = True
                    roi_array = roi_array.reshape(response.shape)
                    if f'{hemi}{roi}' not in roi_images.keys():
                        roi_images[f'{hemi}{roi}'] = roi_array
                    else: 
                        roi_images[f'{hemi}{roi}'] += roi_array

        # Save the ROIs images. The images contain True if that voxel is present in any of the contrasts across runs
        for hemi in self.hemis:
            for roi in self.rois:
                 roi_array = roi_images[f'{hemi}{roi}'].astype(float)
                 roi_image = nib.Nifti1Image(roi_array,
                                             affine=affine)
                 self.visualize_rois(roi_image, f'{self.out_path}/{hemi}{roi}')
        
        #Get the average overlap across runs
        overlap_df = pd.concat(overlap_df).groupby(['roi1', 'roi2', 'hemi']).mean().reset_index()
        return pd.DataFrame(out), overlap_df

    def plot_rois(self, roi_response):
        sns.set_context('talk')
        colors = [
                    "#E57373",  # Soft muted red
                    '#6FC276', # Soft green
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
            ax.set_xticks(range(len(self.plotting_conditions)))
            ax.set_xticklabels(self.plotting_conditions, rotation=45, ha='right')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('')
            ax.set_ylabel(r'$\beta$')
            ax.set_title(f'{hemi} {roi}')

        fig.tight_layout()
        fig.savefig(f'{self.out_path}/../sub-{self.subject_label}_summary.pdf')

    def run(self):
        responses, affine = self.load_responses()
        roi_response, overlap = self.get_roi_response(responses, affine)
        roi_response.to_csv(self.out_file, index=False)
        overlap.to_csv(self.overlap_out_file, index=False)
        overlap.to_csv(index=False)

        roi_response = roi_response.loc[roi_response['trial_type'].isin(self.plotting_conditions)].reset_index(drop=True)
        roi_response['trial_type'] = pd.Categorical(roi_response['trial_type'],
                                            ordered=True,
                                            categories=self.plotting_conditions)
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
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=int, default=1,
                         help='Subject for the GLM')
    parser.add_argument('--n_runs', '-n', type=int, default=9,
                         help='Number of runs to load')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = RunwiseResponse(args)
    processor.run()


if __name__ == '__main__':
    main()
