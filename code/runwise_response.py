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


n_runs = {'pointlight': 4, 'tom': 2, 'communicate': 9}


roi_size = {'EVC': 0.05, 'MT': 0.1,
            'comind-STS': .05,
            'comphy-STS': 0.05,
            'facecom-STS': 0.05,
            'dyadcom-STS': 0.05,
            'com-STS': .05,
            'phy-STS': .05,
            'FFA': .1, 'fSTS': .1, 
            'EBA': .1, 'SI-STS': .05,
            'TPJ': .1}

task_rois = {'communicate': ['EVC', 'MT', 'FFA', 'EBA', 'fSTS',
                             'comind-STS', 'comphy-STS', 'facecom-STS', 
                             'dyadcom-STS', 'com-STS', 'phy-STS'],
             'pointlight': ['SI-STS'],
             'tom': ['TPJ']}

task_conditions = {'communicate': ['object', 'body', 
                                   'face_first', 'face_third', 'face_noncom',
                                   'com_phy', 'phy', 'com_ind', 'ind'],
                   'pointlight': ['interact', 'noninteract'],
                   'tom': ['belief', 'photo']}


class RunwiseResponse:
    def __init__(self, args):
        self.process = 'RunwiseResponse'
        self.space_label = args.space_label
        self.subject_label = str(args.subject_label).zfill(2)
        self.dataset_path = args.dataset_path
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
                           'com_phy', 'phy', 'com_ind', 'ind', 'interact',
                           'noninteract', 'belief', 'photo']
        self.plotting_conditions = ['object', 'body',
                                    'face_first', 'face_third', 'face_noncom',
                                    'com_phy', 'phy', 'com_ind', 'ind',
                                    'interact', 'noninteract',
                                    'belief', 'photo']
        self.tasks = list(n_runs.keys())
        self.hemis = ['l', 'r']
        self.rois = roi_size.keys()
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def load_responses(self):
        response_dict = dict()
        for task in self.tasks:
            for condition in task_conditions[task]:
                response_dict[condition] = []
                for run in range(n_runs[task]):
                    img_file = f'{self.glm_path}/sub-{self.subject_label}_task-{task}_contrast-{condition}_run-{run+1}.nii.gz'
                    if os.path.exists(img_file):
                        img = nib.load(img_file)
                        response_dict[condition].append(img.get_fdata())
        return response_dict
    
    def visualize_rois(self, out, out_name):
        nib.save(out, f'{out_name}.nii.gz')
        plot_glass_brain(out, f'{out_name}.pdf')
        view = view_img_on_surf(out, threshold=0.1)
        view.save_as_html(f'{out_name}.html')  
        plt.close()

    def load_roi_mask(self, run, hemi, roi):
        """Load ROI mask for a specific run and return voxel indices."""
        roi_file = f'{self.glm_path}/sub-{self.subject_label}_run-{run+1}_{hemi}{roi}.nii.gz'
        if not os.path.exists(roi_file):
            # print(f"Looking for ROI file: {roi_file}")
            # print(f"ROI file not found: {roi_file}")
            return None
        
        roi_mask = nib.load(roi_file).get_fdata().flatten()
        voxels = np.where(roi_mask > 0)[0]
        return voxels if len(voxels) > 0 else None

    def extract_response_for_condition(self, responses, condition, run, voxels):
        """Extract mean response for a condition given voxels and run."""
        if condition not in responses or len(responses[condition]) <= run:
            return None
        
        cond_response = responses[condition][run]
        return np.mean(cond_response.flatten()[voxels])

    def process_roi_in_defining_task(self, responses, roi, task):
        """Process ROI for the task where it's defined."""
        task_responses = []
        
        for hemi in self.hemis:
            for run in range(n_runs[task]):
                voxels = self.load_roi_mask(run, hemi, roi)
                # print(f"Loading ROI {roi}, run {run+1}, hemi {hemi}: {'Found' if voxels is not None else 'Not Found'}")
                if voxels is None:
                    continue
                
                for condition in task_conditions[task]:
                    mean_response = self.extract_response_for_condition(responses, condition, run, voxels)
                    if mean_response is not None:
                        task_responses.append({
                            'run': run,
                            'task': task,
                            'trial_type': condition,
                            'response': mean_response,
                            'roi': roi,
                            'hemi': hemi
                        })
        
        return task_responses

    def process_roi_in_other_tasks(self, responses, roi, excluded_task):
        """Process ROI for tasks where it's NOT defined (using other tasks' data)."""
        other_responses = []
        other_tasks = [t for t in self.tasks if t != excluded_task]
        
        for other_task in other_tasks:
            for hemi in self.hemis:
                for run in range(n_runs[other_task]):
                    voxels = self.load_roi_mask(run, hemi, roi)
                    # print(f"Loading ROI {roi} for task {other_task}, run {run+1}, hemi {hemi}: {'Found' if voxels is not None else 'Not Found'}")
                    if voxels is None:
                        continue
                    
                    for condition in task_conditions[other_task]:
                        mean_response = self.extract_response_for_condition(responses, condition, run, voxels)
                        if mean_response is not None:
                            other_responses.append({
                                'run': run,
                                'task': other_task,
                                'trial_type': condition,
                                'response': mean_response,
                                'roi': roi,
                                'hemi': hemi
                            })
        
        return other_responses

    def get_roi_response(self, responses):
        """Extract responses for all ROIs across all tasks."""
        out = []
        
        for roi in self.rois:
            for task in self.tasks:
                # print(f"Processing Task: {task}")
                if roi in task_rois[task]:
                    # ROI is defined in this task - use this task's data
                    roi_data = self.process_roi_in_defining_task(responses, roi, task)
                else:
                    # ROI not defined in this task - use other tasks' data
                    roi_data = self.process_roi_in_other_tasks(responses, roi, task)
                
                out.extend(roi_data)

        return pd.DataFrame(out)

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
                    "#26A69A",  # Dark teal (interact_pointlight)
                    "#80CBC4",  # Light teal (noninteract_pointlight)
                    "#FF9800",  # Dark amber (belief)
                    "#FFCC80",  # Light amber (photo)
                ]

        fig, axes = plt.subplots(len(self.rois), 2,
                               sharex=True, sharey='row',
                               figsize=(10, 24))
        axes = axes.flatten()
        for ax, ((roi, hemi), df) in zip(axes, roi_response.groupby(['roi', 'hemi'], observed=True)):
            sns.barplot(x='trial_type', y='response',
                        hue='trial_type', legend=False,
                        dodge=False, ax=ax,
                        data=df, palette=colors)
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
        responses = self.load_responses()
        roi_response = self.get_roi_response(responses)
        roi_response.groupby(['hemi', 'roi', 'trial_type']).mean(numeric_only=True).reset_index().to_csv(self.out_file, index=False)

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
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject_label', '-s', type=int, default=2,
                         help='Subject for the GLM')
    parser.add_argument('--space_label', type=str, default='MNI152NLin2009cAsym',
                         help='Space of the GLM')
    args = parser.parse_args()

    processor = RunwiseResponse(args)
    processor.run()


if __name__ == '__main__':
    main()
