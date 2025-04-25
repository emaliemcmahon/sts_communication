import os
import nibabel as nib
import argparse
import re

# Dictionary mapping tasks to their expected number of TRs
task_duration = {
    'communicate': 164,
    'pointlight': 88,
    'eploc': 146,
    'tom': 136,
}

def count_trs(nifti_file):
    """Count the number of TRs in a NIfTI file."""
    img = nib.load(nifti_file)
    return img.shape[-1]  # Assumes 4D NIfTI, with time as the last dimension

def extract_task_from_filename(filename):
    """Extract the task name from the filename."""
    match = re.search(r'task-(\w+)_', filename)
    if match:
        return match.group(1)
    return None


class RMAbortedRuns:
    def __init__(self, directory, testing_mode):
        self.directory = directory
        self.testing_mode = testing_mode

    def rename(self, file, nifti_path, new_filename, new_path):
        # Rename the NIfTI file
        if file != new_filename: 
            os.rename(nifti_path, new_path)
            print(f"Renamed {file} to {new_filename}")

        # Rename the corresponding JSON file if it exists
        json_file = file.replace('.nii.gz', '.json')
        json_path = os.path.join(self.directory, json_file)
        if os.path.exists(json_path):
            new_json_filename = new_filename.replace('.nii.gz', '.json')
            new_json_path = os.path.join(self.directory, new_json_filename)
            if json_file != new_json_filename:
                os.rename(json_path, new_json_path)
            print(f"Renamed {json_file} to {new_json_filename}")

    def filter_and_rename_runs(self):
        """Filter runs based on TRs and rename remaining files, starting run numbering from 1 for each task."""
        nifti_files = [f for f in os.listdir(self.directory) if f.endswith(".nii.gz")]
        nifti_files.sort()  # Ensure files are processed in order

        # Dictionary to keep track of run counters for each task
        task_run_counters = {task: 1 for task in task_duration.keys()}

        for file in nifti_files:
            nifti_path = os.path.join(self.directory, file)
            task = extract_task_from_filename(file)

            if task and task in task_duration:
                expected_trs = task_duration[task]
                actual_trs = count_trs(nifti_path)

                if actual_trs == expected_trs:
                    # Construct new filename with continuous run numbering for this task
                    new_filename = re.sub(r'run-\d+', f'run-{task_run_counters[task]:02d}', file)
                    self.rename(file, nifti_path, 
                                new_filename,
                                os.path.join(self.directory, new_filename))

                    # Increment the run counter for this task
                    task_run_counters[task] += 1
                elif actual_trs > expected_trs: 
                    print(f'cutting extra TRs from {file}')
                    nifti_long = nib.load(nifti_path)
                    nifti_long_arr = nifti_long.get_fdata()
                    nifti_cut = nib.Nifti1Image(nifti_long_arr[:,:,:,:expected_trs], 
                                                affine=nifti_long.affine,
                                                header=nifti_long.header)
                    nib.save(nifti_cut, nifti_path)

                    # Construct new filename with continuous run numbering for this task
                    new_filename = re.sub(r'run-\d+', f'run-{task_run_counters[task]:02d}', file)
                    self.rename(file, nifti_path, 
                                new_filename,
                                os.path.join(self.directory, new_filename))

                    # Increment the run counter for this task
                    task_run_counters[task] += 1
                else:
                    if self.testing_mode:
                        # Rename the NIfTI file instead of deleting it
                        invalid_nifti_path = nifti_path.replace('.nii.gz', '_invalid.nii.gz')
                        os.rename(nifti_path, invalid_nifti_path)
                        print(f"Renamed {file} to {os.path.basename(invalid_nifti_path)} (TRs: {actual_trs}, expected: {expected_trs})")

                        # Rename the corresponding JSON file if it exists
                        json_file = file.replace('.nii.gz', '.json')
                        json_path = os.path.join(self.directory, json_file)
                        if os.path.exists(json_path):
                            invalid_json_path = json_path.replace('.json', '_invalid.json')
                            os.rename(json_path, invalid_json_path)
                            print(f"Renamed {json_file} to {os.path.basename(invalid_json_path)}")
                    else:
                        # Delete the NIfTI file
                        os.remove(nifti_path)
                        print(f"Deleted {file} (TRs: {actual_trs}, expected: {expected_trs})")

                        # Delete the corresponding JSON file if it exists
                        json_file = file.replace('.nii.gz', '.json')
                        json_path = os.path.join(self.directory, json_file)
                        if os.path.exists(json_path):
                            os.remove(json_path)
                            print(f"Deleted {json_file}")
            else:
                print(f"Skipping {file}: Task not found in task_duration dictionary")

def main():
    parser = argparse.ArgumentParser(description='Remove aborted runs and rename files, starting run numbering from 1 for each task.')
    parser.add_argument('--directory', '-d', type=str, required=True,
                        help='Directory containing NIfTI files.')
    parser.add_argument('--testing_mode', '-t', action=argparse.BooleanOptionalAction, default=False, 
                        help='Enable testing mode (files will be renamed instead of deleted).')
    args = parser.parse_args()

    processor = RMAbortedRuns(args.directory, args.testing_mode)
    processor.filter_and_rename_runs()

if __name__ == '__main__':
    main()
