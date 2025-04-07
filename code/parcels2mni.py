from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn.image import resample_img

out_path = 'derivatives/parcels-MNI152NLin2009cAsym'
Path(out_path).mkdir(exist_ok=True, parents=True)

epi_path = 'derivatives/fmriprep/sub-CP01/ses-01/func/'
epi_file = 'sub-CP01_ses-01_task-communicate_run-08_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
epi = nib.load(f'{epi_path}/{epi_file}')
dim = epi.shape[:-1]

roi2out = {'FFA': 'FFA',
           'STS': 'fSTS',
           'EBA': 'EBA'}
# DYLOC ROIs
dyloc_path = 'derivatives/raw-parcels/lab-parcels/mni_parcels/dyloc'
for roi in ['FFA', 'STS', 'EBA']: 
    for hemi in ['l', 'r']:
        out_parc = np.zeros(dim)
        parc = nib.load(f'{dyloc_path}/{hemi}{roi}.nii.gz')
        resampled_parc = resample_img(parc, epi.affine, dim,
                                    interpolation='nearest')
        resampled_parc = resampled_parc.get_fdata().astype(bool).squeeze()
        out_parc[resampled_parc] = 1
        out_parc = nib.Nifti1Image(out_parc, epi.affine, 
                                nib.Nifti1Header())
        nib.save(out_parc, f'{out_path}/{hemi}{roi2out[roi]}.nii.gz')

# ToM ROIs
tom_path = 'derivatives/raw-parcels/lab-parcels/mni_parcels/tom'
for roi in ['TPJ']: 
    for hemi in ['l', 'r']:
        out_parc = np.zeros(dim)
        parc = nib.load(f'{tom_path}/{hemi}{roi}.nii.gz')
        resampled_parc = resample_img(parc, epi.affine, dim,
                                    interpolation='nearest')
        resampled_parc = resampled_parc.get_fdata().astype(bool).squeeze()
        out_parc[resampled_parc] = 1
        out_parc = nib.Nifti1Image(out_parc, epi.affine, 
                                nib.Nifti1Header())
        nib.save(out_parc, f'{out_path}/{hemi}{roi}.nii.gz')

# EVC and MT
roi2num = {'EVC': ['1', '2', '3', '4'],
           'MT': ['13']
           }
kastner_path = 'derivatives/raw-parcels/SabineKastner/subj_vol_all'
for roi in roi2num.keys(): 
    for hemi in ['l', 'r']:
        out_parc = np.zeros(dim)
        for roi_num in roi2num[roi]:
            parc = nib.load(f'{kastner_path}/perc_VTPM_vol_roi{roi_num}_{hemi}h.nii.gz')
            resampled_parc = resample_img(parc, epi.affine, dim,
                                        interpolation='nearest')
            resampled_parc = resampled_parc.get_fdata().astype(bool).squeeze()
            out_parc[resampled_parc] = 1
        out_parc = nib.Nifti1Image(out_parc, epi.affine, 
                                nib.Nifti1Header())
        nib.save(out_parc, f'{out_path}/{hemi}{roi}.nii.gz')

# EVC and MT
deen_path = 'derivatives/raw-parcels/BenDeen_fROIs'
roi2out = {'STS': 'anatSTS'}
for roi in ['STS']: 
    for hemi in ['l', 'r']:
        out_parc = np.zeros(dim)
        parc = nib.load(f'{deen_path}/{hemi}{roi}.nii.gz')
        resampled_parc = resample_img(parc, epi.affine, dim,
                                    interpolation='nearest')
        resampled_parc = resampled_parc.get_fdata().astype(bool).squeeze()
        out_parc[resampled_parc] = 1
        out_parc = nib.Nifti1Image(out_parc, epi.affine, 
                                nib.Nifti1Header())
        nib.save(out_parc, f'{out_path}/{hemi}{roi2out[roi]}.nii.gz')