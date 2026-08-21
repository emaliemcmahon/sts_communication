"""Group-level significance testing for the whole-brain searchlight decoding
index.

Mirrors the logic in group_decoding_index.py (test the per-subject
cross-validated decoding index against 0 with a sign-flip permutation test)
but applied per-voxel across the whole brain instead of per-ROI, following
the same non_parametric_inference-based approach as
video_sentence_analysis/second_level/group_searchlight_decoding.py. Unlike
that script's decoding-accuracy maps, searchlight_decoding_index.py's maps
are already centered on chance (0, not 0.5), so no centering step is needed
here before the one-sample test.

For each hemisphere-agnostic whole-brain run:
  1. Load each subject's first-level decoding-index map (SearchlightDecodingIndex).
  2. Build a group mask (voxels finite in every included subject).
  3. Run `non_parametric_inference` (one-sided, sign-flip permutation,
     cluster-mass FWE correction), with smoothing applied internally.
  4. Save all returned stat maps, plus a surface plot of the group mean
     decoding index (each subject's map smoothed the same way as the test,
     then averaged) masked down to only the cluster-mass FWE-significant
     voxels.

``--roi_mask sts`` restricts the search volume (and hence the FWE
correction) to the union of the left/right anatomical STS parcels
(Deen et al., 2015; `derivatives/parcels-<space_label>/{l,r}anatSTS.nii.gz`,
same atlas used elsewhere in this project, e.g. group_parcel_probability.py).
These are single group-level atlas files already in the same voxel grid as
the subject decoding-index maps (both `MNI152NLin2009cAsym` by default), so
no resampling is needed. Restricting the search volume this way is a
standard small-volume-correction move: FWE correction is inherently
volume-dependent, so testing only within an a priori region of interest
gives voxels there a real chance to survive correction that whole-brain
correction may wash out, at the cost of no longer testing (or being able to
claim significance) anywhere outside that region.
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import resample_to_img, smooth_img
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import vol_to_surf

ROI_MASKS = {'none': None, 'sts': ['lanatSTS', 'ranatSTS']}


class GroupSearchlightDecodingIndex:
    def __init__(self, args):
        self.process = 'GroupSearchlightDecodingIndex'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.individual_path = f'{self.derivatives_path}/SearchlightDecodingIndex'
        self.parcel_path = f'{self.derivatives_path}/parcels-{args.space_label}'
        self.overwrite = args.overwrite
        self.roi_mask = args.roi_mask
        out_dir = f'{self.process}_{args.out_tag}' if args.out_tag else self.process
        self.out_path = f'{self.derivatives_path}/{out_dir}'
        self.out_stem = 'group_decoding_index' if self.roi_mask == 'none' else f'group_decoding_index_roi-{self.roi_mask}'
        self.sub_nums = args.sub_nums
        self.subjs = [f'sub-{str(i).zfill(2)}' for i in self.sub_nums]
        self.smoothing_fwhm = args.smoothing_fwhm
        self.n_perm = args.n_perm
        self.cluster_forming_threshold = args.cluster_forming_threshold
        self.alpha = args.alpha
        self.n_jobs = args.n_jobs if args.n_jobs is not None else os.cpu_count()
        self.fsaverage_mesh = 'fsaverage6'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
        print(vars(self))

    # ---------- I/O ----------
    def _subject_file(self, sub: str) -> str:
        return f'{self.individual_path}/{sub}/{sub}_decoding_index.nii.gz'

    def _load_subject_maps(self):
        imgs, missing = [], []
        for sub in self.subjs:
            f = self._subject_file(sub)
            if not os.path.exists(f):
                missing.append(sub)
                continue
            imgs.append(nib.load(f))
        if missing:
            print(f'  Warning: missing decoding index map for {missing}')
        return imgs

    @staticmethod
    def _group_mask(imgs) -> nib.Nifti1Image:
        """Voxels finite (i.e. within the searchlight-covered brain) in
        every included subject's map."""
        data = np.stack([img.get_fdata() for img in imgs], axis=0)
        finite = np.isfinite(data).all(axis=0)
        return nib.Nifti1Image(finite.astype(np.int32), imgs[0].affine)

    def _roi_mask_img(self, ref_img: nib.Nifti1Image):
        """Union of the atlas parcel(s) named in ROI_MASKS[self.roi_mask]
        (e.g. left + right anatomical STS), resampled onto ``ref_img``'s
        grid if needed (a no-op when, as by default, both are
        MNI152NLin2009cAsym at 2mm)."""
        names = ROI_MASKS[self.roi_mask]
        if names is None:
            return None
        parcel_imgs = []
        for name in names:
            f = f'{self.parcel_path}/{name}.nii.gz'
            if not os.path.exists(f):
                raise FileNotFoundError(f'Missing ROI parcel: {f}')
            img = nib.load(f)
            if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
                img = resample_to_img(img, ref_img, interpolation='nearest')
            parcel_imgs.append(img)
        return intersect_masks(parcel_imgs, threshold=0, connected=False)

    # ---------- Plotting ----------
    def _plot_surfs(self, img: nib.Nifti1Image, title: str, outbase: Path,
                    vmin: float, vmax: float) -> None:
        surf = fetch_surf_fsaverage(self.fsaverage_mesh)
        for hemi in ['left', 'right']:
            stat = vol_to_surf(img, surf[f'pial_{hemi}'], interpolation='linear',
                               inner_mesh=surf[f'white_{hemi}'])
            for view in ['lateral', 'medial', 'ventral']:
                # threshold=None: significance is already encoded by masking
                # non-significant voxels to NaN before this is called, so no
                # additional colormap threshold is applied here.
                plot_surf_stat_map(surf[f'infl_{hemi}'], stat_map=stat, bg_map=surf[f'sulc_{hemi}'],
                                   hemi=hemi, view=view, bg_on_data=True, cmap='coolwarm',
                                   threshold=None, vmin=vmin, vmax=vmax,
                                   darkness=None, title=title)
                plt.savefig(f'{outbase}_hemi-{hemi}_view-{view}.png', dpi=150)
                plt.close()

    # ---------- Group test ----------
    def run(self) -> None:
        t_start = time.time()
        outbase = Path(self.out_path) / self.out_stem
        done_marker = f'{outbase}_stat-mass.nii.gz'
        if not self.overwrite and Path(done_marker).exists():
            print(f'Outputs already exist at {outbase}; skipping (use --overwrite).')
            return

        print(f'Loading decoding index maps for {len(self.subjs)} subjects ...')
        imgs = self._load_subject_maps()
        if len(imgs) < 2:
            print('Fewer than 2 subjects available; aborting.')
            return

        mask_img = self._group_mask(imgs)
        n_vox = int(mask_img.get_fdata().sum())
        print(f'{len(imgs)} subjects, group mask has {n_vox} voxels')

        if self.roi_mask != 'none':
            roi_img = self._roi_mask_img(mask_img)
            mask_img = intersect_masks([mask_img, roi_img], threshold=1, connected=False)
            n_vox = int(mask_img.get_fdata().sum())
            print(f'Restricted to ROI mask "{self.roi_mask}": group mask now has {n_vox} voxels')

        # Values outside the mask are irrelevant (non_parametric_inference
        # restricts computation to `mask_img`), so nan_to_num here is only
        # to keep the image itself free of NaNs. Unlike decoding-accuracy
        # maps, the decoding index is already centered on chance (0), so no
        # additional centering is applied before the one-sample test.
        centered_imgs = [nib.Nifti1Image(np.nan_to_num(img.get_fdata()), img.affine)
                         for img in imgs]
        design_matrix = pd.DataFrame({'intercept': np.ones(len(centered_imgs))})

        print(f'Running non_parametric_inference (n_perm={self.n_perm}, '
             f'smoothing_fwhm={self.smoothing_fwhm}mm, cluster-forming threshold '
             f'p<{self.cluster_forming_threshold}, one-sided, n_jobs={self.n_jobs}) ...')
        t_perm = time.time()
        outputs = non_parametric_inference(
            centered_imgs,
            design_matrix=design_matrix,
            mask=mask_img,
            smoothing_fwhm=self.smoothing_fwhm,
            n_perm=self.n_perm,
            two_sided_test=False,
            threshold=self.cluster_forming_threshold,
            tfce=False,
            n_jobs=self.n_jobs,
            random_state=0,
            verbose=1,
        )
        print(f'Permutation test finished in {(time.time() - t_perm) / 60:.1f} min')

        for key, img in outputs.items():
            nib.save(img, f'{outbase}_stat-{key}.nii.gz')

        # Group mean decoding index, smoothed per-subject the same way as the
        # test (so it reflects what was actually tested), masked down to
        # only the cluster-mass FWE-significant voxels. This is what gets
        # plotted -- the significance (logp_max_mass) values themselves are
        # saved above but not shown on the surface.
        smoothed_imgs = smooth_img(imgs, fwhm=self.smoothing_fwhm)
        mean_index = np.mean([img.get_fdata() for img in smoothed_imgs], axis=0)
        logp_mass = outputs['logp_max_mass'].get_fdata()
        sig = logp_mass > -np.log10(self.alpha)
        n_sig = int(sig.sum())
        print(f'{n_sig} voxels significant at cluster-mass FWE p<{self.alpha}')

        thresholded_index = np.where(sig, mean_index, np.nan)
        thresholded_img = nib.Nifti1Image(thresholded_index, mask_img.affine)
        nib.save(thresholded_img, f'{outbase}_stat-meanindex-sig.nii.gz')

        if n_sig > 0:
            vmax = max(float(np.nanmax(np.abs(thresholded_index))), 0.01)
            roi_suffix = '' if self.roi_mask == 'none' else f', ROI={self.roi_mask}'
            title = f'Decoding index (cluster-mass FWE p<{self.alpha}{roi_suffix})'
            print('Plotting surfaces ...')
            self._plot_surfs(thresholded_img, title, outbase, vmin=-vmax, vmax=vmax)
        else:
            print('No significant voxels; skipping surface plots.')

        print(f'Done in {(time.time() - t_start) / 60:.1f} min total')


def parse_args():
    p = argparse.ArgumentParser(description='Group-level significance testing for the '
                                            'whole-brain searchlight decoding index '
                                            '(sign-flip permutation vs. 0, cluster-mass '
                                            'FWE correction).')
    p.add_argument('sub_nums', nargs='*', type=int,
                   help='List of subject numbers',
                   default=[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23])
    p.add_argument('--dataset_path', '-d', type=str,
                   default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    p.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym',
                   help='Space of the first-level decoding index maps and of the ROI '
                        'parcel files under derivatives/parcels-<space_label>/.')
    p.add_argument('--roi_mask', type=str, default='none', choices=list(ROI_MASKS.keys()),
                   help='Restrict the group search volume (and hence the FWE correction) '
                        'to this ROI instead of the whole brain. "sts" uses the union of '
                        'the left/right anatomical STS parcels (small-volume correction).')
    p.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--smoothing_fwhm', type=float, default=6.0,
                   help='FWHM (mm) of Gaussian smoothing applied to each subject\'s '
                        'decoding index map before the group test.')
    p.add_argument('--n_perm', type=int, default=10000,
                   help='Sign-flip permutations for the group null distribution.')
    p.add_argument('--cluster_forming_threshold', type=float, default=0.001,
                   help='Voxel-level p-value threshold used to define clusters for '
                        'cluster-mass FWE correction.')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Cluster-mass FWE-corrected significance level used only for '
                        'the surface plot threshold; all unthresholded stat maps are '
                        'saved regardless.')
    p.add_argument('--n_jobs', type=int, default=None,
                   help='Parallel workers for the permutation procedure (default: all cores).')
    p.add_argument('--out_tag', type=str, default='',
                   help='If set, write to derivatives/GroupSearchlightDecodingIndex_<out_tag>/ '
                        'instead of derivatives/GroupSearchlightDecodingIndex/, so a run on a '
                        'subject subset does not overwrite the full-sample results.')
    return p.parse_args()


def main():
    args = parse_args()
    GroupSearchlightDecodingIndex(args).run()


if __name__ == '__main__':
    main()
