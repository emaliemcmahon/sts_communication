"""Group-level significance testing for the whole-brain (by default,
STS-restricted) searchlight within-condition-pair decoding index.

Mirrors group_searchlight_decoding_index.py's non_parametric_inference
approach, but run separately per pair (dyad: com_ind vs. ind; face:
face_first vs. face_noncom; see within_decoding_index.py /
searchlight_within_decoding_index.py), loading each subject's
`sub-<subject>_within_decoding_index_pair-<pair>.nii.gz` map. The first-level
maps are already restricted to the STS by default
(searchlight_within_decoding_index.py's `--roi_mask sts`), so `--roi_mask`
here defaults to 'none' (a no-op intersection against an already-STS-only
group mask) but is kept for flexibility if first-level was run whole-brain.
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

from within_decoding_index import PAIRS

ROI_MASKS = {'none': None, 'sts': ['lanatSTS', 'ranatSTS']}


class GroupSearchlightWithinDecodingIndex:
    def __init__(self, args):
        self.process = 'GroupSearchlightWithinDecodingIndex'
        self.dataset_path = args.dataset_path
        self.derivatives_path = f'{self.dataset_path}/derivatives'
        self.individual_path = f'{self.derivatives_path}/SearchlightWithinDecodingIndex'
        self.parcel_path = f'{self.derivatives_path}/parcels-{args.space_label}'
        self.overwrite = args.overwrite
        self.roi_mask = args.roi_mask
        out_dir = f'{self.process}_{args.out_tag}' if args.out_tag else self.process
        self.out_path = f'{self.derivatives_path}/{out_dir}'
        self.sub_nums = args.sub_nums
        self.subjs = [f'sub-{str(i).zfill(2)}' for i in self.sub_nums]
        self.pairs = [p[0] for p in PAIRS]
        self.smoothing_fwhm = args.smoothing_fwhm
        self.n_perm = args.n_perm
        self.cluster_forming_threshold = args.cluster_forming_threshold
        self.alpha = args.alpha
        self.surf_search_radius_mm = args.surf_search_radius_mm
        self.n_jobs = args.n_jobs if args.n_jobs is not None else os.cpu_count()
        self.fsaverage_mesh = 'fsaverage6'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
        print(vars(self))

    # ---------- I/O ----------
    def _subject_file(self, sub: str, pair: str) -> str:
        return f'{self.individual_path}/{sub}/{sub}_within_decoding_index_pair-{pair}.nii.gz'

    def _load_subject_maps(self, pair: str):
        imgs, missing = [], []
        for sub in self.subjs:
            f = self._subject_file(sub, pair)
            if not os.path.exists(f):
                missing.append(sub)
                continue
            imgs.append(nib.load(f))
        if missing:
            print(f'  Warning: missing pair={pair} map for {missing}')
        return imgs

    @staticmethod
    def _group_mask(imgs) -> nib.Nifti1Image:
        """Voxels finite (i.e. within the searchlight-covered brain) in
        every included subject's map."""
        data = np.stack([img.get_fdata() for img in imgs], axis=0)
        finite = np.isfinite(data).all(axis=0)
        return nib.Nifti1Image(finite.astype(np.int32), imgs[0].affine)

    def _roi_mask_img(self, ref_img: nib.Nifti1Image):
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
    def _plot_surfs(self, mean_img: nib.Nifti1Image, sig_mask_img: nib.Nifti1Image,
                    title: str, outbase: Path, vmin: float, vmax: float) -> None:
        """See group_searchlight_decoding_index.py's _plot_surfs for why
        significance coloring comes from a separate ball-sampled projection
        of the binary significance mask rather than the same per-vertex
        line sample used for the plotted value -- otherwise a small cluster
        can be entirely invisible."""
        surf = fetch_surf_fsaverage(self.fsaverage_mesh)
        for hemi in ['left', 'right']:
            stat = vol_to_surf(mean_img, surf[f'pial_{hemi}'], interpolation='linear',
                               inner_mesh=surf[f'white_{hemi}'])
            sig_coverage = vol_to_surf(sig_mask_img, surf[f'pial_{hemi}'], kind='ball',
                                       radius=self.surf_search_radius_mm,
                                       interpolation='nearest', n_samples=20)
            vertex_sig = np.nan_to_num(sig_coverage) > 0
            stat = np.where(vertex_sig, stat, np.nan)
            for view in ['lateral', 'medial', 'ventral']:
                plot_surf_stat_map(surf[f'infl_{hemi}'], stat_map=stat, bg_map=surf[f'sulc_{hemi}'],
                                   hemi=hemi, view=view, bg_on_data=True, cmap='coolwarm',
                                   threshold=None, vmin=vmin, vmax=vmax,
                                   darkness=None, title=title)
                plt.savefig(f'{outbase}_hemi-{hemi}_view-{view}.png', dpi=150)
                plt.close()

    # ---------- Per pair ----------
    def run_one(self, pair: str) -> None:
        t_start = time.time()
        stem = f'group_within_decoding_index_pair-{pair}' if self.roi_mask == 'none' \
            else f'group_within_decoding_index_pair-{pair}_roi-{self.roi_mask}'
        outbase = Path(self.out_path) / stem
        done_marker = f'{outbase}_stat-mass.nii.gz'
        if not self.overwrite and Path(done_marker).exists():
            print(f'[{pair}] outputs already exist at {outbase}; skipping (use --overwrite).')
            return

        print(f'[{pair}] loading within decoding index maps for {len(self.subjs)} subjects ...')
        imgs = self._load_subject_maps(pair)
        if len(imgs) < 2:
            print(f'[{pair}] fewer than 2 subjects available; skipping.')
            return

        mask_img = self._group_mask(imgs)
        n_vox = int(mask_img.get_fdata().sum())
        print(f'[{pair}] {len(imgs)} subjects, group mask has {n_vox} voxels')

        if self.roi_mask != 'none':
            roi_img = self._roi_mask_img(mask_img)
            mask_img = intersect_masks([mask_img, roi_img], threshold=1, connected=False)
            n_vox = int(mask_img.get_fdata().sum())
            print(f'[{pair}] restricted to ROI mask "{self.roi_mask}": group mask now has {n_vox} voxels')

        centered_imgs = [nib.Nifti1Image(np.nan_to_num(img.get_fdata()), img.affine)
                         for img in imgs]
        design_matrix = pd.DataFrame({'intercept': np.ones(len(centered_imgs))})

        print(f'[{pair}] running non_parametric_inference (n_perm={self.n_perm}, '
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
        print(f'[{pair}] permutation test finished in {(time.time() - t_perm) / 60:.1f} min')

        for key, img in outputs.items():
            nib.save(img, f'{outbase}_stat-{key}.nii.gz')

        smoothed_imgs = smooth_img(imgs, fwhm=self.smoothing_fwhm)
        mean_index = np.mean([img.get_fdata() for img in smoothed_imgs], axis=0)
        logp_mass = outputs['logp_max_mass'].get_fdata()
        sig = logp_mass > -np.log10(self.alpha)
        n_sig = int(sig.sum())
        print(f'[{pair}] {n_sig} voxels significant at cluster-mass FWE p<{self.alpha}')

        thresholded_index = np.where(sig, mean_index, np.nan)
        thresholded_img = nib.Nifti1Image(thresholded_index, mask_img.affine)
        nib.save(thresholded_img, f'{outbase}_stat-meanindex-sig.nii.gz')

        if n_sig > 0:
            vmax = max(float(np.nanmax(np.abs(thresholded_index))), 0.01)
            roi_suffix = '' if self.roi_mask == 'none' else f', ROI={self.roi_mask}'
            title = f'Within decoding index [{pair}] (cluster-mass FWE p<{self.alpha}{roi_suffix})'
            print(f'[{pair}] plotting surfaces ...')
            mean_img = nib.Nifti1Image(mean_index, mask_img.affine)
            sig_img = nib.Nifti1Image(sig.astype(np.int32), mask_img.affine)
            self._plot_surfs(mean_img, sig_img, title, outbase, vmin=-vmax, vmax=vmax)
        else:
            print(f'[{pair}] no significant voxels; skipping surface plots.')

        print(f'[{pair}] done in {(time.time() - t_start) / 60:.1f} min total')

    def run(self) -> None:
        t0 = time.time()
        for i, pair in enumerate(self.pairs, 1):
            print(f'--- [{i}/{len(self.pairs)}] pair={pair} ---')
            self.run_one(pair)
        print(f'All pairs finished in {(time.time() - t0) / 60:.1f} min')


def parse_args():
    p = argparse.ArgumentParser(description='Group-level significance testing for the '
                                            'whole-brain (or STS-restricted) searchlight '
                                            'within-condition-pair decoding index.')
    p.add_argument('sub_nums', nargs='*', type=int,
                   help='List of subject numbers',
                   default=[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23])
    p.add_argument('--dataset_path', '-d', type=str,
                   default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    p.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym',
                   help='Space of the first-level maps and of the ROI parcel files under '
                        'derivatives/parcels-<space_label>/.')
    p.add_argument('--roi_mask', type=str, default='none', choices=list(ROI_MASKS.keys()),
                   help='Additionally restrict the group search volume to this ROI. '
                        'Default "none": the first-level maps are already STS-restricted '
                        'by default (searchlight_within_decoding_index.py), so this is a '
                        'no-op unless first-level was run with --roi_mask none.')
    p.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--smoothing_fwhm', type=float, default=6.0,
                   help='FWHM (mm) of Gaussian smoothing applied to each subject\'s '
                        'within decoding index map before the group test.')
    p.add_argument('--n_perm', type=int, default=10000,
                   help='Sign-flip permutations for the group null distribution.')
    p.add_argument('--cluster_forming_threshold', type=float, default=0.001,
                   help='Voxel-level p-value threshold used to define clusters for '
                        'cluster-mass FWE correction.')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Cluster-mass FWE-corrected significance level used only for '
                        'the surface plot threshold; all unthresholded stat maps are '
                        'saved regardless.')
    p.add_argument('--surf_search_radius_mm', type=float, default=4.0,
                   help='Radius (mm) of the spherical neighborhood sampled around each '
                        'surface vertex to decide whether it is "covered" by a '
                        'significant voxel for plotting (rendering only).')
    p.add_argument('--n_jobs', type=int, default=None,
                   help='Parallel workers for the permutation procedure (default: all cores).')
    p.add_argument('--out_tag', type=str, default='',
                   help='If set, write to derivatives/GroupSearchlightWithinDecodingIndex_<out_tag>/ '
                        'instead of derivatives/GroupSearchlightWithinDecodingIndex/, so a run on '
                        'a subject subset does not overwrite the full-sample results.')
    return p.parse_args()


def main():
    args = parse_args()
    GroupSearchlightWithinDecodingIndex(args).run()


if __name__ == '__main__':
    main()
