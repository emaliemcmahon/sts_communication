import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from joblib import Parallel, cpu_count, delayed
from nibabel.affines import voxel_sizes
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.image import resample_to_img
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import vol_to_surf
from rsatoolbox.util.searchlight import get_volume_searchlight

from decoding_index import CONDITIONS, FOLD_RUNS
from searchlight_decoding_index import _cv_corr
from utils.mri import load_brain_mask
from within_decoding_index import PAIRS

ROI_MASKS = {'none': None, 'sts': ['lanatSTS', 'ranatSTS']}


def _within_decoding_index_sphere(neighbor_idx, mask_flat, patterns_a, patterns_b, cond1, cond2, min_voxels=2):
    """Cross-validated within-condition-pair decoding index (see
    within_decoding_index.py) for one searchlight sphere: same fold-A/fold-B
    patterns as searchlight_decoding_index.py's cross-pair index, but tests
    only `cond1` vs. `cond2` (e.g. com_ind vs. ind) without requiring
    generalization to the other agent-count pair.

    within decoding index = 0.5 * (r_w1 + r_w2) - r_b
        r_w1 = corr(cond1_A, cond1_B)   [split-half reliability of cond1]
        r_w2 = corr(cond2_A, cond2_B)   [split-half reliability of cond2]
        r_b  = cv_corr(cond1, cond2)    [between, cross-fold]

    `_cv_corr(patterns_a, patterns_b, idx, X, X)` collapses to the ordinary
    split-half reliability `corr(X_A, X_B)`, since correlation is symmetric.
    """
    idx = np.asarray(neighbor_idx)
    idx = idx[mask_flat[idx]]
    if idx.size < min_voxels:
        return np.nan

    r_w1 = _cv_corr(patterns_a, patterns_b, idx, cond1, cond1)
    r_w2 = _cv_corr(patterns_a, patterns_b, idx, cond2, cond2)
    r_b = _cv_corr(patterns_a, patterns_b, idx, cond1, cond2)
    return 0.5 * (r_w1 + r_w2) - r_b


class SearchlightWithinDecodingIndex:
    """
    Whole-brain (by default, STS-restricted) searchlight version of the
    within-condition-pair decoding index computed per-ROI in
    within_decoding_index.py: at every searchlight sphere, tests whether
    com_ind is distinguishable from ind (pair='dyad'), and separately
    whether face_first is distinguishable from face_noncom (pair='face'),
    without requiring generalization across agent count.

    `--roi_mask sts` (the default) restricts the searchlight itself -- not
    just the group-level test -- to the union of the left/right anatomical
    STS parcels, since the whole-brain cross-pair searchlight
    (searchlight_decoding_index.py) only reached significance there under
    small-volume correction (see group_searchlight_decoding_index.py); this
    also cuts first-level compute by roughly the same ~20x the whole-brain
    mask is larger than the STS parcel.

    Otherwise follows the same searchlight setup as
    searchlight_decoding_index.py / video_sentence_analysis's
    first_level_multivariate/searchlight_decoding.py (rsatoolbox
    `get_volume_searchlight`); searchlight radius is in mm and converted to
    voxels using this subject's (anisotropic) voxel size.
    """

    def __init__(self, args):
        self.process = 'SearchlightWithinDecodingIndex'
        self.dataset_path = args.dataset_path
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.fmriprep_path = os.path.join(self.derivatives_path, 'fmriprep')
        self.runwise_glm_path = os.path.join(self.derivatives_path, 'NilearnGLMRunwise', f'sub-{args.subject}')
        self.parcel_path = os.path.join(self.derivatives_path, f'parcels-{args.space_label}')
        self.subject = args.subject
        self.task_label = 'communicate'
        self.space_label = args.space_label
        self.radius_mm = args.radius
        self.threshold = args.threshold
        self.roi_mask = args.roi_mask
        self.n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
        self.overwrite = args.overwrite
        self.out_path = os.path.join(self.derivatives_path, self.process, f'sub-{self.subject}')
        self.out_base = os.path.join(self.out_path, f'sub-{self.subject}_within_decoding_index')
        self.pairs = PAIRS
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def load_fold_pattern(self, condition, fold, affine, shape):
        """Average whole-brain per-run beta pattern across one fold's runs."""
        run_arrays = []
        for run in FOLD_RUNS[fold]:
            beta_file = os.path.join(self.runwise_glm_path,
                                     f'sub-{self.subject}_task-{self.task_label}_'
                                     f'contrast-{condition}_run-{run}.nii.gz')
            if not os.path.exists(beta_file):
                raise FileNotFoundError(f'Missing beta map: {beta_file}')
            img = nib.load(beta_file)
            if img.shape != shape or not np.allclose(img.affine, affine):
                raise ValueError(f'{beta_file} does not match the brain mask geometry')
            run_arrays.append(img.get_fdata().flatten())
        return np.mean(run_arrays, axis=0)

    def load_patterns(self, affine, shape):
        patterns_a = {c: self.load_fold_pattern(c, 'A', affine, shape) for c in CONDITIONS}
        patterns_b = {c: self.load_fold_pattern(c, 'B', affine, shape) for c in CONDITIONS}
        return patterns_a, patterns_b

    def _roi_mask_bool(self, ref_img: nib.Nifti1Image):
        """Union of the atlas parcel(s) named in ROI_MASKS[self.roi_mask]
        (e.g. left + right anatomical STS), resampled onto `ref_img`'s grid
        if needed (a no-op when both are MNI152NLin2009cAsym at 2mm)."""
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
        union = intersect_masks(parcel_imgs, threshold=0, connected=False)
        return union.get_fdata().astype(bool)

    def _save_img(self, arr, affine, outbase):
        nib.save(nib.Nifti1Image(arr, affine=affine), f'{outbase}.nii.gz')

    def _plot_surfs(self, arr, affine, outbase, title_suffix=''):
        img = nib.Nifti1Image(arr, affine=affine)
        surf = fetch_surf_fsaverage('fsaverage6')
        vmax = np.nanpercentile(np.abs(arr), 99) or 1e-6
        for hemi in ['left', 'right']:
            stat = vol_to_surf(img, surf[f'pial_{hemi}'], interpolation='linear',
                               inner_mesh=surf[f'white_{hemi}'])
            for view in ['lateral', 'medial', 'ventral']:
                plot_surf_stat_map(surf[f'infl_{hemi}'], stat_map=stat, bg_map=surf[f'sulc_{hemi}'],
                                   hemi=hemi, view=view, bg_on_data=True, cmap='coolwarm',
                                   vmin=-vmax, vmax=vmax, darkness=None,
                                   title=f'sub-{self.subject} within decoding index{title_suffix} ({hemi} {view})')
                plt.savefig(f'{outbase}_hemi-{hemi}_view-{view}.png', dpi=150)
                plt.close()

    def run(self):
        if not self.overwrite and os.path.exists(f'{self.out_base}_pair-{self.pairs[-1][0]}.nii.gz'):
            print(f'{self.out_base}_pair-*.nii.gz already exist; skipping (use --overwrite).')
            return

        mask_img = load_brain_mask(self.fmriprep_path, self.subject, self.task_label, self.space_label)
        affine = mask_img.affine
        vol_shape = mask_img.shape
        mask_bool = mask_img.get_fdata().astype(bool)

        if self.roi_mask != 'none':
            roi_bool = self._roi_mask_bool(mask_img)
            n_before = int(mask_bool.sum())
            mask_bool = mask_bool & roi_bool
            print(f'[ROI mask] restricted brain mask to "{self.roi_mask}": '
                 f'{n_before} -> {int(mask_bool.sum())} voxels')
        mask_flat = mask_bool.flatten()

        patterns_a, patterns_b = self.load_patterns(affine, vol_shape)

        radius_voxels = self.radius_mm / np.mean(voxel_sizes(affine))
        print(f'[Searchlight] radius = {self.radius_mm} mm ~= {radius_voxels:.2f} voxels; '
              f'computing neighborhoods ...')
        centers, neighbors = get_volume_searchlight(mask_bool, radius=radius_voxels,
                                                     threshold=self.threshold)
        print(f'[Searchlight] {len(centers)} centers.')
        n_vox = int(np.prod(vol_shape))

        for pair_name, cond1, cond2 in self.pairs:
            outbase = f'{self.out_base}_pair-{pair_name}'
            print(f'[{pair_name}] {cond1} vs. {cond2}: computing {len(centers)} spheres ...')
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_within_decoding_index_sphere)(neighbor_idx, mask_flat, patterns_a, patterns_b,
                                                        cond1, cond2)
                for neighbor_idx in neighbors)

            vol = np.full(n_vox, np.nan)
            vol[centers] = np.array(results)
            vol = vol.reshape(vol_shape)

            self._save_img(vol, affine, outbase)
            self._plot_surfs(vol, affine, outbase, title_suffix=f' [{pair_name}]')
            print(f'[{pair_name}] saved {outbase}.nii.gz')


def main():
    parser = argparse.ArgumentParser(
        description='Searchlight version of the within-condition-pair decoding index '
                    '(com_ind vs. ind; face_first vs. face_noncom), per subject.')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym')
    parser.add_argument('--roi_mask', type=str, default='sts', choices=list(ROI_MASKS.keys()),
                        help='Restrict the searchlight itself to this ROI (default: "sts", '
                             'the union of the left/right anatomical STS parcels). "none" runs '
                             'whole-brain instead, like searchlight_decoding_index.py.')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Searchlight radius in mm (converted to voxels internally).')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum proportion of the geometric sphere that must fall '
                             'inside the brain mask for a center to be kept.')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Parallel workers across searchlight spheres (default: all cores).')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    SearchlightWithinDecodingIndex(args).run()


if __name__ == '__main__':
    main()
