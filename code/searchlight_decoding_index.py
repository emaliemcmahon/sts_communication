import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from joblib import Parallel, cpu_count, delayed
from nibabel.affines import voxel_sizes
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import vol_to_surf
from rsatoolbox.util.searchlight import get_volume_searchlight

from decoding_index import CONDITIONS, FOLD_RUNS
from utils.mri import load_brain_mask


def _cv_corr(patterns_a, patterns_b, idx, cond1, cond2):
    """Cross-fold correlation between two conditions, averaged over both fold
    directions (see decoding_index.py's DecodingIndex.cv_corr)."""
    r1 = np.corrcoef(patterns_a[cond1][idx], patterns_b[cond2][idx])[0, 1]
    r2 = np.corrcoef(patterns_b[cond1][idx], patterns_a[cond2][idx])[0, 1]
    return 0.5 * (r1 + r2)


def _decoding_index_sphere(neighbor_idx, mask_flat, patterns_a, patterns_b, min_voxels=2):
    """Cross-validated Haxby-style within/between decoding index (see
    decoding_index.py) for one searchlight sphere: ``patterns_a``/
    ``patterns_b`` are whole-brain condition patterns averaged within fold A
    (runs 1-5) / fold B (runs 6-9) respectively, and every term is computed
    *across* folds to avoid the GLM design-collinearity bias described there.

    ``neighbor_idx`` are flat voxel indices for the full geometric sphere
    (rsatoolbox returns these without regard to the brain mask), so they are
    first restricted to voxels actually inside ``mask_flat``.

    decoding index = 0.5 * (r_w1 + r_w2 - r_b1 - r_b2)
        r_w1 = corr(com_ind, face_first)   [within communicate, dyad vs. face]
        r_w2 = corr(ind, face_noncom)      [within independent, dyad vs. face]
        r_b1 = corr(com_ind, face_noncom)  [between: communicate-dyad vs. independent-face]
        r_b2 = corr(face_first, ind)       [between: communicate-face vs. independent-dyad]
    """
    idx = np.asarray(neighbor_idx)
    idx = idx[mask_flat[idx]]
    if idx.size < min_voxels:
        return np.nan

    r_w1 = _cv_corr(patterns_a, patterns_b, idx, 'com_ind', 'face_first')
    r_w2 = _cv_corr(patterns_a, patterns_b, idx, 'ind', 'face_noncom')
    r_b1 = _cv_corr(patterns_a, patterns_b, idx, 'com_ind', 'face_noncom')
    r_b2 = _cv_corr(patterns_a, patterns_b, idx, 'face_first', 'ind')
    return 0.5 * (r_w1 + r_w2 - r_b1 - r_b2)


class SearchlightDecodingIndex:
    """
    Whole-brain searchlight version of the cross-validated Haxby-style
    within/between decoding index computed per-ROI in decoding_index.py: at
    every searchlight sphere, tests whether communicative (com_ind,
    face_first) and independent (ind, face_noncom) patterns are
    distinguishable in a way that generalizes across agent count (dyad vs.
    single face).

    As in decoding_index.py, each pairwise pattern correlation is computed
    across two disjoint sets of runs (fold A = runs 1-5, fold B = runs 6-9;
    NilearnGLMRunwise per-run betas) rather than from a single all-runs-pooled
    GLM, which removes a GLM design-collinearity bias that otherwise inflates
    same-fold pattern correlations independent of any true signal (see
    methods.md).

    Searchlight neighborhoods and whole-brain nifti/surface output follow the
    same setup as video_sentence_analysis's searchlight_decoding.py
    (rsatoolbox ``get_volume_searchlight``). rsatoolbox's searchlight radius
    is in voxel units, so ``--radius`` (given in mm) is converted using this
    subject's (anisotropic) voxel size before being passed in.
    """

    def __init__(self, args):
        self.process = 'SearchlightDecodingIndex'
        self.dataset_path = args.dataset_path
        self.derivatives_path = os.path.join(self.dataset_path, 'derivatives')
        self.fmriprep_path = os.path.join(self.derivatives_path, 'fmriprep')
        self.runwise_glm_path = os.path.join(self.derivatives_path, 'NilearnGLMRunwise', f'sub-{args.subject}')
        self.subject = args.subject
        self.task_label = 'communicate'
        self.space_label = args.space_label
        self.radius_mm = args.radius
        self.threshold = args.threshold
        self.n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
        self.overwrite = args.overwrite
        self.out_path = os.path.join(self.derivatives_path, self.process, f'sub-{self.subject}')
        self.out_base = os.path.join(self.out_path, f'sub-{self.subject}_decoding_index')
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

    def _save_img(self, arr, affine, outbase):
        nib.save(nib.Nifti1Image(arr, affine=affine), f'{outbase}.nii.gz')

    def _plot_surfs(self, arr, affine, outbase):
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
                                   title=f'sub-{self.subject} decoding index ({hemi} {view})')
                plt.savefig(f'{outbase}_hemi-{hemi}_view-{view}.png', dpi=150)
                plt.close()

    def run(self):
        if not self.overwrite and os.path.exists(f'{self.out_base}.nii.gz'):
            print(f'{self.out_base}.nii.gz already exists; skipping (use --overwrite).')
            return

        mask_img = load_brain_mask(self.fmriprep_path, self.subject, self.task_label, self.space_label)
        affine = mask_img.affine
        vol_shape = mask_img.shape
        mask_bool = mask_img.get_fdata().astype(bool)
        mask_flat = mask_bool.flatten()

        patterns_a, patterns_b = self.load_patterns(affine, vol_shape)

        radius_voxels = self.radius_mm / np.mean(voxel_sizes(affine))
        print(f'[Searchlight] radius = {self.radius_mm} mm ~= {radius_voxels:.2f} voxels; '
              f'computing neighborhoods ...')
        centers, neighbors = get_volume_searchlight(mask_bool, radius=radius_voxels,
                                                     threshold=self.threshold)
        print(f'[Searchlight] {len(centers)} centers.')

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_decoding_index_sphere)(neighbor_idx, mask_flat, patterns_a, patterns_b)
            for neighbor_idx in neighbors)

        n_vox = int(np.prod(vol_shape))
        vol = np.full(n_vox, np.nan)
        vol[centers] = np.array(results)
        vol = vol.reshape(vol_shape)

        self._save_img(vol, affine, self.out_base)
        self._plot_surfs(vol, affine, self.out_base)
        print(f'Saved {self.out_base}.nii.gz')


def main():
    parser = argparse.ArgumentParser(
        description='Whole-brain searchlight version of the cross-validated Haxby-style '
                    'decoding index (com_ind/ind/face_first/face_noncom), per subject.')
    parser.add_argument('--dataset_path', '-d', type=str,
                        default='/orcd/data/ngk/001/users/emaliem/sts_communication')
    parser.add_argument('--subject', '-s', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--space_label', '-sp', type=str, default='MNI152NLin2009cAsym')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Searchlight radius in mm (converted to voxels internally).')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum proportion of the geometric sphere that must fall '
                             'inside the brain mask for a center to be kept.')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Parallel workers across searchlight spheres (default: all cores).')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    SearchlightDecodingIndex(args).run()


if __name__ == '__main__':
    main()
