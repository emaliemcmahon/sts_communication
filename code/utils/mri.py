from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import SurfaceImage



def vol2surf(stat_img):
    fsaverage_meshes = load_fsaverage(mesh="fsaverage")
    fsaverage_sulcal = load_fsaverage_data(
        mesh="fsaverage",
        data_type="sulcal",
        mesh_type="inflated",
    )

    img = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=stat_img,
    )
    return img, fsaverage_meshes, fsaverage_sulcal


def check_motion_filtering(sample_masks, n_trs, threshold=12, one_indexed=False):
    """
    Return a list of runs where more than `threshold` frames were removed.

    Parameters
    ----------
    sample_masks : array-like or list of array-like
        Each element is a numpy array of kept-volume indices (from load_confounds).
        If a single run, can pass a single array directly.
    n_trs : int or list of int
        Total number of volumes (TRs) in each corresponding run.
    threshold : int, default=12
        Number of removed frames above which a run is considered exceeding.
    one_indexed : bool, default=True
        If True, return runs numbered from 1 (run 1, 2, ...). 
        If False, return 0-indexed run indices.

    Returns
    -------
    list of int
        Runs that exceed the threshold for removed frames.
    list of int
        Runs that do not exceed the threshold for removed frames.
    """
    import numpy as np

    # Normalize to lists
    if not isinstance(sample_masks, (list, tuple)):
        sample_masks = [sample_masks]
    if not isinstance(n_trs, (list, tuple, np.ndarray)):
        n_trs = [n_trs] * len(sample_masks)

    bad_runs = []
    good_runs = []
    for i, (mask, total) in enumerate(zip(sample_masks, n_trs)):
        n_kept = len(mask) if mask is not None else total
        n_removed = total - n_kept
        if n_removed > threshold:
            bad_runs.append(i + 1 if one_indexed else i)
        else:
            good_runs.append(i + 1 if one_indexed else i)   

    return bad_runs, good_runs