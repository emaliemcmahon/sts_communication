import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform


def p2star(p):
    if 0.001 > p:
        star = '***'
    elif 0.01 > p >= 0.001:
        star = '**'
    elif 0.05 > p >= 0.01:
        star = '*'
    else:
        star = None
    return star


def fisher_z_transform(r):
    """Convert correlation coefficient to Fisher z-score."""
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_inverse(z):
    """Convert Fisher z-score back to correlation coefficient."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def _spearman_nan(x, y):
    """Spearman correlation ignoring pairs where either value is NaN."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return spearmanr(x[mask], y[mask]).statistic


def _fisher_mean(rs):
    """Fisher-z average of correlation coefficients, back-transformed."""
    rs = np.asarray(rs, dtype=float)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return np.nan
    rs_clip = np.clip(rs, -0.9999, 0.9999)
    return float(fisher_z_inverse(np.mean(fisher_z_transform(rs_clip))))


def mantel_permutation_pvalue(x, y, n_permutations=5000,
                               alternative='greater', random_state=42,
                               chunk_size=500):
    """Mantel permutation test for the Spearman correlation between two RDMs.

    Reconstructs the squareform of ``x``, randomly permutes stimulus labels
    (i.e. applies the same permutation to rows and columns), re-condenses,
    and correlates against ``y``. This preserves the RDM's within-condition
    dependence structure and only shuffles the mapping of pairs to stimuli.

    Parameters
    ----------
    x, y : np.ndarray of shape (n_pairs,)
        Condensed RDM vectors (same ordering, as produced by
        ``scipy.spatial.distance.squareform``). NaN entries are ignored.
    n_permutations : int
    alternative : {'greater', 'less', 'two-sided'}
    random_state : int
    chunk_size : int
        Number of permutations processed at once in the vectorized path.

    Returns
    -------
    observed_r : float
    p_value : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed_r = _spearman_nan(x, y)
    if not np.isfinite(observed_r):
        return observed_r, np.nan

    x_sq = squareform(x, checks=False)
    n = x_sq.shape[0]
    triu_r, triu_c = np.triu_indices(n, k=1)
    rng = np.random.RandomState(random_state)

    has_nan = (not np.isfinite(x).all()) or (not np.isfinite(y).all())

    if not has_nan:
        # Fast path: rank y once, batch-rank permuted x, single matmul.
        n_pairs = x.size
        y_rank = y.argsort().argsort().astype(np.float64) + 1.0
        y_centered = y_rank - y_rank.mean()
        y_norm = y_centered / np.sqrt((y_centered ** 2).sum())
        rank_mean = (n_pairs + 1) / 2.0
        rank_var_sum = n_pairs * (n_pairs ** 2 - 1) / 12.0
        rank_denom = np.sqrt(rank_var_sum)

    r_perms = np.empty(n_permutations)
    for start in range(0, n_permutations, chunk_size):
        end = min(start + chunk_size, n_permutations)
        perms = rng.random((end - start, n)).argsort(axis=1)
        x_batch = x_sq[perms[:, triu_r], perms[:, triu_c]]

        if not has_nan:
            order = x_batch.argsort(axis=1)
            x_ranks = np.empty_like(order, dtype=np.float64)
            row_idx = np.arange(order.shape[0])[:, None]
            x_ranks[row_idx, order] = np.arange(1, n_pairs + 1,
                                                dtype=np.float64)
            x_centered = x_ranks - rank_mean
            r_perms[start:end] = (x_centered @ y_norm) / rank_denom
        else:
            for i in range(end - start):
                r_perms[start + i] = _spearman_nan(x_batch[i], y)

    valid = r_perms[np.isfinite(r_perms)]
    if valid.size == 0:
        return observed_r, np.nan

    if alternative == 'greater':
        count = int(np.sum(valid >= observed_r))
    elif alternative == 'less':
        count = int(np.sum(valid <= observed_r))
    else:
        count = int(np.sum(np.abs(valid) >= abs(observed_r)))

    return observed_r, (count + 1) / (valid.size + 1)


def sign_flip_permutation_pvalue(values, n_permutations=10000,
                                  alternative='greater', random_state=42,
                                  exact_threshold=15):
    """Sign-flip permutation test that the mean of ``values`` is 0.

    Under the null of a distribution symmetric about zero, each subject's
    value is equally likely to have a flipped sign. If ``n <= exact_threshold``
    all 2**n sign assignments are enumerated; otherwise ``n_permutations``
    Monte Carlo sign patterns are drawn.

    Parameters
    ----------
    values : array-like of shape (n_subjects,)
    n_permutations : int
    alternative : {'greater', 'less', 'two-sided'}
    random_state : int
    exact_threshold : int
        Maximum n for which exact enumeration is used (default 15).

    Returns
    -------
    p_value : float
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return np.nan
    observed = float(np.mean(values))

    if n <= exact_threshold:
        bits = np.arange(1 << n, dtype=np.int64)
        signs = 1 - 2 * ((bits[:, None] >> np.arange(n)) & 1).astype(np.int8)
        perm_means = signs @ values / n
    else:
        rng = np.random.RandomState(random_state)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8),
                           size=(n_permutations, n))
        perm_means = signs @ values / n

    if alternative == 'greater':
        count = int(np.sum(perm_means >= observed))
    elif alternative == 'less':
        count = int(np.sum(perm_means <= observed))
    else:
        count = int(np.sum(np.abs(perm_means) >= abs(observed)))

    return (count + 1) / (perm_means.size + 1)


def loso_noise_ceiling(rdms, use_fisher=True):
    """Leave-one-subject-out noise ceiling for a set of subject RDMs.

    For each subject i, computes Spearman(subject_i_rdm, mean of other
    subjects' RDMs). Averages across subjects (optionally via Fisher-z).

    Parameters
    ----------
    rdms : np.ndarray of shape (n_subjects, n_pairs)
    use_fisher : bool
        If True (default), average subject-wise correlations via Fisher-z.

    Returns
    -------
    noise_ceiling : float
    per_subject : np.ndarray of shape (n_subjects,)
        Per-subject LOSO correlations.
    """
    rdms = np.asarray(rdms, dtype=float)
    n_subjects = rdms.shape[0]
    if n_subjects < 2:
        return np.nan, np.array([np.nan] * n_subjects)

    per_subj = np.full(n_subjects, np.nan)
    all_idx = np.arange(n_subjects)
    for i in range(n_subjects):
        others = rdms[all_idx != i]
        mean_others = np.nanmean(others, axis=0)
        per_subj[i] = _spearman_nan(rdms[i], mean_others)

    if use_fisher:
        nc = _fisher_mean(per_subj)
    else:
        nc = float(np.nanmean(per_subj))
    return nc, per_subj


def bootstrap_mean_ci(values, n_bootstrap=1000, n_permutations=10000,
                       conf_level=0.95, random_state=42,
                       alternative='greater'):
    """Bootstrap CI and sign-flip p-value for the mean of per-subject scalars.

    Confidence interval: percentile CI over ``n_bootstrap`` subject
    resamples of the sample mean.

    P-value: sign-flip permutation. For each of ``n_permutations`` draws,
    each subject's value is multiplied by a random +/-1 and the mean
    recomputed; when the number of subjects is small (``n <= 15``) all
    2**n sign patterns are enumerated exactly.

    Parameters
    ----------
    values : array-like of shape (n_subjects,)
    n_bootstrap : int
    n_permutations : int
    conf_level : float
    random_state : int
    alternative : {'greater', 'less', 'two-sided'}

    Returns
    -------
    observed_mean : float
    ci_low, ci_high : float
    p_value : float
    boot_dist : np.ndarray of shape (n_bootstrap,)
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return (float(np.nanmean(values)) if n > 0 else np.nan,
                np.nan, np.nan, np.nan, np.full(n_bootstrap, np.nan))

    observed = float(np.mean(values))
    rng = np.random.RandomState(random_state)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot[i] = float(np.mean(values[idx]))

    alpha = 1 - conf_level
    ci_low = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    p_value = sign_flip_permutation_pvalue(
        values, n_permutations=n_permutations,
        alternative=alternative, random_state=random_state,
    )
    return observed, ci_low, ci_high, p_value, boot
