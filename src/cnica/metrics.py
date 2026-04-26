import numpy as np
from numpy.typing import NDArray
from itertools import permutations
from scipy.stats import pearsonr # type: ignore[import-untyped]
from typing import NamedTuple


class EvaluationResult(NamedTuple):
    """
    Results of CNICA evaluation against ground truth.

    Attributes
    ----------
    r_S_mean : float
        Mean Pearson r across all spectral components.
    r_S_per_component : np.ndarray
        Pearson r for each spectral component, shape (n_components,).
    r_C_mean : float
        Mean Pearson r across all concentration components.
    r_C_per_component : np.ndarray
        Pearson r for each concentration component, shape (n_components,).
    r_S_raman : float or None
        Pearson r for the Raman spectral component only.
        None if raman_idx is None.
    r_S_fluorescence_mean : float or None
        Mean Pearson r across fluorescence spectral components only.
        None if raman_idx is None.
    reconstruction_error : float
        Relative Frobenius reconstruction error ||D - C^T S||_F / ||D||_F.
    best_permutation : list of int
        Component ordering that maximized mean spectral Pearson r.
    """
    r_S_mean: float
    r_S_per_component: NDArray[np.float64]
    r_C_mean: float
    r_C_per_component: NDArray[np.float64]
    r_S_raman: float | None
    r_S_fluorescence_mean: float | None
    reconstruction_error: float
    best_permutation: list[int]


def evaluate_cnica(
    S_true: NDArray[np.float64],
    C_true: NDArray[np.float64],
    S_est: NDArray[np.float64],
    C_est: NDArray[np.float64],
    D: NDArray[np.float64],
    raman_idx: int | None = 0,
) -> EvaluationResult:
    r"""
    Evaluate CNICA output against ground truth factor matrices.

    Resolves the permutation ambiguity of blind source separation by finding
    the component ordering that maximizes mean spectral Pearson r, then
    computes all metrics under that ordering. Scale ambiguity is handled
    naturally by Pearson r (scale-invariant) and the relative reconstruction
    error (normalized by ``||D||_F``).

    Parameters
    ----------
    S_true : NDArray[np.float64]
        Ground truth source matrix, shape (n_components, n_wave).
    C_true : NDArray[np.float64]
        Ground truth concentration matrix, shape (n_components, n_channels).
    S_est : NDArray[np.float64]
        Estimated source matrix, shape (n_components, n_wave).
    C_est : NDArray[np.float64]
        Estimated concentration matrix, shape (n_components, n_channels).
    D : NDArray[np.float64]
        Observed data matrix, shape (n_channels, n_wave). Used only for
        reconstruction error computation.
    raman_idx : int or None, default=0
        Index of the Raman component in ``S_true``. If not None, computes
        separate Raman and fluorescence spectral Pearson r values. Set to
        None for data without a sparse Raman component (e.g. pure
        fluorescence, spectroelectrochemistry).

    Returns
    -------
    EvaluationResult
        Named tuple containing all metrics. See class docstring for details.

    Notes
    -----
    The permutation search is :math:`O(r!)` in the number of components.
    For :math:`r > 6` this becomes slow; consider a greedy matching
    approach instead.

    Examples
    --------
    >>> result = evaluate_cnica(S_true, C_true, S_est, C_est, D, raman_idx=0)
    >>> print(f"Raman r:   {result.r_S_raman:.4f}")
    >>> print(f"Spectral r: {result.r_S_mean:.4f}")
    >>> print(f"Recon err: {result.reconstruction_error:.4f}")
    """
    n_components = S_true.shape[0]

    if n_components > 6:
        raise ValueError(
            f"n_components={n_components} would require {np.math.factorial(n_components)} "
            f"permutations. Use a greedy matching approach for r > 6."
        )

    # --- Find best permutation by maximizing mean spectral Pearson r ---
    best_perm = list(range(n_components))
    best_r_S_mean = -np.inf

    for perm in permutations(range(n_components)):
        r_S_mean = float(np.mean([
            pearsonr(S_true[i], S_est[perm[i]])[0]
            for i in range(n_components)
        ]))
        if r_S_mean > best_r_S_mean:
            best_r_S_mean = r_S_mean
            best_perm = list(perm)

    # Reorder estimated matrices to best permutation
    S_est_perm = S_est[best_perm]
    C_est_perm = C_est[best_perm]

    # --- Spectral Pearson r ---
    r_S_per_component = np.array([
        pearsonr(S_true[i], S_est_perm[i])[0]
        for i in range(n_components)
    ])
    r_S_mean = float(np.mean(r_S_per_component))

    # --- Concentration Pearson r ---
    r_C_per_component = np.array([
        pearsonr(C_true[i], C_est_perm[i])[0]
        for i in range(n_components)
    ])
    r_C_mean = float(np.mean(r_C_per_component))

    # --- Raman and fluorescence r (if applicable) ---
    if raman_idx is not None:
        r_S_raman = float(r_S_per_component[raman_idx])
        fluorescence_indices = [i for i in range(n_components) if i != raman_idx]
        r_S_fluorescence_mean = float(
            np.mean(r_S_per_component[fluorescence_indices])
        ) if fluorescence_indices else None
    else:
        r_S_raman = None
        r_S_fluorescence_mean = None

    # --- Relative reconstruction error ---
    D_rec = C_est_perm.T @ S_est_perm
    recon_error = float(
        np.linalg.norm(D - D_rec, 'fro') / (np.linalg.norm(D, 'fro') + 1e-12)
    )

    return EvaluationResult(
        r_S_mean=r_S_mean,
        r_S_per_component=r_S_per_component,
        r_C_mean=r_C_mean,
        r_C_per_component=r_C_per_component,
        r_S_raman=r_S_raman,
        r_S_fluorescence_mean=r_S_fluorescence_mean,
        reconstruction_error=recon_error,
        best_permutation=best_perm,
    )