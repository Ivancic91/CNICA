# pyright: reportConstantRedefinition=false
r"""
Core functionality (:mod:`~CNICA.core`)
=================================================================
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls # type: ignore[import-untyped]
from sklearn.decomposition import NMF
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from typing import Any, cast
import warnings

from .models import NMFParams, MIOParams
from .optimizer import MIOptimizer


logger = logging.getLogger(__name__)


class CNICA(BaseEstimator, TransformerMixin):
    r"""
    Coupled Non-negative Independent Component Analysis (CNICA).

    CNICA is designed for blind source separation of non-negative signals,
    with particular application to spectroscopic data analysis. Given an
    observed data matrix :math:`X` of shape (n_channels, n_samples) — for
    example, a matrix of mixed spectra measured at multiple time points or
    concentrations — CNICA decomposes :math:`X` into two non-negative factor
    matrices:

    .. math::

        X \approx C^T S

    where :math:`C` (shape: n_components :math:`\times` n_channels) is the mixing
    matrix (e.g. concentration profiles) and :math:`S` (shape: n_components
    :math:`\times` n_samples) is the source matrix (e.g. pure component
    spectra).

    The decomposition proceeds in two stages:

    1. **NMF initialization**: A Non-negative Matrix Factorization
    provides an initial estimate of :math:`C` and :math:`S`. A fast Frobenius
    coordinate-descent fit is always run first; if a different target
    loss (e.g. Kullback-Leibler) is specified via `nmf_params`, a
    second multiplicative-update refinement follows.

    2. **Riemannian mutual information minimization**: A linear
    transformation :math:`M` is found that minimizes the statistical
    dependence between components of :math:`S` (and optionally :math:`C`) and
    their temporal/spectral derivatives, while strictly enforcing
    non-negativity of :math:`MS` and :math:`M^{-T}C` via log-barrier penalties
    on a Riemannian manifold. This step recovers components that are
    physically distinct rather than merely mathematically orthogonal,
    which is the key advantage over standard NMF.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of latent components to extract (e.g. number of pure
        chemical species). If None, defaults to the number of rows in X.
    nmf_params : NMFParams or None, default=None
        Keyword arguments forwarded to `sklearn.decomposition.NMF` for
        the initialization stage. Defaults to ``{'init': 'nndsvd',
        'beta_loss': 'kullback-leibler', 'solver': 'mu',
        'max_iter': 500}``. Note that `beta_loss='frobenius'` with
        `solver='cd'` is always run first as a rough-in step regardless
        of this setting.
    mio_params : MIOParams or None, default=None
        Keyword arguments forwarded to `MIOptimizer.__init__` for the
        Riemannian refinement stage. Defaults to ``{'params': (True,
        True, False)}``, which includes 0th and 1st order derivative
        covariances in the mutual information objective.
    max_outer_iters : int, default=10
        Maximum number of outer barrier cooling iterations passed to
        `MIOptimizer`.
    tol : float, default=1e-12
        Convergence tolerance passed to `MIOptimizer`.

    Attributes
    ----------
    C_ : np.ndarray
        Refined mixing matrix (e.g. concentration profiles) after
        Riemannian optimization. Has shape (n_components, n_channels)
        following the signal.
    S_ : np.ndarray 
        Refined source matrix (e.g. pure component spectra) after
        Riemannian optimization. Has shape (n_components, n_samples)
        following the signal.
    M_ : np.ndarray
        Optimal transformation matrix found by `MIOptimizer`. Has shape
        (n_components, n_components).
    nmf_obj_ : sklearn.decomposition.NMF
        The fitted NMF object from the final NMF stage, exposing
        `reconstruction_err_` and `n_iter_` among other attributes.
    mio_result_ : MIOptimizerResult
        Full result object from `MIOptimizer.fit_transform`, including
        convergence status, final objective, gradient, and the transformed
        constraint matrices for verification.
    Examples
    --------
    >>> import numpy as np
    >>> from cnica import CNICA
    >>> # Simulate 3-component spectroscopic mixture (5 time points, 100 wavenumbers)
    >>> X = np.abs(np.random.randn(5, 100))
    >>> model = CNICA(n_components=3)
    >>> S = model.fit_transform(X)  # Pure component spectra
    >>> C = model.C_                # Concentration profiles
    >>> X_rec = C.T @ S             # Reconstruction

    Notes
    -----
    CNICA is particularly well-suited to spectroscopic datasets where:

    - Components are non-negative by physical constraint (e.g. absorbance,
      emission intensity).
    - Pure component signals are expected to be statistically independent
      — i.e. the presence of one species does not predict another.
    - Derivatives of the spectra or concentration profiles carry
      additional discriminative information (e.g. sharp peaks vs. broad
      backgrounds).

    The Riemannian step is sensitive to the quality of the NMF
    initialization. NND-SVD initialization (``init='nndsvd'``) is
    recommended for sparse data; random initialization may be preferable
    for dense data.
    """
    def __init__(
        self,
        n_components: int | None = None,
        nmf_params: NMFParams | None = None,
        mio_params: MIOParams | None = None
    ):
        self.n_components = n_components
        self.nmf_params = nmf_params
        self.mio_params = mio_params

    def fit_transform(
            self,
            X: Any,
            y: Any = None,
            **fit_params: Any
        ) -> NDArray[np.float64]:
        r"""
        Fit the CNICA model to X and return the refined source matrix.

        Runs NMF initialization followed by Riemannian mutual information
        minimization. The NMF stage always begins with a fast Frobenius
        coordinate-descent rough-in; if `nmf_params` specifies a different
        target loss (e.g. Kullback-Leibler), a second multiplicative-update
        refinement follows using the rough-in as initialization.

        Parameters
        ----------
        X : array-like
            Observed mixture matrix. Follows the signal processing convention
            of channels :math:`\times` samples — e.g. for spectroscopic data,
            rows are time points and columns are wavenumbers. Must be
            non-negative. Has shape (n_channels, n_samples).
        y : None
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        S_ : np.ndarray
            Refined source matrix (e.g. pure component spectra) after
            Riemannian mutual information minimization. Has shape 
            (n_components, n_samples).
        """
        X_2d: NDArray[np.float64] = check_array(X, accept_sparse=False)

        nmf_params = self.nmf_params or NMFParams()
        mio_params = self.mio_params or MIOParams()

        # Stage 1: Frobenius CD rough-in
        logger.info("Stage 1: Frobenius CD initialization...")
        nmf_stage1 = NMF(
            n_components=self.n_components,
            init=nmf_params.init,
            solver='cd',
            beta_loss='frobenius',
            max_iter=nmf_params.max_iter,
            tol=nmf_params.tol,
            random_state=nmf_params.random_state,
        )
        W_init = np.asarray(
            nmf_stage1.fit_transform(X_2d),  # type: ignore[union-attr]
            dtype=np.float64
        )
        H_init = np.asarray(
            nmf_stage1.components_,  # type: ignore[union-attr]
            dtype=np.float64
        )

        # Stage 2: Target loss refinement if requested
        if nmf_params.beta_loss == 'frobenius':
            self.nmf_obj_ = nmf_stage1
        else:
            logger.info(f"Stage 2: {nmf_params.beta_loss} MU refinement...")
            self.nmf_obj_ = NMF(
                n_components=self.n_components,
                init='custom',
                solver='mu',
                beta_loss=nmf_params.beta_loss,
                max_iter=nmf_params.max_iter,
                tol=nmf_params.tol,
                random_state=nmf_params.random_state,
            )

            W_init = np.asarray(
                self.nmf_obj_.fit_transform(X_2d, W=W_init, H=H_init),  # type: ignore[union-attr]
                dtype=np.float64
            )
            H_init = np.asarray(
                self.nmf_obj_.components_,  # type: ignore[union-attr]
                dtype=np.float64
            )

        logger.info(
            f"NMF complete. Reconstruction error: "
            f"{self.nmf_obj_.reconstruction_err_:.4f}"
        )

        # Adopt signal processing convention:
        # C: (n_components, n_channels) — mixing matrix / concentration profiles
        # S: (n_components, n_samples)  — source matrix / pure component spectra
        C_init = W_init.T
        S_init = H_init
        self.C_initial_ = C_init.copy()  # Store initial C for potential analysis
        self.S_initial_ = S_init.copy()  # Store initial S for potential analysis

        # Stage 3: Riemannian mutual information minimization.
        # Finds M such that MS > 0 and M^{-T}C > 0, minimizing statistical
        # dependence between source components.
        logger.info("Stage 3: Riemannian MI minimization...")
        mio = MIOptimizer(
            params=mio_params.params,
            max_outer_loop_iters=mio_params.max_outer_loop_iters,
            max_inner_loop_iters=mio_params.max_inner_loop_iters,
            max_line_search_iters=mio_params.max_line_search_iters,
            tol=mio_params.tol,
            gamma=mio_params.gamma,
            initial_lr=mio_params.initial_lr,
            filter_half_order=mio_params.filter_half_order,
            filter_max_lag=mio_params.filter_max_lag,
            lam=mio_params.lam,
            sparsity_gate_threshold=mio_params.sparsity_gate_threshold,
        )

        self.mio_result_ = mio.fit_transform(C=C_init, S=S_init)

        if not self.mio_result_.success:
            warnings.warn(
                f"MIOptimizer did not converge: {self.mio_result_.message}",
                stacklevel=2
            )

        logger.info(
            f"MIOptimizer complete. Success: {self.mio_result_.success}. "
            f"Message: {self.mio_result_.message}"
        )

        # Apply the optimal transformation to recover refined C and S
        self.M_ = self.mio_result_.x
        self.S_ = self.mio_result_.MS
        self.C_ = self.mio_result_.inv_MT_C

        return self.S_

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Estimate mixing coefficients for new data given fixed source spectra.

        Solves the non-negative least squares problem:

        .. math::

            \min_{C \geq 0} \|X - C^T S_*\|_F

        column by column, holding :math:`S_*` fixed. This is appropriate when
        :math:`S_*` contains known or previously identified pure component spectra
        and new observations :math:`X` are mixtures of those same components.

        Parameters
        ----------
        X : np.ndarray 
            New observed mixture matrix, where rows are channels (e.g.
            time points) and columns are samples (e.g. wavenumbers). Has shape
            (n_channels, n_samples).

        Returns
        -------
        C_new : np.ndarray
            Non-negative mixing coefficients for each channel in X. Has shape
            (n_components, n_channels).
        """
        S: NDArray[np.float64] = self.S_
        n_channels = X.shape[0]
        n_components = S.shape[0]
        C_new = np.zeros((n_components, n_channels))
        for i in range(n_channels):
            C_new[:, i] = cast(NDArray[np.float64], nnls(S.T, X[i])[0])
        return C_new