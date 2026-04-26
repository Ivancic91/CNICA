import numpy as np
from numpy.typing import NDArray
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array


class NICАInit(BaseEstimator, TransformerMixin):
    r"""
    NICA-based initialization for Non-negative Matrix Factorization.

    Estimates initial factor matrices :math:`F^{(0)}` and :math:`G^{(0)}`
    for NMF by combining PCA dimensionality reduction with Non-negative
    Independent Component Analysis (NICA). The resulting initialization
    places bases along the edges of the convex polyhedral cone defined by
    the data, which is geometrically consistent with the NMF solution
    structure.

    The procedure follows three stages:

    1. **PCA reduction:** Project :math:`D` onto the top :math:`K`
       principal components:

       .. math::

           P_1 D = A S

    2. **Whitening and NICA:** Whiten the projected data and find a
       rotation matrix :math:`W` that minimizes the total power of
       negative residuals:

       .. math::

           \min_W \sum_{k,n} \min(0, y_{k,n})^2

       via steepest gradient descent with orthogonalization:

       .. math::

           \tilde{w}_k = w_k - 2\gamma \sum_n \min(0, y_{k,n}) z_{k,n}

           W = (\tilde{W}\tilde{W}^T)^{-1/2} \tilde{W}

    3. **Non-negativization:** Compute initial factor matrices as:

       .. math::

           F^{(0)} = |F|

           G^{(0)} = \alpha_G {F^{(0)}}^T D

       where :math:`\alpha_G = \sum_{m,n} x_{m,n} c_{m,n} /
       \sum_{m,n} c_{m,n}^2` and
       :math:`c_{m,n} = [F^{(0)} {F^{(0)}}^T D]_{m,n}`.

    Parameters
    ----------
    n_components : int
        Number of latent components :math:`K` to extract.
    gamma : float, default=0.1
        Step size for the NICA gradient descent update.
    max_iter : int, default=1000
        Maximum number of NICA gradient descent iterations.
    tol : float, default=1e-6
        Convergence tolerance on the change in objective value.
    whiten : bool, default=True
        Whether to whiten the PCA-projected data before NICA. Should
        always be True for correct NICA behavior.
    random_state : int or None, default=None
        Random seed for reproducibility of the initial rotation matrix.

    Attributes
    ----------
    F_ : np.ndarray of shape (n_components, n_features)
        Initial basis matrix after non-negativization.
    G_ : np.ndarray of shape (n_components, n_samples)
        Initial weight matrix after non-negativization.
    W_ : np.ndarray of shape (n_components, n_components)
        Estimated NICA demixing matrix.
    n_iter_ : int
        Number of NICA iterations performed.
    objective_history_ : list of float
        Objective value at each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from nica_init import NICAInit
    >>> D = np.abs(np.random.randn(20, 100))
    >>> init = NICAInit(n_components=3, random_state=42)
    >>> F, G = init.fit_transform(D)
    >>> F.shape
    (3, 20)
    >>> G.shape
    (3, 100)

    References
    ----------
    Kitamura, D., & Ono, N. (2016). Efficient nonnegative matrix
    factorization with random projections and nonnegative independent
    component analysis. IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP).
    """

    def __init__(
        self,
        n_components: int = 2,
        gamma: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        whiten: bool = True,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state

    def fit_transform(
        self,
        D: NDArray[np.float64],
        y: None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Fit the NICA initialization model and return factor matrices.

        Parameters
        ----------
        D : np.ndarray of shape (n_samples, n_features)
            Observed non-negative data matrix. Rows are observations
            (e.g. time points), columns are features (e.g. wavenumbers).
        y : None
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        F : np.ndarray of shape (n_components, n_features)
            Non-negative initial basis matrix :math:`F^{(0)}`.
        G : np.ndarray of shape (n_components, n_samples)
            Non-negative initial weight matrix :math:`G^{(0)}`.
        """
        D = np.asarray(check_array(D, accept_sparse=False), dtype=np.float64)
        n_samples, n_features = D.shape

        # --- Stage 1: PCA dimensionality reduction ---
        # Project D onto top K principal components without centering
        # to preserve non-negativity structure of the data.
        # P1 has shape (K, M), P1 @ D has shape (K, N)
        pca = PCA(n_components=self.n_components, whiten=False)
        # PCA expects (n_samples, n_features) so we transpose:
        # D.T has shape (n_features, n_samples)
        pca.fit(D.T)
        P1: NDArray[np.float64] = np.asarray(
            pca.components_, dtype=np.float64)  # (K, M)
        P1_D: NDArray[np.float64] = P1 @ D      # (K, N)

        # --- Stage 2: Whitening ---
        # Whiten P1_D so that P1_D @ P1_D.T becomes the identity matrix.
        # Note: no mean subtraction to preserve non-negativity structure.
        if self.whiten:
            cov = P1_D @ P1_D.T / n_samples          # (K, K)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-12)      # numerical stability
            V: NDArray[np.float64] = np.asarray(
                np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T,
                dtype=np.float64)                     # (K, K)
        else:
            V = np.eye(self.n_components, dtype=np.float64)

        Z: NDArray[np.float64] = V @ P1_D            # (K, N)

        # --- Stage 3: NICA gradient descent ---
        # Initialize W as a random orthogonal matrix
        rng = np.random.default_rng(self.random_state)
        W_raw = rng.standard_normal(
            (self.n_components, self.n_components))
        W: NDArray[np.float64] = np.asarray(
            self._orthogonalize(W_raw), dtype=np.float64)

        self.objective_history_: list[float] = []
        prev_obj = np.inf

        for iteration in range(self.max_iter):
            Y: NDArray[np.float64] = W @ Z           # (K, N)

            # Objective: total power of negative residuals
            neg_Y = np.minimum(0.0, Y)               # (K, N)
            obj = float(np.sum(neg_Y ** 2))
            self.objective_history_.append(obj)

            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj

            # Gradient step (equation 6)
            W_tilde = W - 2.0 * self.gamma * (neg_Y @ Z.T)

            # Orthogonalization (equation 7)
            W = np.asarray(
                self._orthogonalize(W_tilde), dtype=np.float64)

        self.W_ = W
        self.n_iter_ = iteration + 1

        # --- Stage 4: Recover basis matrix F ---
        # From equation 9: F ≈ P^{-1} [(WV)^{-1}; 0]
        # Since P is orthogonal, P^{-1} = P^T
        # (WV)^{-1} has shape (K, K), giving F shape (M, K) -> transpose to (K, M)
        WV: NDArray[np.float64] = W @ V              # (K, K)
        WV_inv: NDArray[np.float64] = np.asarray(
            np.linalg.inv(WV), dtype=np.float64)     # (K, K)

        # Embed into full M-dimensional space: pad with zeros for
        # the discarded PCA dimensions, then apply P^T
        # F_embedded shape: (M, K)
        F_embedded = P1.T @ WV_inv                   # (M, K)
        F_raw: NDArray[np.float64] = F_embedded.T    # (K, M)

        # --- Stage 5: Non-negativization ---
        # F^{(0)} = |F|
        F: NDArray[np.float64] = np.abs(F_raw)

        # G^{(0)} = alpha_G * F^{(0)T} @ D
        # where alpha_G = sum(D * C) / sum(C^2)
        # and C = F^{(0)} @ F^{(0)T} @ D  (shape: K x N)
        C: NDArray[np.float64] = F @ F.T @ D         # (K, N)
        alpha_G = float(
            np.sum(D * C.T) /                        # D is (M,N), C.T is (N,K)... 
            (np.sum(C ** 2) + 1e-12)
        )

        # Wait — D is (n_samples, n_features) = (N, M) but F is (K, M)
        # so F @ F.T @ D requires D to be (M, N).
        # We use D.T throughout to match the paper's convention.
        D_t: NDArray[np.float64] = D.T               # (M, N)
        C = F @ F.T @ D_t                            # (K, N)
        alpha_G = float(np.sum(D_t * (F.T @ C)) / (np.sum(C ** 2) + 1e-12))
        G: NDArray[np.float64] = np.maximum(
            0.0, alpha_G * F @ D_t)                  # (K, N)

        self.F_ = F
        self.G_ = G

        return F, G

    @staticmethod
    def _orthogonalize(W: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Symmetric orthogonalization via :math:`(\tilde{W}\tilde{W}^T)^{-1/2} \tilde{W}`.

        Parameters
        ----------
        W : np.ndarray of shape (K, K)
            Matrix to orthogonalize.

        Returns
        -------
        np.ndarray of shape (K, K)
            Orthogonalized matrix.
        """
        WWT = W @ W.T
        eigvals, eigvecs = np.linalg.eigh(WWT)
        eigvals = np.maximum(eigvals, 1e-12)
        WWT_inv_sqrt: NDArray[np.float64] = np.asarray(
            eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T,
            dtype=np.float64)
        return np.asarray(WWT_inv_sqrt @ W, dtype=np.float64)