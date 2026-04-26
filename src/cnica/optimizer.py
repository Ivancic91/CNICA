import logging
from dataclasses import dataclass
import warnings
from numpy.typing import NDArray
import numpy as np
from typing import Any, List, Tuple, Optional, cast
from scipy.linalg import expm # type: ignore[import-untyped]
from scipy.spatial import ConvexHull # type: ignore[import-untyped]

from .filter import Filter
from .models import MIOptimizerResult

logger = logging.getLogger(__name__)

NUMERICAL_EPSILON = 1.0e-14 # Guards log(0) and 1/0 throughout
BARRIER_DELTA = 1e-2        # Barrier boundary offset: enforces MS > -delta
ARMIJO_SUFFICIENT_DECREASE = 1e-4
LINE_SEARCH_BACKTRACK_FACTOR = 0.5
BARRIER_COOLING_RATE = 1 / 50.0
SPARSITY_GATE_STEEPNESS = 30.0
NUM_PROBES_PER_D = 50
CONVEX_HULL_MAX_DIM = 5
CONVEX_HULL_MAX_SAMPLES = 1_000_000
RNG_SEED = 42


@dataclass
class FuncResult:
    """
    Result of an objective function evaluation.

    Attributes
    ----------
    f : float
        Scalar objective value.
    grad : np.ndarray or None
        Gradient array of the same shape as the input matrix :math:`M`,
        or None if ``compute_grad=False``.
    """
    f: float
    grad: Optional[NDArray[np.float64]] = None


class MIOptimizer:
    r"""
    Riemannian Manifold Optimizer using Interior Point (Log-Barrier) methods.

    This class minimizes a composite objective function on the manifold of
    invertible matrices. It is designed to find an optimal linear transformation
    :math:`M` that decorrelates signal components while strictly satisfying
    non-negativity constraints via log-barrier penalties.

    The objective function is defined as:

    .. math::

        f(M) = f_{\text{main}}(M) + \mu (h_R(M) + h_L(M))

    where :math:`f_{\text{main}}` is the log-determinant of transformed
    covariances, and :math:`h_R, h_L` are the barrier functions for the
    constraints :math:`MS > 0` and :math:`M^{-T}C > 0` respectively.

    Parameters
    ----------
    params : tuple
        Three flags ``(use_0th, use_1st, use_2nd)`` selecting which derivative
        orders to include in the decorrelation objective. At least one
        must be True.
    max_outer_loop_iters : int, default=10
        Number of iterations for the barrier cooling schedule. In each outer
        iteration, the barrier parameter :math:`\mu` is reduced.
    max_inner_loop_iters : int, default=1000
        Maximum number of descent steps to take for a fixed value of :math:`\mu`.
    max_line_search_iters : int, default=50
        Maximum number of backtracking steps to perform during the line search
        to satisfy the Armijo condition.
    tol : float, default=1e-12
        Convergence tolerance for the norm of the projected gradient.
    gamma : float, default=0.02
        The fraction of lowest-magnitude points to prune from the dataset to
        avoid numerical issues near the origin when enforcing the barrier
        constraints. The removed points are those with the minimum of either
        the percentile threshold or the max magnitude threshold.
    initial_lr : float, default=0.1
        The initial step size (learning rate) for the backtracking line search.
    filter_half_order : int, default=10
        Half-order :math:`m` of the Parks-McClellan FIR filters used to compute
        1st and 2nd order temporal derivatives of the input signals.
    filter_max_lag : int, default=10
        Maximum autocorrelation lag for unsupervised MSRAC-based filter tuning.
    lam : float, default=10.0
        Regularization strength for the sparsity norm. Set to 0 to disable.
    sparsity_gate_threshold : float, default=0.5
        Midpoint of the sigmoid gate in the sparsity norm. Rows with a
        Hoyer density below this value are penalized; rows above it
        (e.g. dense sine-like signals) are protected.

    Methods
    -------
    fit_transform(C, S)
        Executes the full optimization process to find the optimal matrix :math:`M`.
    optimize_step(M, inv_M, Sigmas, C, S, mu, initial_lr, max_line_search_iters)
        Performs a single Riemannian descent step with linearized backtracking.
    f(M, inv_M, Sigmas, C, S, mu, compute_grad=True)
        Calculates the total objective function and its projected tangent gradient.
    normalize(C, S)
        Rescales constraint matrices to balance barrier gradients.
    get_covariances(S, params)
        Computes the covariance matrices of the data and its derivatives.
    prune(data, gamma)
        Remove low-magnitude points near the origin from a data matrix.
    extract_boundary(data)
        Convex hull or support-function approximation of boundary points.

    Notes
    -----
    Three key numerical techniques keep the optimizer efficient and stable:

    1. **Boundary Reduction**: Barrier constraints are enforced only on the
    convex hull vertices of C and S (or a support-function approximation
    for d > 5), reducing barrier terms from O(samples) to O(vertices).

    2. **Tangent Space Projection**: The Euclidean gradient is projected onto
    the tangent space of SL(n) via

    .. math::

        Z = \nabla f - \frac{\operatorname{Tr}(M^{-1} \nabla f)}{n} \cdot M

    keeping :math:`\det(M)` stable throughout.

    3. **Two-Phase Line Search**: Backtracking first uses a cheap first-order
    inverse approximation

    .. math::

        (M - \alpha Z)^{-1} \approx M^{-1} + \alpha M^{-1} Z M^{-1}

    to find a candidate step size, then confirms with the exact matrix
    exponential retraction.

    Examples
    --------
    >>> opt = MIOptimizer(params=(True, True, False), tol=1e-10)
    >>> result = opt.fit_transform(C_data, S_data)
    >>> optimal_M = result.x
    """

    def __init__(
            self,
            params: Tuple[bool, bool, bool] = (True, True, False),
            max_outer_loop_iters: int = 10,
            max_inner_loop_iters: int = 1000,
            max_line_search_iters: int = 50,
            tol: float = 1e-14,
            gamma: float = 0.02,
            initial_lr: float = 0.1,
            filter_half_order: int = 10,
            filter_max_lag: int = 10,
            lam: float = 10.0,
            sparsity_gate_threshold: float = 0.5):
        self.params = params
        self.max_outer_loop_iters = max_outer_loop_iters
        self.max_inner_loop_iters = max_inner_loop_iters
        self.max_line_search_iters = max_line_search_iters
        self.tol = tol
        self.gamma = gamma
        self.initial_lr = initial_lr
        self.filter_half_order = filter_half_order
        self.filter_max_lag = filter_max_lag
        self.lam = lam
        self.sparsity_gate_threshold = sparsity_gate_threshold

    @staticmethod
    def main(
        M: NDArray[np.float64], 
        Sigmas: List[NDArray[np.float64]],
        compute_grad: bool = True
    ) -> FuncResult:
        r"""
        Decorrelation objective: sum of log diagonal variances of transformed
        covariances.

        For each covariance :math:`\Sigma` in `Sigmas` (one per active derivative
        order), computes:

        .. math::

            f_{\text{main}}(M) = \frac{1}{2} \sum_{\Sigma} \sum_{i}
            \log\left([M \Sigma M^T]_{ii} + \epsilon\right)

        Minimizing this drives :math:`M` toward a transformation where each output
        channel has low variance across all derivative orders. This is the
        diagonal approximation to the differential entropy of a Gaussian,
        where the :math:`1/2` arises from the information-theoretic derivation.
        The resulting gradient is :math:`D \cdot M\Sigma`, where
        :math:`D = \text{diag}(1 / [M \Sigma M^T]_{ii})`.

        Parameters
        ----------
        M : np.ndarray
            The :math:`(n, n)` linear transformation matrix.
        Sigmas : list of np.ndarray
            Covariance matrices of the 0th, 1st, or 2nd order derivatives
            of the source signals, as selected by `params` in `fit_transform`.
            Each must be :math:`(n, n)` and positive semi-definite.
        compute_grad : bool, default=True
            Whether to compute and return the Euclidean gradient.

        Returns
        -------
        FuncResult
            Scalar objective `f` and :math:`(n, n)` gradient `grad`
            (:math:`D \cdot M\Sigma`, summed over all :math:`\Sigma`).
        """
        res = FuncResult(f=0.0)
        if compute_grad:
            res.grad = np.zeros_like(M)
        
        for Sigma in Sigmas:
            M_Sigma = M @ Sigma
            transformed_Sigma = M_Sigma @ M.T
            diag_elements = np.diag(transformed_Sigma)
            res.f += 0.5 * np.sum(np.log(diag_elements + NUMERICAL_EPSILON))

            if compute_grad:
                D = np.diag(1.0 / (diag_elements + NUMERICAL_EPSILON))
                assert res.grad is not None
                res.grad += D @ M_Sigma

        return res

    @staticmethod
    def right_barrier(
        M: NDArray[np.float64], 
        S: NDArray[np.float64],
        mu: float,
        compute_grad: bool = True
    ) -> FuncResult:
        r"""
        Log-barrier enforcing the non-negativity constraint :math:`MS > 0`.

        Computes:

        .. math::

            h_R(M) = -\mu \sum_{i,j} \log\left([MS]_{ij} + \delta\right)

        The barrier approaches :math:`+\infty` as any element of :math:`MS`
        approaches :math:`-\delta`, repelling the optimizer from the constraint
        boundary and ensuring the transformed source estimates remain non-negative.
        Returns :math:`f = \infty` if the current :math:`M` is already infeasible.

        Parameters
        ----------
        M : np.ndarray
            The :math:`(n, n)` transformation matrix.
        S : np.ndarray
            The :math:`(n, n_S)` constraint matrix (convex hull vertices of the
            source estimates).
        mu : float
            Barrier strength. Reduced each outer iteration via
            `BARRIER_COOLING_RATE`.
        compute_grad : bool, default=True
            Whether to compute the Euclidean gradient
            :math:`-\mu \cdot W S^T` where :math:`W_{ij} = 1/([MS]_{ij} + \delta)`.

        Returns
        -------
        FuncResult
            Scalar objective `f` and :math:`(n, n)` gradient `grad`.
            Returns `f=np.inf` if any element of :math:`MS` violates the
            constraint boundary :math:`-\delta`.
        """
        M_S = M @ S

        # Infeasible: any entry has crossed the barrier boundary at -delta
        if np.any(M_S < -BARRIER_DELTA + NUMERICAL_EPSILON):
            res = FuncResult(f=np.inf)
            if compute_grad:
                res.grad = np.zeros_like(M)
            return res

        res = FuncResult(f=-mu * np.sum(np.log(M_S + BARRIER_DELTA)))

        if compute_grad:
            W = 1.0 / (M_S + BARRIER_DELTA)
            res.grad = -mu * (W @ S.T)
       
        return res

    @staticmethod
    def left_barrier(
        inv_M: NDArray[np.float64], 
        C: NDArray[np.float64],
        mu: float,
        compute_grad: bool = True
    ) -> FuncResult:
        r"""
        Log-barrier enforcing the non-negativity constraint :math:`M^{-T}C > 0`.

        Computes:

        .. math::

            h_L(M) = -\mu \sum_{i,j} \log\left([M^{-T}C]_{ij} + \delta\right)

        This is the dual constraint to :math:`MS > 0`: it ensures that the columns
        of :math:`C`, when mapped by the inverse transformation :math:`M^{-T}`,
        remain non-negative. Returns :math:`f = \infty` if the current :math:`M`
        is already infeasible.

        Parameters
        ----------
        inv_M : np.ndarray
            The :math:`(n, n)` inverse of the transformation matrix :math:`M`.
        C : np.ndarray
            The :math:`(n, n_C)` constraint matrix (convex hull vertices of the
            mixing matrix columns).
        mu : float
            Barrier strength. Reduced each outer iteration via
            `BARRIER_COOLING_RATE`.
        compute_grad : bool, default=True
            Whether to compute the Euclidean gradient with respect to :math:`M`,
            derived via
            :math:`\partial M^{-1}/\partial M \cdot dM = -M^{-1} dM\, M^{-1}`:

            .. math::

                \frac{\partial h_L}{\partial M} = \mu \left(M^{-T}C\right) V^T M^{-T}

            where :math:`V_{ij} = 1 / ([M^{-T}C]_{ij} + \delta)`.

        Returns
        -------
        FuncResult
            Scalar objective `f` and :math:`(n, n)` gradient `grad`.
            Returns `f=np.inf` if any element of :math:`M^{-T}C` violates the
            constraint boundary :math:`-\delta`.
        """
        inv_MT = inv_M.T
        inv_MT_C = inv_MT @ C

        # Check feasibility
        if np.any(inv_MT_C < -BARRIER_DELTA + NUMERICAL_EPSILON):
            res = FuncResult(f=np.inf)
            if compute_grad:
                res.grad = np.zeros_like(inv_M)
            return res

        # Value of the barrier function
        res = FuncResult(f=-mu * np.sum(np.log(inv_MT_C + BARRIER_DELTA)))

        # Gradient of the barrier function
        if compute_grad:
            V = 1.0 / (inv_MT_C + BARRIER_DELTA)
            res.grad = mu * (inv_MT_C @ V.T @ inv_MT)

        return res

    @staticmethod
    def sparse_norm(
        M: NDArray[np.float64],
        S_sums: NDArray[np.float64],
        S_gram: NDArray[np.float64],
        lam: float,
        n_samples: int,
        sparsity_gate_threshold: float = 0.5,
        compute_grad: bool = True
    ) -> FuncResult:
        r"""
        'Safe-Plateau' sparsity regularizer based on a gated Hoyer density.

        For each row :math:`c` of :math:`MS`, computes the Hoyer normalized density:

        .. math::

            d_c = \frac{\|MS_c\|_1 / \|MS_c\|_2 - 1}{\sqrt{N} - 1} \in [0, 1]

        where :math:`d_c = 0` is maximally sparse (one nonzero) and :math:`d_c = 1`
        is maximally dense (uniform). A negative softplus gate then rewards sparse
        rows while protecting dense ones:

        .. math::

            f_s(M) = -\frac{\lambda}{k} \sum_c
            \log\left(1 + e^{k(\beta - d_c)}\right)

        When :math:`d_c \ll \beta` the gate is open and the objective is a large
        negative number, rewarding the optimizer for preserving sparse (peaky) rows.
        When :math:`d_c \gg \beta` the gate saturates and the term vanishes,
        leaving dense rows (e.g. sine waves) governed solely by the decorrelation
        objective.

        Parameters
        ----------
        M : np.ndarray
            The :math:`(n, n)` transformation matrix.
        S_sums : np.ndarray
            Row sums of the source matrix: :math:`S \mathbf{1} / N`, length :math:`n`.
            Used to efficiently compute :math:`\|MS\|_1` without forming :math:`MS`.
        S_gram : np.ndarray
            Normalised Gram matrix :math:`SS^T / N`, shape :math:`(n, n)`.
            Used to efficiently compute :math:`\|MS\|_2` without forming :math:`MS`.
        lam : float
            Regularization strength. Set to 0 to disable entirely.
        n_samples : int
            Number of columns in :math:`S`, used to normalize the Hoyer density.
        sparsity_gate_threshold : float, default=0.5
            Midpoint :math:`\beta` of the sigmoid gate. Rows with density below
            this value are penalized; rows above it are protected.
        compute_grad : bool, default=True
            Whether to compute the gradient with respect to :math:`M`.

        Returns
        -------
        FuncResult
            Scalar objective `f` and :math:`(n, n)` gradient `grad`.
            Returns `f=0, grad=0` if `lam <= 0`.
        """
        if lam <= 0:
            return FuncResult(
                f=0.0,
                grad=np.zeros_like(M) if compute_grad else None
            )

        # L1 norm per row: ||MS_c||_1 = M @ S_sums
        num_vec = M @ S_sums

        # L2 norm per row: ||MS_c||_2 = sqrt(diag(M @ S_gram @ M^T))
        M_Sgram = M @ S_gram
        denom2_vec = np.sum(M_Sgram * M, axis=1)
        denom_vec = np.sqrt(denom2_vec) + NUMERICAL_EPSILON

        # Hoyer normalized density d_c in [0, 1]
        hoyer_scale = 1.0 / (np.sqrt(n_samples) - 1.0)
        ratio_vec = num_vec / denom_vec
        d_c = hoyer_scale * (ratio_vec - 1.0)

        # Negative softplus gate, numerically stable via logaddexp
        # f = -(lam/k) * sum(log(1 + exp(k * (beta - d_c))))
        diff = SPARSITY_GATE_STEEPNESS * (sparsity_gate_threshold - d_c)
        f = -(lam / SPARSITY_GATE_STEEPNESS) * np.sum(np.logaddexp(0.0, diff))
        res = FuncResult(f=f)

        if compute_grad:
            # Sigmoid gate: d/d(diff) of -log(1 + exp(diff)) = -sigma(diff)
            gates = lam / (1.0 + 1.0 / np.exp(diff))
            scalars = hoyer_scale * gates / denom_vec

            # Gradient of L1 term: d/dM ||MS||_1 = S_sums^T (outer product per row)
            grad_L1: NDArray[np.float64] = cast(
                NDArray[np.float64],
                np.outer(scalars, S_sums)
            )

            # Gradient of L2 term via quotient rule: (L1/L2^2) * M @ S_gram
            l2_scalars = ratio_vec / denom_vec
            grad_L2 = (scalars * l2_scalars)[:, np.newaxis] * M_Sgram

            res.grad = grad_L1 - grad_L2

        return res

    @staticmethod
    def f(
        M: NDArray[np.float64],
        inv_M: NDArray[np.float64],
        Sigmas: List[NDArray[np.float64]],
        C: NDArray[np.float64],
        S: NDArray[np.float64],
        mu: float,
        S_sums: NDArray[np.float64],
        S_gram: NDArray[np.float64],
        lam: float,
        n_samples: int,
        sparsity_gate_threshold: float = 0.8,
        compute_grad: bool = True
    ) -> FuncResult:
        r"""
        Total objective and its projection onto the tangent space of :math:`SL(n)`.

        Computes:

        .. math::

            f(M) = f_{\text{main}}(M) + h_R(M) + h_L(M) + f_s(M)

        When `compute_grad=True`, the summed Euclidean gradient :math:`\nabla f` is
        projected onto the tangent space by removing the component that would
        change :math:`\det(M)`:

        .. math::

            Z = \nabla f - \frac{\operatorname{Tr}(M^{-1} \nabla f)}{n} \cdot M

        This is derived by solving :math:`\operatorname{Tr}(M^{-1} Z) = 0` with the
        ansatz :math:`Z = \nabla f - cM`, which gives
        :math:`c = \operatorname{Tr}(M^{-1} \nabla f) / \operatorname{Tr}(I_n) =
        \operatorname{Tr}(M^{-1} \nabla f) / n`.

        Parameters
        ----------
        M : np.ndarray
            Current :math:`(n, n)` transformation matrix.
        inv_M : np.ndarray
            Inverse of :math:`M`, shape :math:`(n, n)`.
        Sigmas : list of np.ndarray
            Covariance matrices of the source signals and their derivatives,
            each of shape :math:`(n, n)`.
        C : np.ndarray
            :math:`(n, n_C)` left constraint matrix (convex hull vertices).
        S : np.ndarray
            :math:`(n, n_S)` right constraint matrix (convex hull vertices).
        mu : float
            Current barrier strength.
        S_sums : np.ndarray
            Row sums of the source matrix, length :math:`n`.
        S_gram : np.ndarray
            Normalised Gram matrix :math:`SS^T / N`, shape :math:`(n, n)`.
        lam : float
            Sparsity regularization strength.
        n_samples : int
            Number of columns in :math:`S`.
        sparsity_gate_threshold : float, default=0.5
            Passed through to `sparse_norm` as the sigmoid gate midpoint :math:`\beta`.
        compute_grad : bool, default=True
            Whether to compute the projected tangent gradient :math:`Z`.

        Returns
        -------
        FuncResult
            Scalar objective `f` and projected tangent gradient `grad` :math:`Z`.
        """
        res_m = MIOptimizer.main(M, Sigmas, compute_grad=compute_grad)
        res_r = MIOptimizer.right_barrier(M, S, mu, compute_grad=compute_grad)
        res_l = MIOptimizer.left_barrier(inv_M, C, mu, compute_grad=compute_grad)
        res_s = MIOptimizer.sparse_norm(M, S_sums, S_gram, lam, n_samples,
                                        sparsity_gate_threshold,
                                        compute_grad=compute_grad)

        res = FuncResult(f=res_m.f + res_r.f + res_l.f + res_s.f)

        if compute_grad:
            assert (res_m.grad is not None and res_r.grad is not None
                    and res_l.grad is not None and res_s.grad is not None), \
                    "All gradients must be computed"
            grad = res_m.grad + res_r.grad + res_l.grad + res_s.grad

            # Project onto the tangent space of SL(n): remove the det-changing
            # component so Tr(M^{-1} Z) = 0
            n = M.shape[0]
            proj_term = np.trace(inv_M @ grad) / n
            res.grad = grad - proj_term * M

        return res

    @staticmethod
    def optimize_step(
        M: NDArray[np.float64],
        inv_M: NDArray[np.float64],
        Sigmas: List[NDArray[np.float64]],
        C: NDArray[np.float64],
        S: NDArray[np.float64],
        mu: float,
        S_sums: NDArray[np.float64],
        S_gram: NDArray[np.float64],
        lam: float,
        n_samples: int,
        sparsity_gate_threshold: float,
        initial_lr: float,
        max_line_search_iters: int
    ) -> Tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Single Riemannian descent step via two-phase backtracking line search.

        **Phase 1 — Linearized backtracking:** Uses the first-order Taylor
        approximation of the matrix inverse to cheaply find a step size
        :math:`\alpha` satisfying the Armijo sufficient decrease condition:

        .. math::

            (M - \alpha Z)^{-1} \approx M^{-1} + \alpha M^{-1} Z M^{-1}

        **Phase 2 — Manifold retraction:** Confirms the step using the exact
        matrix exponential retraction, starting from the :math:`\alpha` found in
        Phase 1 and continuing to backtrack if needed:

        .. math::

            M_{\text{new}} = M \exp(-\alpha \Omega), \quad
            M_{\text{new}}^{-1} = \exp(\alpha \Omega) M^{-1}

        where :math:`\Omega = M^{-1} Z` is the descent direction in the Lie algebra.

        Parameters
        ----------
        M : np.ndarray
            Current :math:`(n, n)` transformation matrix.
        inv_M : np.ndarray
            Inverse of :math:`M`, shape :math:`(n, n)`.
        Sigmas : list of np.ndarray
            Covariance matrices of the source signals and their derivatives,
            each of shape :math:`(n, n)`.
        C : np.ndarray
            :math:`(n, n_C)` left constraint matrix (convex hull vertices).
        S : np.ndarray
            :math:`(n, n_S)` right constraint matrix (convex hull vertices).
        mu : float
            Current barrier strength.
        S_sums : np.ndarray
            Row sums of the source matrix, length :math:`n`.
        S_gram : np.ndarray
            Normalised Gram matrix :math:`SS^T / N`, shape :math:`(n, n)`.
        lam : float
            Sparsity regularization strength.
        n_samples : int
            Number of columns in :math:`S`.
        sparsity_gate_threshold : float
            Sigmoid gate midpoint :math:`\beta` passed through to `sparse_norm`.
        initial_lr : float
            Starting step size :math:`\alpha` for backtracking.
        max_line_search_iters : int
            Maximum number of backtracking steps in each phase before
            declaring failure.

        Returns
        -------
        success : bool
            True if a descent step satisfying the Armijo condition was found.
        M_new : np.ndarray
            Updated transformation matrix (unchanged if `success=False`).
        inv_M_new : np.ndarray
            Inverse of `M_new` (unchanged if `success=False`).
        """
        res = MIOptimizer.f(M, inv_M, Sigmas, C, S, mu, S_sums, S_gram, lam,
                            n_samples, sparsity_gate_threshold)
        f_curr = res.f
        Z = res.grad
        assert Z is not None, "Gradient must be computed for optimization step"

        # Descent direction in the Lie algebra: Omega = M^{-1} Z
        Omega = inv_M @ Z
        norm_Z_sq = float(np.dot(Z.ravel(), Z.ravel()))

        # Precompute the alpha coefficient of the inverse Taylor expansion:
        # (M - alpha Z)^{-1} approx M^{-1} + alpha * inv_M_Z_inv_M
        inv_M_Z_inv_M = Omega @ inv_M

        # Phase 1: Linearized backtracking to find a candidate alpha
        alpha = initial_lr
        found_step = False
        for _ in range(max_line_search_iters):
            M_try = M - alpha * Z
            inv_M_try = inv_M + alpha * inv_M_Z_inv_M

            res_try = MIOptimizer.f(M_try, inv_M_try, Sigmas, C, S, mu,
                                    S_sums, S_gram, lam, n_samples,
                                    sparsity_gate_threshold,
                                    compute_grad=False)

            if res_try.f < f_curr - ARMIJO_SUFFICIENT_DECREASE * alpha * norm_Z_sq:
                found_step = True
                break
            alpha *= LINE_SEARCH_BACKTRACK_FACTOR

        if not found_step:
            return False, M, inv_M
        
        # Phase 2: Exact manifold retraction, starting from alpha found in Phase 1
        for _ in range(max_line_search_iters):
            M_try = M @ expm(-alpha * Omega)
            inv_M_try = expm(alpha * Omega) @ inv_M

            res_try = MIOptimizer.f(M_try, inv_M_try, Sigmas, C, S, mu,
                                    S_sums, S_gram, lam, n_samples,
                                    sparsity_gate_threshold,
                                    compute_grad=False)

            if res_try.f < f_curr - ARMIJO_SUFFICIENT_DECREASE * alpha * norm_Z_sq:
                return True, M_try, inv_M_try
            alpha *= LINE_SEARCH_BACKTRACK_FACTOR

        return False, M, inv_M
    
    @staticmethod
    def normalize(
        C: NDArray[np.float64],
        S: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Rescales :math:`C` and :math:`S` so that their row variances are balanced
        across the left and right barrier terms, and both matrices have unit
        standard deviation.

        Computes two diagonal scaling matrices :math:`\Lambda_C` and
        :math:`\Lambda_S` such that:

        .. math::

            C_{\text{norm}} = \Lambda_C \cdot C, \quad
            S_{\text{norm}} = \Lambda_S \cdot S

        The scaling has two components applied sequentially:

        1. **Barrier balancing:** :math:`\lambda_i = (\text{Var}(C_i) /
        \text{Var}(S_i))^{1/4}` distributes the variance ratio evenly
        between :math:`C` and :math:`S` in log-space, equalizing barrier
        gradient magnitudes.

        2. **Std normalization:** both matrices are divided by their global
        standard deviation so typical entry magnitudes are :math:`O(1)`,
        making `BARRIER_DELTA` meaningful across datasets of different scales.

        The scaling matrices allow the transformation to be inverted exactly:

        .. math::

            C = \Lambda_C^{-1} C_{\text{norm}}, \quad
            S = \Lambda_S^{-1} S_{\text{norm}}

        Parameters
        ----------
        C : np.ndarray
            :math:`(n, n_C)` left constraint matrix.
        S : np.ndarray
            :math:`(n, n_S)` right constraint matrix.

        Returns
        -------
        C_norm : np.ndarray
            Normalized version of :math:`C`, shape :math:`(n, n_C)`.
        S_norm : np.ndarray
            Normalized version of :math:`S`, shape :math:`(n, n_S)`.
        Lambda_C : np.ndarray
            :math:`(n, n)` diagonal scaling matrix for :math:`C`.
        Lambda_S : np.ndarray
            :math:`(n, n)` diagonal scaling matrix for :math:`S`.
        """
        # Step 1: Barrier balancing via row variance ratio
        diag = np.diag(np.cov(C)) / (np.diag(np.cov(S)) + NUMERICAL_EPSILON)
        row_scale = diag ** 0.25
        C_norm = C / row_scale[:, None]
        S_norm = S * row_scale[:, None]

        # Step 2: Global std normalization so entries are O(1)
        scale_C = np.std(C_norm) + NUMERICAL_EPSILON
        scale_S = np.std(S_norm) + NUMERICAL_EPSILON
        C_norm = C_norm / scale_C
        S_norm = S_norm / scale_S

        # Compose both steps into single diagonal scaling matrices
        # Lambda_C @ C = C_norm, Lambda_S @ S = S_norm
        Lambda_C = np.diag(1.0 / (row_scale * scale_C))
        Lambda_S = np.diag(row_scale / scale_S)

        return C_norm, S_norm, Lambda_C, Lambda_S

    def get_covariances(
            self,
            S: NDArray[np.float64],
            params: Tuple[bool, bool, bool]) -> List[NDArray[np.float64]]:
        r"""
        Covariance matrices of :math:`S` and its filtered temporal derivatives.

        For each active flag in `params`, computes the :math:`(n, n)` covariance
        matrix of the corresponding derivative of :math:`S`. All orders are
        computed on the band-limited signal via unsupervised Parks-McClellan FIR
        filters tuned by minimizing the MSRAC of the rejected band, ensuring
        consistent frequency content across all covariance matrices.

        If the signal is too short to apply filtering safely, all orders fall
        back to using the raw (unfiltered) signal.

        Parameters
        ----------
        S : np.ndarray
            :math:`(n, n_{\text{samples}})` source signal matrix, one row per
            channel.
        params : tuple
            Flags selecting which derivative orders to include:
            ``(use_0th, use_1st, use_2nd)``. At least one must be True.
            Multiple True flags produce multiple covariance matrices, all
            included in the returned list.

        Returns
        -------
        list of np.ndarray
            Covariance matrices of shape :math:`(n, n)`, one per active flag in
            `params`, in order of derivative order.

        Warns
        -----
        UserWarning
            If :math:`n_{\text{samples}} < 2m + \text{max_{\text{lag}}} + 1`, the signal
            is too short to filter safely. All derivative orders fall back to
            raw (unfiltered) covariances.
        """
        n, n_samples = S.shape
        m = self.filter_half_order
        max_lag = self.filter_max_lag
        covariances: List[NDArray[Any]] = []

        # Orders 0 and 1 both trim m samples from each end, making
        # 2m + max_lag + 1 the binding minimum length constraint.
        min_samples = 2 * m + max_lag + 1
        if n_samples < min_samples:
            warnings.warn(
                f"Signal length ({n_samples}) is too short for filtering "
                f"(need {min_samples}). Falling back to raw covariances for "
                f"all orders. Consider reducing filter_half_order or "
                f"filter_max_lag."
            )
            covariances = [np.cov(S).astype(np.float64)]
            return covariances

        for index, use_derivative in enumerate(params):
            if not use_derivative:
                continue

            trimmed_len = n_samples - 2 * m
            filtered_S = np.zeros((n, trimmed_len))
            for i in range(n):
                filt = Filter(m=m, max_lag=max_lag, trim=True)
                filtered_S[i] = filt.transform(S[i], order=index)
            covariances.append(np.cov(filtered_S))

        return covariances

    def prune(
        self,
        data: NDArray[np.float64],
        gamma: float = 0.01
    ) -> NDArray[np.float64]:
        r"""
        Removes low-magnitude points clustered near the origin.

        Two thresholds are computed and the minimum is taken as the effective
        cutoff, protecting against outliers inflating the max-based threshold
        and against dense origin clusters inflating the percentile-based one:

        - Percentile threshold: the :math:`\gamma`-percentile of point magnitudes.
        - Max threshold: :math:`\gamma \times \max(\|x\|)`.

        Points with magnitude below the effective threshold are removed. If
        pruning would leave fewer than :math:`d + 1` points (the minimum for a
        non-degenerate convex hull in :math:`d` dimensions), the original data
        is returned unchanged.

        Parameters
        ----------
        data : np.ndarray
            :math:`(d, N)` data matrix.
        gamma : float, default=0.01
            Controls the pruning threshold as a fraction of the data's
            magnitude range. Larger values prune more aggressively.

        Returns
        -------
        np.ndarray
            Pruned data matrix of shape :math:`(d, N')` where :math:`N' \leq N`.
        """
        d, N = data.shape
        if N < 3:
            return data

        mags = np.linalg.norm(data, axis=0)
        thresh_percentile: float = cast(
            float, 
            np.percentile(mags, gamma * 100)
        )
        thresh_max: float = gamma * np.max(mags)
        effective_threshold: float = min(thresh_percentile, thresh_max)

        pruned_data = data[:, mags > effective_threshold]

        # d + 1 is the minimum number of points for a non-degenerate
        # convex hull in d dimensions
        if pruned_data.shape[1] <= d + 1:
            return data

        return pruned_data

    def extract_boundary(
        self,
        data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Extracts the outer boundary points of the dataset.

        Uses one of two methods depending on dimensionality and dataset size:

        **Tier 1 — Exact Convex Hull** (:math:`d \leq` `CONVEX_HULL_MAX_DIM`,
        :math:`N \leq` `CONVEX_HULL_MAX_SAMPLES`): computes the exact convex hull
        via QHull and returns its vertices. Falls back to Tier 2 if QHull fails
        (e.g. perfectly collinear data). These thresholds reflect empirically
        observed performance limits of the QHull algorithm.

        **Tier 2 — Support Function Approximation** (high-dimensional or large
        :math:`N`): evaluates the support function of the dataset along :math:`2d`
        axis-aligned basis directions and
        :math:`d \times \texttt{NUM_PROBES_PER_D}` random unit directions,
        returning the unique set of extreme points found.

        Reference
        ---------
        Sartipizadeh, H., & Suryanarayanan, S. (2016).
        "Computing the Approximate Convex Hull in High Dimensions."
        IEEE Access.

        Parameters
        ----------
        data : np.ndarray
            :math:`(d, N)` data matrix of boundary candidates.

        Returns
        -------
        np.ndarray
            :math:`(d, N')` matrix of boundary points where :math:`N' \leq N`.
            Returns `data` unchanged if :math:`N < 3`.
        """
        d, N = data.shape
        if N < 3:
            return data

        # Tier 1: Exact Convex Hull for low-dimensional, manageable datasets
        if d <= CONVEX_HULL_MAX_DIM and N <= CONVEX_HULL_MAX_SAMPLES:
            try:
                hull_indices = ConvexHull(data.T).vertices
                return data[:, hull_indices]
            except Exception:
                pass  # Fall through to Tier 2 if QHull fails

        # Tier 2: Support Function Approximation via Directional Probes

        # Axis-aligned probes: guard against constraint violations along
        # each coordinate axis and its negation
        basis = np.eye(d)
        probes_list = [basis, -basis]

        # Random probes: guard against violations along diagonal directions
        rng = np.random.default_rng(RNG_SEED)
        random_probes = rng.standard_normal(size=(NUM_PROBES_PER_D * d, d))
        random_probes /= np.linalg.norm(random_probes, axis=1, keepdims=True)
        probes_list.append(random_probes)

        # For each probe direction, find the point with maximum projection
        # all_probes: (P, d) @ data: (d, N) -> (P, N)
        all_probes = np.vstack(probes_list)
        extreme_indices = np.argmax(all_probes @ data, axis=1)

        return data[:, np.unique(extreme_indices)]

    def fit_transform(
            self,
            C: NDArray[np.float64],
            S: NDArray[np.float64]) -> MIOptimizerResult:
        r"""
        Orchestrates the barrier method to find the optimal transformation
        :math:`M`.

        Follows an interior point central path: starting from a large :math:`\mu`,
        the inner loop minimizes :math:`f(M)` for fixed :math:`\mu`, then
        :math:`\mu` is cooled by `BARRIER_COOLING_RATE` and the process repeats
        until the duality gap
        :math:`n_{\text{constraints}} \cdot \mu` falls below `tol`.

        :math:`M` is optimized in :math:`SL(n)` by projecting gradients onto the
        tangent space, keeping :math:`\det(M)` stable throughout.

        Parameters
        ----------
        C : np.ndarray
            :math:`(n, n_C)` left constraint matrix. All elements must be
            non-negative.
        S : np.ndarray
            :math:`(n, n_S)` right constraint (source) matrix. All elements must
            be non-negative.

        Returns
        -------
        MIOptimizerResult
            Contains the optimized matrix ``x``, convergence info
            ``success`` and ``message``, final objective ``fun``, final
            gradient ``jac``, iteration counts ``nit`` and ``nit_outer``,
            and the transformed constraint matrices ``inv_MT_C`` and ``MS``
            for verification.
        """
        params = self.params
        use_0, use_1, use_2 = params
        assert use_0 or use_1 or use_2, \
            "At least one of use_0, use_1, use_2 must be True"
        assert np.all(C >= 0), "All elements of C must be non-negative"
        assert np.all(S >= 0), "All elements of S must be non-negative"

        n_samples_S = S.shape[1]

        # Normalize C and S to balance left and right barrier gradients
        # Normalize and obtain scaling matrices
        C_copy, S_copy, Lambda_C, Lambda_S = MIOptimizer.normalize(
            np.array(C, dtype=float, copy=True),
            np.array(S, dtype=float, copy=True)
        )

        # Reduce C and S to boundary points: pruning removes near-origin
        # noise, extract_boundary retains only convex hull vertices.
        # Barrier constraints need only be enforced on the boundary since
        # the optimal solution lies there.
        C_const = self.extract_boundary(self.prune(C_copy, gamma=self.gamma))
        S_const = self.extract_boundary(self.prune(S_copy, gamma=self.gamma))
        n_constraints = C_const.size + S_const.size

        # Covariance matrices for the decorrelation objective
        Sigmas = self.get_covariances(S_copy, params)

        # Validate covariance matrices are positive semi-definite
        assert all(
            np.all(np.linalg.eigvalsh(Sigma) >= -NUMERICAL_EPSILON)
            for Sigma in Sigmas
        ), "All covariance matrices must be positive semi-definite"

        # Precompute sparsity norm statistics
        S_sums = np.sum(S_copy, axis=1) / n_samples_S
        S_gram = S_copy @ S_copy.T / n_samples_S
        logger.info(f"Sparsity regularization strength lambda: {self.lam:.4f}")

        # Initial M is the identity
        n = C.shape[0]
        M = np.eye(n)
        inv_M = np.eye(n)

        # Set initial mu so the barrier gradients are a fixed fraction
        # BARRIER_COOLING_RATE of the main objective gradient at M=I
        res_main = MIOptimizer.main(M, Sigmas)
        res_right = MIOptimizer.right_barrier(M, S_const, mu=1.0)
        res_left = MIOptimizer.left_barrier(inv_M, C_const, mu=1.0)
        logger.debug(f"f_main={res_main.f:.4f}, f_right_barrier={res_right.f:.4f}, "
              f"f_left_barrier={res_left.f:.4f}")
        assert (res_main.grad is not None and res_right.grad is not None
                and res_left.grad is not None), \
            "Gradients must be computed for initial mu calculation"
        logger.debug(f"||grad_main||={np.linalg.norm(res_main.grad):.4e}, "
              f"||grad_right||={np.linalg.norm(res_right.grad):.4e}, "
              f"||grad_left||={np.linalg.norm(res_left.grad):.4e}, "
              f"||grad_right + grad_left||={np.linalg.norm(res_right.grad + res_left.grad):.4e}")
        mu = float(
            BARRIER_COOLING_RATE
            * np.linalg.norm(res_main.grad)
            / np.linalg.norm(res_right.grad + res_left.grad)
        )
        logger.debug(f"mu={mu:.4e} calculated to balance barrier and main gradients")

        res = MIOptimizer.f(M, inv_M, Sigmas, C_const, S_const, mu,
                            S_sums, S_gram, self.lam, n_samples_S,
                            self.sparsity_gate_threshold)

        total_inner_iters = 0
        outer_iter = 0
        converged = False

        for outer_iter in range(self.max_outer_loop_iters):
            logger.info(
                f"Outer iteration {outer_iter + 1}/"
                f"{self.max_outer_loop_iters}, "
                f"mu={mu:.2e}, f={res.f:.4f}"
            )

            for __ in range(self.max_inner_loop_iters):
                success, M_new, inv_M_new = MIOptimizer.optimize_step(
                    M, inv_M, Sigmas, C_const, S_const, mu,
                    S_sums, S_gram, self.lam, n_samples_S,
                    self.sparsity_gate_threshold,
                    self.initial_lr, self.max_line_search_iters)

                if not success:
                    break

                M[:] = M_new
                inv_M[:] = inv_M_new
                total_inner_iters += 1

                res = MIOptimizer.f(
                    M, inv_M, Sigmas, C_const, S_const, mu,
                    S_sums, S_gram, self.lam, n_samples_S,
                    self.sparsity_gate_threshold)
                assert res.grad is not None, \
                    "Gradient must be computed for convergence check"

                if np.linalg.norm(res.grad) < self.tol:
                    break

            # Duality gap proxy: n_constraints * mu bounds the suboptimality
            # of the current central path solution
            if n_constraints * mu < self.tol:
                converged = True
                break

            mu *= BARRIER_COOLING_RATE

        assert res.grad is not None, \
            "Gradient must be computed for final result"

        message = (
            "Converged: duality gap below tolerance."
            if converged
            else "Did not converge: maximum iterations reached."
        )
        
        # Recover results in original space using Lambda inverses.
        inv_Lambda_C = np.diag(1.0 / np.diag(Lambda_C))
        inv_Lambda_S = np.diag(1.0 / np.diag(Lambda_S))

        return MIOptimizerResult(
            x=M,
            success=converged,
            message=message,
            fun=res.f,
            jac=res.grad,
            nit=total_inner_iters,
            nit_outer=outer_iter + 1,
            inv_MT_C=inv_M.T @ C,
            MS=M @ S,
            C_const=inv_Lambda_C @ C_const,
            S_const=inv_Lambda_S @ S_const
        )