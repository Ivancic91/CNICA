import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class MIOptimizerResult:
    """
    Result of the MIOptimizer.fit_transform optimization.

    Designed to mirror scipy.optimize.OptimizeResult conventions.

    Attributes
    ----------
    x : np.ndarray
        The optimized transformation matrix M, shape (n, n).
    success : bool
        True if the optimizer converged within tolerance.
    message : str
        Human-readable description of the termination condition.
    fun : float
        Final value of the objective function.
    jac : np.ndarray
        Final projected tangent gradient, shape (n, n).
    nit : int
        Total number of inner iterations performed.
    nit_outer : int
        Number of outer (barrier cooling) iterations performed.
    inv_MT_C : np.ndarray
        M^{-T} @ C, for verifying the left constraint M^{-T} C > 0.
    MS : np.ndarray
        M @ S, for verifying the right constraint M S > 0.
    C_const : np.ndarray
        Boundary points of :math:`C` used for barrier constraints,
        in original data space.
    S_const : np.ndarray
        Boundary points of :math:`S` used for barrier constraints,
        in original data space.
    """
    x: NDArray[np.float64]
    success: bool
    message: str
    fun: float
    jac: NDArray[np.float64]
    nit: int
    nit_outer: int
    inv_MT_C: NDArray[np.float64]
    MS: NDArray[np.float64]
    C_const: NDArray[np.float64]
    S_const: NDArray[np.float64]

@dataclass
class NMFParams:
    """
    Parameters for the NMF initialization stage of CNICA.

    Attributes
    ----------
    init : str, default='nndsvd'
        Initialization method for the factor matrices. See
        `sklearn.decomposition.NMF` for valid options. Note that
        'custom' is set automatically in Stage 2 and should not
        be passed here.
    beta_loss : str, default='kullback-leibler'
        Loss function for the NMF objective. If 'frobenius', only
        a single coordinate-descent stage is run. Any other value
        triggers a second multiplicative-update refinement stage.
    max_iter : int, default=500
        Maximum number of NMF iterations per stage.
    tol : float, default=1e-4
        Convergence tolerance for the NMF updates.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    init: Literal['random', 'nndsvd', 'nndsvda', 'nndsvdar'] = 'nndsvd'
    beta_loss: Literal['frobenius', 'kullback-leibler', 'itakura-saito'] = 'kullback-leibler'
    max_iter: int = 100000
    tol: float = 1e-5
    random_state: int | None = None


@dataclass
class MIOParams:
    """
    Parameters for the MIOptimizer Riemannian refinement stage of CNICA.

    Attributes
    ----------
    params : tuple
        Three flags selecting which derivative orders to include in the mutual
        information objective: (0th, 1st, 2nd). At least one must be True. Default 
        is (True, True, False) for 0th and 1st order only.
    max_outer_loop_iters : int, default=10
        Number of barrier cooling steps.
    max_inner_loop_iters : int, default=1000
        Maximum Riemannian descent steps per fixed mu.
    max_line_search_iters : int, default=50
        Maximum backtracking steps per descent iteration.
    tol : float, default=1e-14
        Convergence tolerance on the duality gap.
    gamma : float, default=0.02
        Pruning fraction for near-origin points.
    initial_lr : float, default=0.1
        Initial step size for backtracking line search.
    filter_half_order : int, default=10
        Half-order of the Parks-McClellan FIR derivative filters.
    filter_max_lag : int, default=10
        Maximum autocorrelation lag for unsupervised filter tuning.
    lam : float, default=10.0
        Sparsity regularization strength.
    sparsity_gate_threshold : float, default=0.5
        Sigmoid gate midpoint for the sparsity norm.
    """
    params: Tuple[bool, bool, bool] = field(default_factory=lambda: (True, True, False))
    max_outer_loop_iters: int = 10
    max_inner_loop_iters: int = 1000
    max_line_search_iters: int = 50
    tol: float = 1e-8
    gamma: float = 0.02
    initial_lr: float = 0.1
    filter_half_order: int = 10
    filter_max_lag: int = 10
    lam: float = 20.0
    sparsity_gate_threshold: float = 0.8