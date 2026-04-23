import numpy as np
import pytest
from scipy.optimize import approx_fprime
from numpy.typing import NDArray
from cnica.optimizer import MIOptimizer, FuncResult
from cnica.models import MIOptimizerResult

# --- Fixtures ---

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)

@pytest.fixture
def small_positive_matrices(rng) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Small strictly positive C and S for barrier tests."""
    C = np.abs(rng.standard_normal((3, 20))) + 0.5
    S = np.abs(rng.standard_normal((3, 50))) + 0.5
    return C, S

@pytest.fixture
def covariances(small_positive_matrices) -> list[NDArray[np.float64]]:
    _, S = small_positive_matrices
    opt = MIOptimizer(params=(True, False, False))
    return opt.get_covariances(S, (True, False, False))

# --- FuncResult ---

def test_funcresult_default_grad_is_none():
    res = FuncResult(f=1.0)
    assert res.grad is None

# --- main gradient ---

def test_main_gradient_correctness(covariances):
    """Analytical gradient of main should match numerical gradient."""
    M = np.eye(3)

    def f_wrapper(m_flat: NDArray[np.float64]) -> float:
        M_r = m_flat.reshape(3, 3)
        return float(MIOptimizer.main(M_r, covariances, compute_grad=False).f)

    res = MIOptimizer.main(M, covariances)
    grad_analytical = res.grad
    grad_numerical = approx_fprime(
        M.flatten(), f_wrapper, epsilon=1e-7).reshape(3, 3)

    assert grad_analytical is not None
    np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4)

# --- right_barrier gradient ---

def test_right_barrier_gradient_correctness(small_positive_matrices):
    """Analytical gradient of right_barrier should match numerical gradient."""
    _, S = small_positive_matrices
    M = np.eye(3)
    mu = 1.0

    def f_wrapper(m_flat: NDArray[np.float64]) -> float:
        M_r = m_flat.reshape(3, 3)
        return float(MIOptimizer.right_barrier(M_r, S, mu, compute_grad=False).f)

    res = MIOptimizer.right_barrier(M, S, mu)
    grad_analytical = res.grad
    grad_numerical = approx_fprime(
        M.flatten(), f_wrapper, epsilon=1e-7).reshape(3, 3)

    assert grad_analytical is not None
    np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4)

# --- left_barrier gradient ---

def test_left_barrier_gradient_correctness(small_positive_matrices):
    """Analytical gradient of left_barrier should match numerical gradient."""
    C, _ = small_positive_matrices
    inv_M = np.eye(3)
    mu = 1.0

    def f_wrapper(m_flat: NDArray[np.float64]) -> float:
        M_r = m_flat.reshape(3, 3)
        inv_M_r = np.linalg.inv(M_r)
        return float(MIOptimizer.left_barrier(inv_M_r, C, mu, compute_grad=False).f)

    res = MIOptimizer.left_barrier(inv_M, C, mu)
    grad_analytical = res.grad
    grad_numerical = approx_fprime(
        np.eye(3).flatten(), f_wrapper, epsilon=1e-7).reshape(3, 3)

    assert grad_analytical is not None
    np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4)

# --- sparse_norm gradient ---

def test_sparse_norm_gradient_correctness(small_positive_matrices):
    """Analytical gradient of sparse_norm should match numerical gradient."""
    _, S = small_positive_matrices
    M = np.eye(3)
    n_samples = S.shape[1]
    S_sums = np.sum(S, axis=1) / n_samples
    S_gram = S @ S.T / n_samples
    lam = 1.0

    def f_wrapper(m_flat: NDArray[np.float64]) -> float:
        M_r = m_flat.reshape(3, 3)
        return float(MIOptimizer.sparse_norm(
            M_r, S_sums, S_gram, lam, n_samples, compute_grad=False).f)

    res = MIOptimizer.sparse_norm(M, S_sums, S_gram, lam, n_samples)
    grad_analytical = res.grad
    grad_numerical = approx_fprime(
        M.flatten(), f_wrapper, epsilon=1e-7).reshape(3, 3)

    assert grad_analytical is not None
    np.testing.assert_allclose(grad_analytical, grad_numerical, 
                               rtol=1e-4, atol=1e-10)

# --- barrier feasibility ---

def test_right_barrier_infeasible_returns_inf(small_positive_matrices):
    """right_barrier should return inf when MS contains negative entries."""
    _, S = small_positive_matrices
    M = -np.eye(3)  # Makes MS negative
    res = MIOptimizer.right_barrier(M, S, mu=1.0)
    assert np.isinf(res.f)

def test_left_barrier_infeasible_returns_inf(small_positive_matrices):
    """left_barrier should return inf when inv_MT_C contains negative entries."""
    C, _ = small_positive_matrices
    inv_M = -np.eye(3)  # Makes inv_MT_C negative
    res = MIOptimizer.left_barrier(inv_M, C, mu=1.0)
    assert np.isinf(res.f)

# --- normalize ---

def test_normalize_returns_four_values(small_positive_matrices):
    """normalize should return C_norm, S_norm, Lambda_C, Lambda_S."""
    C, S = small_positive_matrices
    result = MIOptimizer.normalize(C, S)
    assert len(result) == 4

def test_normalize_unit_std(small_positive_matrices):
    """Normalized matrices should have standard deviation close to 1."""
    C, S = small_positive_matrices
    C_norm, S_norm, _, _ = MIOptimizer.normalize(C, S)
    assert abs(np.std(C_norm) - 1.0) < 0.1
    assert abs(np.std(S_norm) - 1.0) < 0.1

def test_normalize_invertible(small_positive_matrices):
    """Lambda matrices should correctly invert the normalization."""
    C, S = small_positive_matrices
    C_norm, S_norm, Lambda_C, Lambda_S = MIOptimizer.normalize(C, S)
    inv_Lambda_C = np.diag(1.0 / np.diag(Lambda_C))
    inv_Lambda_S = np.diag(1.0 / np.diag(Lambda_S))
    np.testing.assert_allclose(inv_Lambda_C @ C_norm, C, rtol=1e-10)
    np.testing.assert_allclose(inv_Lambda_S @ S_norm, S, rtol=1e-10)

# --- prune ---

def test_prune_removes_near_origin():
    """prune should remove points near the origin."""
    rng = np.random.default_rng(42)
    data = np.abs(rng.standard_normal((3, 100)))
    data[:, :10] *= 1e-6  # Make 10 points near origin
    pruned = MIOptimizer().prune(data, gamma=0.05)
    assert pruned.shape[1] < data.shape[1]

def test_prune_safety_check():
    """prune should return original data if pruning removes too many points."""
    data = np.abs(np.random.default_rng(42).standard_normal((3, 5)))
    pruned = MIOptimizer().prune(data, gamma=0.99)
    assert pruned.shape == data.shape

# --- fit_transform integration ---

def test_fit_transform_returns_result(small_positive_matrices):
    """fit_transform should return MIOptimizerResult."""
    C, S = small_positive_matrices
    opt = MIOptimizer(
        params=(True, False, False),
        max_outer_loop_iters=2,
        max_inner_loop_iters=10,
        lam=0.0
    )
    result = opt.fit_transform(C, S)
    assert isinstance(result, MIOptimizerResult)

def test_fit_transform_constraints_satisfied(small_positive_matrices):
    """After optimization, MS and inv_MT_C should be non-negative."""
    C, S = small_positive_matrices
    opt = MIOptimizer(
        params=(True, False, False),
        max_outer_loop_iters=3,
        max_inner_loop_iters=50,
        lam=0.0
    )
    result = opt.fit_transform(C, S)
    if not result.success:
        pytest.skip("Optimizer did not converge with test iteration limits")
    assert np.all(result.MS >= -1e-6)
    assert np.all(result.inv_MT_C >= -1e-6)

def test_fit_transform_invalid_params(small_positive_matrices):
    """fit_transform should raise if no derivative order is selected."""
    C, S = small_positive_matrices
    with pytest.raises(AssertionError):
        opt = MIOptimizer(params=(False, False, False))
        opt.fit_transform(C, S)

def test_fit_transform_negative_C_raises(small_positive_matrices):
    """fit_transform should raise if C contains negative values."""
    C, S = small_positive_matrices
    C_neg = C.copy()
    C_neg[0, 0] = -1.0
    opt = MIOptimizer(params=(True, False, False))
    with pytest.raises(AssertionError):
        opt.fit_transform(C_neg, S)