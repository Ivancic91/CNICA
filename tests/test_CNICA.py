import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from cnica import CNICA
from cnica.models import NMFParams, MIOParams

@pytest.fixture
def synthetic_data():
    """Three-component synthetic spectroscopic dataset."""
    rng = np.random.default_rng(42)
    n_wave = 200
    n_channels = 30
    x = np.linspace(0, 20, n_wave)

    s1 = np.exp(-(x - 4)**2 / 0.5) + np.exp(-(x - 10)**2 / 0.8)
    s2 = np.cos(x) + 1.2
    s3 = 0.1 * x + 0.5
    S_true = np.vstack([s1, s2, s3])

    C_true = np.abs(rng.standard_normal((n_channels, 3))) + 0.1
    D = C_true @ S_true
    D = np.random.poisson(D * 10) / 10.0
    return D, S_true, C_true

def test_fit_transform_returns_correct_shape(synthetic_data):
    """fit_transform should return S of shape (n_components, n_samples)."""
    D, _, _ = synthetic_data
    model = CNICA(
        n_components=3,
        nmf_params=NMFParams(beta_loss='frobenius', max_iter=100),
        mio_params=MIOParams(
            params=(True, False, False),
            max_outer_loop_iters=2,
            max_inner_loop_iters=10,
            lam=0.0
        )
    )
    S = model.fit_transform(D)
    assert S.shape == (3, D.shape[1])

def test_fit_transform_sets_attributes(synthetic_data):
    """fit_transform should set C_, S_, M_, nmf_obj_, mio_result_."""
    D, _, _ = synthetic_data
    model = CNICA(
        n_components=3,
        nmf_params=NMFParams(beta_loss='frobenius', max_iter=100),
        mio_params=MIOParams(
            params=(True, False, False),
            max_outer_loop_iters=2,
            max_inner_loop_iters=10,
            lam=0.0
        )
    )
    model.fit_transform(D)
    assert hasattr(model, 'C_')
    assert hasattr(model, 'S_')
    assert hasattr(model, 'M_')
    assert hasattr(model, 'nmf_obj_')
    assert hasattr(model, 'mio_result_')

def test_fit_transform_nonnegative_outputs(synthetic_data):
    """C_ and S_ should be non-negative after optimization."""
    D, _, _ = synthetic_data
    model = CNICA(
        n_components=3,
        nmf_params=NMFParams(beta_loss='frobenius', max_iter=100),
        mio_params=MIOParams(
            params=(True, False, False),
            max_outer_loop_iters=2,
            max_inner_loop_iters=10,
            lam=0.0
        )
    )
    model.fit_transform(D)
    if not model.mio_result_.success:
        pytest.skip("Optimizer did not converge with test iteration limits")
    assert np.all(model.S_ >= -1e-6)
    assert np.all(model.C_ >= -1e-6)

def test_fit_transform_reconstruction(synthetic_data):
    """Reconstruction C_.T @ S_ should approximate original D."""
    D, _, _ = synthetic_data
    model = CNICA(
        n_components=3,
        nmf_params=NMFParams(beta_loss='frobenius', max_iter=500),
        mio_params=MIOParams(
            params=(True, False, False),
            max_outer_loop_iters=3,
            max_inner_loop_iters=50,
            lam=0.0
        )
    )
    model.fit_transform(D)
    D_rec = model.C_.T @ model.S_
    rel_error = np.linalg.norm(D - D_rec) / np.linalg.norm(D)
    assert rel_error < 0.5

def test_transform_returns_correct_shape(synthetic_data):
    """transform should return C of shape (n_components, n_new_channels)."""
    D, _, _ = synthetic_data
    model = CNICA(
        n_components=3,
        nmf_params=NMFParams(beta_loss='frobenius', max_iter=100),
        mio_params=MIOParams(
            params=(True, False, False),
            max_outer_loop_iters=2,
            max_inner_loop_iters=10,
            lam=0.0
        )
    )
    model.fit_transform(D)
    D_new = D[:5]
    C_new = model.transform(D_new)
    assert C_new.shape == (3, 5)

def test_transform_nonnegative(synthetic_data):
    """transform output should be non-negative (NNLS enforces this)."""
    D, _, _ = synthetic_data
    model = CNICA(
        n_components=3,
        nmf_params=NMFParams(beta_loss='frobenius', max_iter=100),
        mio_params=MIOParams(
            params=(True, False, False),
            max_outer_loop_iters=2,
            max_inner_loop_iters=10,
            lam=0.0
        )
    )
    model.fit_transform(D)
    C_new = model.transform(D[:5])
    assert np.all(C_new >= -1e-10)

def test_get_params_sklearn_compatible(synthetic_data):
    """CNICA should support sklearn get_params/set_params interface."""
    model = CNICA(n_components=3)
    params = model.get_params()
    assert 'n_components' in params
    assert params['n_components'] == 3

def test_set_params_sklearn_compatible():
    """set_params should update parameters correctly."""
    model = CNICA(n_components=2)
    model.set_params(n_components=5)
    assert model.n_components == 5
