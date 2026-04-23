import numpy as np
import pytest
from cnica.models import NMFParams, MIOParams, MIOptimizerResult

def test_nmfparams_defaults():
    p = NMFParams()
    assert p.init == 'nndsvd'
    assert p.beta_loss == 'kullback-leibler'
    assert p.max_iter == 100000  # match actual default
    assert p.tol == 1e-5         # match actual default
    assert p.random_state is None

def test_mioparams_defaults():
    p = MIOParams()
    assert p.params == (True, True, False)
    assert p.lam == 100.0        # match actual default
    assert p.tol == 1e-8         # match actual default
    assert p.sparsity_gate_threshold == 0.7  # match actual default

def test_mioparams_custom():
    p = MIOParams(params=(True, False, False), lam=0.0)
    assert p.params == (True, False, False)
    assert p.lam == 0.0

def test_mioptimizerresult_fields():
    """MIOptimizerResult should store all fields correctly."""
    n = 3
    result = MIOptimizerResult(
        x=np.eye(n),
        success=True,
        message="Converged.",
        fun=-1.0,
        jac=np.zeros((n, n)),
        nit=10,
        nit_outer=2,
        inv_MT_C=np.ones((n, 5)),
        MS=np.ones((n, 10)),
        C_const=np.ones((n, 5)),
        S_const=np.ones((n, 10)),
    )
    assert result.success is True
    assert result.nit == 10
    assert result.x.shape == (n, n)