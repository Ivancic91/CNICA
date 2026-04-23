import numpy as np
import pytest
from numpy.typing import NDArray
from cnica.filter import Filter

@pytest.fixture
def sine_signal() -> NDArray[np.float64]:
    """Simple sine wave for filter testing."""
    t = np.linspace(0, 10, 500)
    return np.sin(2 * np.pi * t).astype(np.float64)

def test_transform_order0_output_length(sine_signal):
    """Low-pass filter with trim=False should return same length as input."""
    filt = Filter(m=10, trim=False)
    out = filt.transform(sine_signal, order=0)
    assert len(out) == len(sine_signal)

def test_transform_order0_trim_output_length(sine_signal):
    """Low-pass filter with trim=True should return length n - 2m."""
    m = 10
    filt = Filter(m=m, trim=True)
    out = filt.transform(sine_signal, order=0)
    assert len(out) == len(sine_signal) - 2 * m

def test_transform_order1_output_length(sine_signal):
    """First derivative with trim=True should return length n - 2m."""
    m = 10
    filt = Filter(m=m, trim=True)
    out = filt.transform(sine_signal, order=1)
    assert len(out) == len(sine_signal) - 2 * m

def test_transform_order2_output_length(sine_signal):
    """Second derivative with trim=True should return length n - 2m."""
    m = 10
    filt = Filter(m=m, trim=True)
    out = filt.transform(sine_signal, order=2)
    assert len(out) == len(sine_signal) - 2 * m

def test_transform_empty_input():
    """Empty input should raise ValueError."""
    filt = Filter(m=10)
    with pytest.raises(ValueError, match="empty"):
        filt.transform(np.array([]), order=0)

def test_transform_too_short_for_trim():
    """Signal shorter than 2m+1 with trim=True should raise ValueError."""
    filt = Filter(m=10, trim=True)
    with pytest.raises(ValueError, match="too short"):
        filt.transform(np.ones(5), order=0)

def test_transform_invalid_order(sine_signal):
    """Unsupported order should raise ValueError."""
    filt = Filter(m=10)
    with pytest.raises(ValueError, match="Unsupported"):
        filt.transform(sine_signal, order=3)

def test_tune_returns_float(sine_signal):
    """tune() should return a float within the candidate range."""
    filt = Filter(m=10, trim=True)
    wc = filt.tune(sine_signal)
    assert isinstance(wc, float)
    assert filt.candidate_wc[0] <= wc <= filt.candidate_wc[-1]

def test_msrac_white_noise():
    """MSRAC of white noise should be near zero."""
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(1000).astype(np.float64)
    filt = Filter(m=10)
    msrac = filt._msrac(noise, max_lag=10)
    assert msrac < 0.1

def test_msrac_invalid_max_lag(sine_signal):
    """max_lag <= 0 should raise ValueError."""
    filt = Filter(m=10)
    with pytest.raises(ValueError, match="max_lag"):
        filt._msrac(sine_signal, max_lag=0)

def test_msrac_too_short():
    """Signal shorter than max_lag + 1 should raise ValueError."""
    filt = Filter(m=10)
    with pytest.raises(ValueError, match="too short"):
        filt._msrac(np.ones(5), max_lag=10)