---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Example Usage

This page demonstrates CNICA on a synthetic spectroscopic dataset with three
known pure component spectra: a peaky signal, a smooth oscillatory signal, and
a linear background. These are mixed with random non-negative concentration
profiles and corrupted with Poisson noise to simulate realistic measurement
conditions.

## Setup

```python
import numpy as np
from cnica import CNICA
from cnica.models import NMFParams, MIOParams

n_wave = 1000
n_channels = 100
n_components = 3
x = np.linspace(0, 20, n_wave)

s1 = np.exp(-(x - 4)**2 / 0.5) + np.exp(-(x - 10)**2 / 0.8) + np.exp(-(x - 15)**2 / 0.4)
s2 = np.cos(x) + 1.2
s3 = 0.1 * x + 0.5
S_true = np.vstack([s1, s2, s3])

np.random.seed(42)
C_true = np.random.gamma(shape=2.0, scale=1.0, size=(n_channels, n_components))
D_clean = C_true @ S_true
D_noisy = np.random.poisson(D_clean * 100) / 100
```

## Running CNICA

```python
model = CNICA(
    n_components=n_components,
    nmf_params=NMFParams(
        init="nndsvd",
        beta_loss="frobenius",
        max_iter=10000,
        tol=1e-9
    ),
    mio_params=MIOParams(
        params=(True, True, False),
        tol=1e-14,
        lam=10.0,
        max_outer_loop_iters=10,
        max_inner_loop_iters=1000
    )
)

S_est = model.fit_transform(D_noisy)
C_est = model.C_
```

`S_est` has shape `(n_components, n_wave)` and contains the estimated pure
component spectra. `C_est` has shape `(n_components, n_channels)` and contains
the estimated concentration profiles. The original data is approximately
reconstructed as `C_est.T @ S_est`.

## Convergence Information

The full optimization result is available via `model.mio_result_`:

```python
print(model.mio_result_.success)   # True if converged
print(model.mio_result_.message)   # Convergence message
print(model.mio_result_.nit)       # Total inner iterations
print(model.mio_result_.fun)       # Final objective value
```

## Applying to New Data

If you have new observed spectra and want to extract concentrations using
the pure component spectra already identified:

```python
D_new = ...  # shape (n_new_channels, n_wave)
C_new = model.transform(D_new)  # shape (n_components, n_new_channels)
```