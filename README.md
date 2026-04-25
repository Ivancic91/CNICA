<!-- markdownlint-disable MD041 -->

<!-- prettier-ignore-start -->
[![Repo][repo-badge]][repo-link]
[![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: ruff][ruff-badge]][ruff-link]
[![uv][uv-badge]][uv-link]

<!--
  For more badges, see
  https://shields.io/category/other
  https://naereen.github.io/badges/
  [pypi-badge]: https://badge.fury.io/py/CNICA
-->

[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-link]: https://github.com/astral-sh/ruff
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv-link]: https://github.com/astral-sh/uv
[pypi-badge]: https://img.shields.io/pypi/v/CNICA
[pypi-link]: https://pypi.org/project/CNICA
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/CNICA/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/Ivancic91/CNICA
[conda-badge]: https://img.shields.io/conda/v/Ivancic91/CNICA
[conda-link]: https://anaconda.org/Ivancic91/CNICA
[license-badge]: https://img.shields.io/pypi/l/CNICA?color=informational
[license-link]: https://github.com/Ivancic91/CNICA/blob/main/LICENSE

<!-- other links -->

<!-- prettier-ignore-end -->

# `CNICA`

Coupled Non-negative Independent Component Analysis for blind source separation
of non-negative signals, with particular application to spectroscopic data.

## Overview

CNICA decomposes an observed data matrix `D` of shape `(n_channels, n_samples)`
into two non-negative factor matrices:

```math
X \approx C^T S
```

where `C` (shape: `n_components × n_channels`) is the mixing matrix (e.g.
concentration profiles) and `S` (shape: `n_components × n_samples`) is the
source matrix (e.g. pure component spectra).

The decomposition proceeds in two stages:

1. **NMF initialization** — Non-negative Matrix Factorization provides an
   initial estimate of `C` and `S`.
2. **Riemannian mutual information minimization** — A linear transformation `M`
   is found that minimizes statistical dependence between source components
   while strictly enforcing non-negativity of `MS` and `M⁻ᵀC` via log-barrier
   penalties on a Riemannian manifold. This recovers components that are
   physically distinct rather than merely mathematically orthogonal, which is
   the key advantage over standard NMF.

CNICA is particularly well-suited to spectroscopic datasets where:

- Components are non-negative by physical constraint (e.g. absorbance, emission
  intensity).
- Pure component signals are expected to be statistically independent.
- Derivatives of the spectra or concentration profiles carry additional
  discriminative information (e.g. sharp peaks vs. broad backgrounds).

## Installation

```bash
pip install cnica
```

Requires Python 3.10+ and depends on `numpy`, `scipy`, and `scikit-learn`.

## Quick start

```python
import numpy as np
from cnica import CNICA
from cnica.models import NMFParams, MIOParams

# Observed mixture matrix: 30 channels, 200 wavenumbers
D = np.abs(np.random.randn(30, 200))

model = CNICA(
    n_components=3,
    nmf_params=NMFParams(beta_loss='frobenius'),
    mio_params=MIOParams(params=(True, True, False))
)

S = model.fit_transform(D)  # Pure component spectra: (3, 200)
C = model.C_                 # Concentration profiles: (3, 30)
```

## Example usage

A complete worked example using synthetic spectroscopic data with three known
pure components (Gaussian peaks, cosine wave, linear background) is available
in the [documentation][docs-link].

```python
import numpy as np
from cnica import CNICA
from cnica.models import NMFParams, MIOParams

# Ground truth sources
n_wave, n_channels, n_components = 1000, 100, 3
x = np.linspace(0, 20, n_wave)

s1 = np.exp(-(x-4)**2/0.5) + np.exp(-(x-10)**2/0.8) + np.exp(-(x-15)**2/0.4)
s2 = np.cos(x) + 1.2
s3 = 0.1 * x + 0.5
S_true = np.vstack([s1, s2, s3])

# Random non-negative mixing matrix and noisy observations
np.random.seed(42)
C_true = np.random.gamma(shape=2.0, scale=1.0, size=(n_channels, n_components))
D_noisy = np.random.poisson(C_true @ S_true * 100) / 100

# Fit CNICA
model = CNICA(
    n_components=n_components,
    nmf_params=NMFParams(beta_loss='frobenius', max_iter=10000, tol=1e-9),
    mio_params=MIOParams(params=(True, True, False), lam=10.0)
)
S_est = model.fit_transform(D_noisy)

# Check convergence
print(model.mio_result_.success)   # True if converged
print(model.mio_result_.message)   # Convergence message

# Apply to new data (estimate concentrations given known spectra)
D_new = np.abs(np.random.randn(10, n_wave))
C_new = model.transform(D_new)     # shape: (3, 10)
```

## Features

- **Similar to scikit-learn API** — implements `BaseEstimator` and
  `TransformerMixin` for use in sklearn pipelines.
- **Two-stage optimization** — NMF initialization followed by Riemannian
  manifold optimization with log-barrier constraints.
- **Unsupervised filter tuning** — Parks-McClellan FIR filters are tuned
  automatically via Mean Squared Residual Autocorrelation (MSRAC) to compute
  signal derivatives for the mutual information objective.
- **Convex hull reduction** — barrier constraints are enforced only on convex
  hull vertices of the constraint matrices, reducing cost from O(samples) to
  O(vertices).
- **Sparsity regularization** — optional Safe-Plateau sparsity norm rewards
  peaky components (e.g. sharp Raman bands) while leaving dense components
  (e.g. broad fluorescence) unaffected.

## Status

This package is actively developed and used by the author. Please feel free to
open an issue or pull request for bug reports, feature requests, or suggestions.

<!-- end-docs -->

## Documentation

See the [documentation][docs-link] for the full API reference and worked
examples.


## License

This is free software. See [LICENSE][license-link].

## Related work

Publication eminent... 

## Contact

The author can be reached at <ivancic91@gmail.com>.

## Credits

This package was created using
[Cookiecutter](https://github.com/audreyr/cookiecutter) with the
[usnistgov/cookiecutter-nist-python](https://github.com/usnistgov/cookiecutter-nist-python)
template.
