"""
Top level API (:mod:`CNICA`)
======================================================
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from cnica.core import CNICA
from cnica.models import MIOptimizerResult, NMFParams, MIOParams
from cnica.optimizer import MIOptimizer, FuncResult
from cnica.metrics import evaluate_cnica, EvaluationResult


try:
    __version__ = _version("CNICA")
except PackageNotFoundError:  # pragma: no cover
    # Package not installed via pip (e.g. running from source)
    __version__ = "999"


__author__ = """Robert J. S. Ivancic"""
__email__ = "ivancic91@gmail.com"


__all__ = [
    "CNICA",
    "MIOptimizer",
    "FuncResult",
    "MIOptimizerResult",
    "NMFParams",
    "MIOParams",
    "evaluate_cnica",
    "EvaluationResult",
    "__version__",
]


