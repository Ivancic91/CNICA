"""
Top level API (:mod:`CNICA`)
======================================================
"""

#from importlib.metadata import PackageNotFoundError
#from importlib.metadata import version as _version

from .cnica import CNICA, WNMF

__version__ = "0.0.1" 
#try:
#    __version__ = _version("CNICA")
#except PackageNotFoundError:  # pragma: no cover
#    __version__ = "999"


__author__ = """Robert J. S. Ivancic"""
__email__ = "ivancic91@gmail.com"


__all__ = [
    "__version__",
    "CNICA",
    "WNMF",
]


