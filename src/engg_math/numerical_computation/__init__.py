"""Root finding, differentiation, quadrature and interpolation with typed, explicit results."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._differentiation import derivative, gradient, jacobian
from ._errors import BracketError, ConvergenceError, DomainError
from ._interpolation import Interpolant
from ._quadrature import integrate_function, simpson, trapezoid
from ._results import QuadratureResult, RootResult, SystemRootResult
from ._roots import bisect, brent, newton, solve_system

try:
    __version__ = version("engg-math-numerical-computation")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "BracketError",
    "ConvergenceError",
    "DomainError",
    "Interpolant",
    "QuadratureResult",
    "RootResult",
    "SystemRootResult",
    "__version__",
    "bisect",
    "brent",
    "derivative",
    "gradient",
    "integrate_function",
    "jacobian",
    "newton",
    "simpson",
    "solve_system",
    "trapezoid",
]
