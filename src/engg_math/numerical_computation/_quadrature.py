"""Integration of functions (adaptive) and of sampled data (trapezoid, Simpson)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import integrate

from engg_math.core import ShapeError

from ._errors import ConvergenceError, DomainError
from ._results import QuadratureResult

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import ScalarFunction

DEFAULT_TOL = 1.49e-8  # SciPy's quad default; stated here so it is explicit and testable
DEFAULT_LIMIT = 200
SIMPSON_MIN_SAMPLES = 3
QUAD_OUTPUT_WITH_MESSAGE = 4  # quad appends a message when it did not converge


def integrate_function(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    rtol: float = DEFAULT_TOL,
    atol: float = DEFAULT_TOL,
    limit: int = DEFAULT_LIMIT,
) -> QuadratureResult:
    """Adaptive Gauss-Kronrod integral of ``f`` on ``[a, b]`` (bounds may be infinite).

    Raises ``ConvergenceError`` if the requested accuracy was not reached within ``limit``
    subdivisions, rather than returning a value of unknown quality.
    """
    if math.isnan(a) or math.isnan(b):
        raise DomainError("integration bounds must not be NaN")
    if atol <= 0 or rtol <= 0 or limit < 1:
        raise DomainError(f"need atol, rtol > 0 and limit >= 1; got {atol}, {rtol}, {limit}")
    outcome: tuple[Any, ...] = integrate.quad(
        f, a, b, epsabs=atol, epsrel=rtol, limit=limit, full_output=1
    )
    value, error, info = outcome[0], outcome[1], outcome[2]
    if (
        len(outcome) >= QUAD_OUTPUT_WITH_MESSAGE
    ):  # SciPy appends a message when the routine did not converge
        raise ConvergenceError(f"integration on [{a}, {b}] failed: {outcome[3]}")
    return QuadratureResult(float(value), float(error), int(info["neval"]))


def _validate_samples(y: npt.ArrayLike, x: npt.ArrayLike, minimum: int) -> tuple[np.ndarray, ...]:
    ys, xs = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if ys.ndim != 1 or xs.shape != ys.shape:
        raise ShapeError(f"x and y must be 1-D and equal length, got {xs.shape} and {ys.shape}")
    if ys.size < minimum:
        raise ShapeError(f"need at least {minimum} samples, got {ys.size}")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise DomainError("samples must be finite")
    if not np.all(np.diff(xs) > 0):
        raise DomainError("x must be strictly increasing")
    return ys, xs


def trapezoid(y: npt.ArrayLike, x: npt.ArrayLike) -> float:
    """Composite trapezoid rule on samples ``(x, y)``; O(h^2) for smooth data."""
    ys, xs = _validate_samples(y, x, 2)
    return float(integrate.trapezoid(ys, xs))


def simpson(y: npt.ArrayLike, x: npt.ArrayLike) -> float:
    """Composite Simpson rule on samples ``(x, y)``; O(h^4) for smooth data, exact for cubics."""
    ys, xs = _validate_samples(y, x, SIMPSON_MIN_SAMPLES)
    return float(integrate.simpson(ys, x=xs))
