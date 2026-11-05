"""Finite-difference derivatives with machine-precision-optimal step sizes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from engg_math.core import ShapeError
from engg_math.linear_systems import DenseMatrix

from ._errors import DomainError

if TYPE_CHECKING:
    from ._types import Array, ScalarField, ScalarFunction, VectorFunction

EPS = float(np.finfo(float).eps)
# Truncation error ~ h^p and roundoff ~ eps/h balance at h ~ eps^(1/(p+1)).
CENTRAL_STEP = EPS ** (1 / 3)
FORWARD_STEP = EPS**0.5

Scheme = Literal["central", "forward"]


def _step(x: float, h: float | None, scheme: Scheme) -> float:
    """Step scaled to ``|x|``, rounded so that ``x + h`` is exactly representable."""
    if h is not None:
        if h <= 0 or not np.isfinite(h):
            raise DomainError(f"step h must be positive and finite, got {h}")
        return float(h)
    base = CENTRAL_STEP if scheme == "central" else FORWARD_STEP
    raw = base * max(1.0, abs(x))
    return float((x + raw) - x)


def derivative(
    f: ScalarFunction, x: float, *, h: float | None = None, scheme: Scheme = "central"
) -> float:
    """Differentiate ``f`` at ``x``: central (O(h^2)) or forward (O(h)) difference."""
    if scheme not in {"central", "forward"}:
        raise DomainError(f"scheme must be 'central' or 'forward', got {scheme!r}")
    step = _step(float(x), h, scheme)
    if scheme == "central":
        return (f(x + step) - f(x - step)) / (2.0 * step)
    return (f(x + step) - f(x)) / step


def gradient(f: ScalarField, x: Array | list[float], *, h: float | None = None) -> Array:
    """Central-difference gradient of a scalar field ``f: R^n -> R``."""
    point = np.array(x, dtype=float)
    if point.ndim != 1:
        raise ShapeError(f"x must be 1-D, got shape {point.shape}")
    grad = np.empty_like(point)
    for i in range(point.size):
        step = _step(point[i], h, "central")
        forward, backward = point.copy(), point.copy()
        forward[i] += step
        backward[i] -= step
        grad[i] = (f(forward) - f(backward)) / (2.0 * step)
    return grad


def jacobian(f: VectorFunction, x: Array | list[float], *, h: float | None = None) -> DenseMatrix:
    """Central-difference Jacobian of ``f: R^n -> R^m`` as a column-major ``DenseMatrix``."""
    point = np.array(x, dtype=float)
    if point.ndim != 1:
        raise ShapeError(f"x must be 1-D, got shape {point.shape}")
    columns = []
    for i in range(point.size):
        step = _step(point[i], h, "central")
        forward, backward = point.copy(), point.copy()
        forward[i] += step
        backward[i] -= step
        columns.append((np.asarray(f(forward)) - np.asarray(f(backward))) / (2.0 * step))
    return DenseMatrix(np.column_stack(columns))
