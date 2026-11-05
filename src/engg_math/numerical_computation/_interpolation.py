"""One-dimensional interpolation with explicit extrapolation behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import interpolate

from engg_math.core import ShapeError

from ._errors import DomainError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._types import Array

Kind = Literal["linear", "cubic", "pchip"]


def _build(kind: Kind, x: Array, y: Array) -> Any:  # noqa: ANN401 - SciPy spline types share no base
    if kind == "linear":
        return interpolate.make_interp_spline(x, y, k=1)
    if kind == "cubic":
        return interpolate.CubicSpline(x, y, bc_type="not-a-knot")
    if kind == "pchip":
        return interpolate.PchipInterpolator(x, y)
    raise DomainError(f"kind must be 'linear', 'cubic' or 'pchip', got {kind!r}")


class Interpolant:
    """Interpolating function through ``(x, y)``.

    ``linear`` is C0; ``cubic`` is the C2 not-a-knot spline; ``pchip`` is C1 and
    monotonicity-preserving (no overshoot between samples). Evaluating outside the data
    range raises ``DomainError`` unless ``extrapolate=True``.
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        kind: Kind = "linear",
        *,
        extrapolate: bool = False,
    ) -> None:
        """Validate the samples and build the interpolant."""
        xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if xs.ndim != 1 or xs.shape != ys.shape:
            raise ShapeError(f"x and y must be 1-D and equal length, got {xs.shape}, {ys.shape}")
        if xs.size < 2:  # noqa: PLR2004
            raise ShapeError(f"need at least 2 samples, got {xs.size}")
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            raise DomainError("samples must be finite")
        if not np.all(np.diff(xs) > 0):
            raise DomainError("x must be strictly increasing")
        self.kind: Kind = kind
        self.extrapolate = extrapolate
        self._spline = _build(kind, xs, ys)
        self._domain = (float(xs[0]), float(xs[-1]))

    @property
    def domain(self) -> tuple[float, float]:
        """``(x_min, x_max)`` of the data."""
        return self._domain

    def _check(self, x: npt.ArrayLike) -> Array:
        points = np.asarray(x, dtype=float)
        low, high = self._domain
        if not self.extrapolate and (np.any(points < low) or np.any(points > high)):
            raise DomainError(
                f"query outside the data range [{low}, {high}]; pass extrapolate=True to allow it"
            )
        return points

    def __call__(self, x: npt.ArrayLike) -> Array:
        """Evaluate at ``x`` (scalar or array)."""
        result: Array = self._spline(self._check(x))
        return result

    def derivative(self, x: npt.ArrayLike, order: int = 1) -> Array:
        """``order``-th derivative of the interpolant at ``x``."""
        if order < 1:
            raise DomainError(f"order must be >= 1, got {order}")
        result: Array = self._spline.derivative(order)(self._check(x))
        return result

    def __repr__(self) -> str:
        """Unambiguous summary."""
        return (
            f"Interpolant(kind={self.kind!r}, domain={self._domain}, "
            f"extrapolate={self.extrapolate})"
        )
