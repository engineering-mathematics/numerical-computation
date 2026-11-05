"""Typed result objects. Every iterative routine reports how it stopped."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Array


@dataclass(frozen=True, slots=True)
class RootResult:
    """Outcome of a scalar root search."""

    root: float
    residual: float
    iterations: int
    converged: bool
    method: str


@dataclass(frozen=True, slots=True)
class SystemRootResult:
    """Outcome of a nonlinear system solve; ``residual_norm`` is the 2-norm of ``f(x)``."""

    x: Array
    residual_norm: float
    iterations: int
    converged: bool
    method: str


@dataclass(frozen=True, slots=True)
class QuadratureResult:
    """Integral estimate with the integrator's own absolute error estimate."""

    value: float
    error_estimate: float
    evaluations: int
