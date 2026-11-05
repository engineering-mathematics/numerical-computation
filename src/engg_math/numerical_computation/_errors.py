"""Exception types."""

from __future__ import annotations

from engg_math.core import EnggMathError, NumericalError


class ConvergenceError(NumericalError):
    """An iteration did not reach its tolerance within the iteration budget."""


class BracketError(NumericalError, ValueError):
    """The interval does not bracket a sign change."""


class DomainError(EnggMathError, ValueError):
    """An argument lies outside the domain where the result is defined."""
