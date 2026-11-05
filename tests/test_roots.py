"""Root-finding: reference solutions, error paths."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from engg_math.core import ShapeError
from engg_math.numerical_computation import (
    BracketError,
    ConvergenceError,
    DomainError,
    RootResult,
    bisect,
    brent,
    newton,
    solve_system,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TOL = 1e-10


@pytest.mark.parametrize("method", [bisect, brent])
def test_bracketed_methods_find_cube_root_of_two(method: Callable[..., RootResult]) -> None:
    """Reference: 2**(1/3); xtol=1e-12 so error is well below TOL."""
    result = method(lambda x: x**3 - 2, 0.0, 2.0)
    assert result.converged
    assert result.root == pytest.approx(2 ** (1 / 3), abs=TOL)


@pytest.mark.parametrize("method", [bisect, brent])
def test_bracketed_methods_raise_without_sign_change(method: Callable[..., RootResult]) -> None:
    with pytest.raises(BracketError):
        method(lambda x: x * x + 1, -1.0, 1.0)


@pytest.mark.parametrize("method", [bisect, brent])
def test_bracketed_methods_reject_bad_input(method: Callable[..., RootResult]) -> None:
    with pytest.raises(DomainError):
        method(lambda x: x, math.nan, 1.0)
    with pytest.raises(DomainError):
        method(lambda x: x, -1.0, 1.0, xtol=0.0)


def test_bisect_raises_when_iterations_exhausted() -> None:
    with pytest.raises(ConvergenceError):
        bisect(lambda x: x - 1 / 3, 0.0, 1.0, xtol=1e-14, max_iter=3)


def test_newton_with_analytic_derivative() -> None:
    result = newton(lambda x: x * x - 2, 1.0, lambda x: 2 * x)
    assert result.root == pytest.approx(math.sqrt(2), abs=TOL)
    assert result.iterations < 10


def test_newton_with_numerical_derivative() -> None:
    result = newton(math.cos, 1.0)
    assert result.root == pytest.approx(math.pi / 2, abs=1e-8)


def test_newton_zero_slope_raises() -> None:
    with pytest.raises(ConvergenceError, match="slope"):
        newton(lambda x: x * x + 1, 0.0, lambda x: 2 * x)


def test_newton_raises_when_not_converging() -> None:
    with pytest.raises(ConvergenceError, match="did not converge"):
        newton(lambda x: x * x + 1, 0.5, lambda x: 2 * x, max_iter=5)


def test_solve_system_circle_line() -> None:
    """Reference: x = y = 1/sqrt(2)."""
    result = solve_system(lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 1, v[0] - v[1]]), [1.0, 0.5])
    assert result.converged
    np.testing.assert_allclose(result.x, [math.sqrt(0.5)] * 2, atol=1e-8)
    assert result.residual_norm <= 1e-10


def test_solve_system_with_analytic_jacobian() -> None:
    def f(v: np.ndarray) -> np.ndarray:
        return np.array([v[0] ** 2 + v[1] ** 2 - 1, v[0] - v[1]])

    def jac(v: np.ndarray) -> np.ndarray:
        return np.array([[2 * v[0], 2 * v[1]], [1.0, -1.0]])

    result = solve_system(f, [1.0, 0.5], jac)
    np.testing.assert_allclose(result.x, [math.sqrt(0.5)] * 2, atol=1e-8)


def test_solve_system_already_at_root_takes_zero_iterations() -> None:
    result = solve_system(lambda v: v - 1.0, [1.0, 1.0])
    assert result.iterations == 0
    assert result.converged


def test_solve_system_rejects_non_square_and_bad_shape() -> None:
    with pytest.raises(ShapeError):
        solve_system(lambda v: np.array([v[0]]), [1.0, 2.0])
    with pytest.raises(ShapeError):
        solve_system(lambda v: v, np.ones((1, 2)))


def test_solve_system_reports_failure_without_raising_when_asked() -> None:
    result = solve_system(lambda v: v**2 + 1.0, [2.0], max_iter=3, raise_on_failure=False)
    assert not result.converged
    with pytest.raises(ConvergenceError):
        solve_system(lambda v: v**2 + 1.0, [2.0], max_iter=3)


def test_solve_system_exhausted_iterations() -> None:
    with pytest.raises(ConvergenceError, match="iterations exhausted"):
        solve_system(lambda v: np.exp(v) - 1e-3, [5.0], max_iter=2)
