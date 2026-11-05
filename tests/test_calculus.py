"""Differentiation, quadrature, interpolation: reference solutions and errors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engg_math.core import ShapeError
from engg_math.linear_systems import DenseMatrix
from engg_math.numerical_computation import (
    ConvergenceError,
    DomainError,
    Interpolant,
    derivative,
    gradient,
    integrate_function,
    jacobian,
    simpson,
    trapezoid,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# --- differentiation -------------------------------------------------------------------
def test_central_derivative_of_sin() -> None:
    """Central error ~ eps^(2/3) ~ 4e-11 times a modest constant; 1e-9 is conservative."""
    assert derivative(math.sin, 0.7) == pytest.approx(math.cos(0.7), abs=1e-9)


def test_forward_derivative_is_less_accurate_but_close() -> None:
    assert derivative(math.sin, 0.7, scheme="forward") == pytest.approx(math.cos(0.7), abs=1e-6)


def test_derivative_rejects_bad_scheme_and_step() -> None:
    with pytest.raises(DomainError):
        derivative(math.sin, 0.0, scheme="backward")  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        derivative(math.sin, 0.0, h=-1.0)


@given(st.floats(-50, 50), st.floats(-10, 10))
@settings(max_examples=60)
def test_derivative_of_linear_function_is_slope(x: float, slope: float) -> None:
    """REQ-NC-001: d/dx (m x + c) = m for any x."""
    assert derivative(lambda t: slope * t + 3.0, x) == pytest.approx(slope, abs=1e-6)


def test_gradient_of_quadratic_form() -> None:
    grad = gradient(lambda v: v[0] ** 2 + 3 * v[0] * v[1], [1.0, 2.0])
    np.testing.assert_allclose(grad, [2 * 1 + 3 * 2, 3 * 1], atol=1e-8)


def test_jacobian_is_dense_matrix_with_correct_values() -> None:
    jac = jacobian(lambda v: np.array([v[0] * v[1], v[0] + v[1] ** 2]), [2.0, 3.0])
    assert isinstance(jac, DenseMatrix)
    np.testing.assert_allclose(jac.data, [[3.0, 2.0], [1.0, 6.0]], atol=1e-8)


def test_gradient_and_jacobian_reject_non_1d() -> None:
    with pytest.raises(ShapeError):
        gradient(lambda v: 0.0, np.ones((1, 1)))
    with pytest.raises(ShapeError):
        jacobian(lambda v: v, np.ones((1, 1)))


# --- quadrature ------------------------------------------------------------------------
def test_integrate_polynomial_exact() -> None:
    result = integrate_function(lambda x: x**2, 0.0, 3.0)
    assert result.value == pytest.approx(9.0, abs=1e-10)
    assert result.evaluations > 0


def test_integrate_infinite_interval() -> None:
    """Reference: integral of exp(-x^2) over R is sqrt(pi)."""
    result = integrate_function(lambda x: math.exp(-x * x), -math.inf, math.inf)
    assert result.value == pytest.approx(math.sqrt(math.pi), abs=1e-7)


def test_integrate_reports_non_convergence() -> None:
    with pytest.raises(ConvergenceError):
        integrate_function(lambda x: math.sin(1 / x) if x else 0.0, 0.0, 1.0, limit=2, rtol=1e-12)


def test_integrate_rejects_bad_arguments() -> None:
    with pytest.raises(DomainError):
        integrate_function(math.sin, math.nan, 1.0)
    with pytest.raises(DomainError):
        integrate_function(math.sin, 0.0, 1.0, atol=0.0)


def test_trapezoid_and_simpson_reference() -> None:
    x = np.linspace(0.0, 1.0, 101)
    assert trapezoid(x**2, x) == pytest.approx(1 / 3, abs=1e-4)
    assert simpson(x**3, x) == pytest.approx(0.25, abs=1e-12)  # Simpson is exact for cubics


@pytest.mark.parametrize("rule", [trapezoid, simpson])
def test_sample_rules_validate_input(rule: Callable[..., float]) -> None:
    with pytest.raises(ShapeError):
        rule([1.0, 2.0], [0.0, 1.0, 2.0])
    with pytest.raises(ShapeError):
        rule([1.0], [0.0])
    with pytest.raises(DomainError):
        rule([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    with pytest.raises(DomainError):
        rule([1.0, math.nan, 3.0], [0.0, 1.0, 2.0])


# --- interpolation ---------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["linear", "cubic", "pchip"])
def test_interpolant_reproduces_knots(kind: str) -> None:
    x = np.array([0.0, 1.0, 2.5, 4.0])
    y = np.array([1.0, -1.0, 0.5, 3.0])
    np.testing.assert_allclose(Interpolant(x, y, kind)(x), y, atol=1e-12)  # type: ignore[arg-type]


def test_cubic_is_exact_for_cubic_data() -> None:
    """Not-a-knot spline reproduces cubic polynomials exactly."""
    x = np.linspace(-1, 2, 8)
    interp = Interpolant(x, x**3 - x, "cubic")
    assert interp(0.37) == pytest.approx(0.37**3 - 0.37, abs=1e-12)
    assert interp.derivative(0.37) == pytest.approx(3 * 0.37**2 - 1, abs=1e-10)


def test_pchip_does_not_overshoot_monotone_data() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.1, 5.0, 5.1])
    values = Interpolant(x, y, "pchip")(np.linspace(0, 3, 200))
    assert values.min() >= 0.0 - 1e-12
    assert values.max() <= 5.1 + 1e-12


def test_interpolant_refuses_extrapolation_by_default() -> None:
    interp = Interpolant([0.0, 1.0], [0.0, 1.0])
    assert interp.domain == (0.0, 1.0)
    with pytest.raises(DomainError):
        interp(1.5)
    assert Interpolant([0.0, 1.0], [0.0, 1.0], extrapolate=True)(1.5) == pytest.approx(1.5)
    assert "linear" in repr(interp)


def test_interpolant_validates_construction() -> None:
    with pytest.raises(ShapeError):
        Interpolant([0.0, 1.0], [0.0])
    with pytest.raises(ShapeError):
        Interpolant([0.0], [0.0])
    with pytest.raises(DomainError):
        Interpolant([1.0, 0.0], [0.0, 1.0])
    with pytest.raises(DomainError):
        Interpolant([0.0, math.inf], [0.0, 1.0])
    with pytest.raises(DomainError):
        Interpolant([0.0, 1.0], [0.0, 1.0], "quintic")  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        Interpolant([0.0, 1.0], [0.0, 1.0]).derivative(0.5, order=0)
