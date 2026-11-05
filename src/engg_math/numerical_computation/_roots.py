"""Root finding: bracketing methods, scalar Newton, and damped Newton for systems."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from engg_math.core import ShapeError
from engg_math.linear_systems import DenseMatrix, RobustSolver

from ._differentiation import derivative
from ._differentiation import jacobian as _numerical_jacobian
from ._errors import BracketError, ConvergenceError, DomainError
from ._results import RootResult, SystemRootResult

if TYPE_CHECKING:
    from ._types import Array, ScalarFunction, VectorFunction

DEFAULT_XTOL = 1e-12
DEFAULT_RTOL = 4 * np.finfo(float).eps
DEFAULT_MAX_ITER = 100
MIN_STEP_FRACTION = 2.0**-30  # backtracking gives up below this fraction of the Newton step


def _check_tolerances(xtol: float, rtol: float, max_iter: int) -> None:
    if xtol <= 0 or rtol < DEFAULT_RTOL or max_iter < 1:
        raise DomainError(
            f"need xtol > 0, rtol >= {DEFAULT_RTOL:.3g}, max_iter >= 1; "
            f"got xtol={xtol}, rtol={rtol}, max_iter={max_iter}"
        )


def _bracketed(
    solver: str,
    f: ScalarFunction,
    a: float,
    b: float,
    xtol: float,
    rtol: float,
    max_iter: int,
) -> RootResult:
    _check_tolerances(xtol, rtol, max_iter)
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise DomainError(f"f must be finite at the bracket ends: f({a})={fa}, f({b})={fb}")
    if fa * fb > 0:
        raise BracketError(f"no sign change on [{a}, {b}]: f(a)={fa:.6g}, f(b)={fb:.6g}")
    scipy_solver = {"bisect": optimize.bisect, "brent": optimize.brentq}[solver]
    root, info = scipy_solver(
        f, a, b, xtol=xtol, rtol=rtol, maxiter=max_iter, full_output=True, disp=False
    )
    result = RootResult(
        root=float(root),
        residual=float(f(root)),
        iterations=int(info.iterations),
        converged=bool(info.converged),
        method=solver,
    )
    if not result.converged:
        raise ConvergenceError(f"{solver} did not converge in {max_iter} iterations on [{a}, {b}]")
    return result


def bisect(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    xtol: float = DEFAULT_XTOL,
    rtol: float = DEFAULT_RTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> RootResult:
    """Bisection on ``[a, b]``; linear convergence, never fails once bracketed."""
    return _bracketed("bisect", f, a, b, xtol, rtol, max_iter)


def brent(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    xtol: float = DEFAULT_XTOL,
    rtol: float = DEFAULT_RTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> RootResult:
    """Brent's method on ``[a, b]``; superlinear with bisection's guarantee."""
    return _bracketed("brent", f, a, b, xtol, rtol, max_iter)


def newton(
    f: ScalarFunction,
    x0: float,
    df: ScalarFunction | None = None,
    *,
    tol: float = DEFAULT_XTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> RootResult:
    """Newton's method from ``x0``; falls back to a central-difference slope without ``df``.

    Stops when the step is below ``tol * max(1, |x|)``. A zero slope raises
    ``ConvergenceError`` rather than dividing by it.
    """
    _check_tolerances(tol, DEFAULT_RTOL, max_iter)
    x = float(x0)
    for iteration in range(1, max_iter + 1):
        fx = f(x)
        slope = df(x) if df is not None else derivative(f, x)
        if slope == 0.0 or not math.isfinite(slope):
            raise ConvergenceError(f"newton: slope is {slope} at x={x} (iteration {iteration})")
        step = fx / slope
        x -= step
        if abs(step) <= tol * max(1.0, abs(x)):
            return RootResult(x, float(f(x)), iteration, True, "newton")
    raise ConvergenceError(f"newton did not converge in {max_iter} iterations from x0={x0}")


def solve_system(
    f: VectorFunction,
    x0: Array | list[float],
    jacobian: VectorFunction | None = None,
    *,
    tol: float = 1e-10,
    max_iter: int = 50,
    raise_on_failure: bool = True,
) -> SystemRootResult:
    """Damped Newton for ``f(x) = 0``; each step is a conditioning-checked linear *solve*.

    The step is halved until ``||f||`` decreases (Armijo-style backtracking). ``jacobian``
    returns the ``n x n`` matrix; it defaults to central differences. Converges when
    ``||f(x)||_2 <= tol``.
    """
    _check_tolerances(tol, DEFAULT_RTOL, max_iter)
    x = np.array(x0, dtype=float)
    if x.ndim != 1:
        raise ShapeError(f"x0 must be 1-D, got shape {x.shape}")
    residual = np.asarray(f(x), dtype=float)
    if residual.shape != x.shape:
        raise ShapeError(f"f maps shape {x.shape} to {residual.shape}; the system must be square")
    norm = float(np.linalg.norm(residual))
    for iteration in range(1, max_iter + 1):
        if norm <= tol:
            return SystemRootResult(x, norm, iteration - 1, True, "newton-damped")
        jac = (
            DenseMatrix(np.asarray(jacobian(x), dtype=float))
            if jacobian is not None
            else _numerical_jacobian(f, x)
        )
        step = RobustSolver(jac).solve(-residual).data[:, 0]
        scale = 1.0
        while scale >= MIN_STEP_FRACTION:
            trial = x + scale * step
            trial_residual = np.asarray(f(trial), dtype=float)
            trial_norm = float(np.linalg.norm(trial_residual))
            if trial_norm < norm:
                break
            scale /= 2.0
        else:
            return _system_failure(x, norm, iteration, "line search stalled", raise_on_failure)
        x, residual, norm = trial, trial_residual, trial_norm
    if norm <= tol:
        return SystemRootResult(x, norm, max_iter, True, "newton-damped")
    return _system_failure(x, norm, max_iter, f"{max_iter} iterations exhausted", raise_on_failure)


def _system_failure(
    x: Array,
    norm: float,
    iterations: int,
    reason: str,
    raise_on_failure: bool,
) -> SystemRootResult:
    if raise_on_failure:
        raise ConvergenceError(f"solve_system: {reason}; ||f||={norm:.3e}")
    return SystemRootResult(x, norm, iterations, False, "newton-damped")
