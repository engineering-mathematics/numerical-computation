# engg-math-numerical-computation

Root finding, numerical differentiation, quadrature and interpolation for `engg_math`. Thin, typed
layers over SciPy with explicit tolerances, typed errors and result objects that say how an iteration stopped.

```bash
pip install engg-math-numerical-computation
```

```python
from engg_math.numerical_computation import brent, solve_system, integrate_function

brent(lambda x: x**3 - 2, 0, 2).root  # 1.2599...
integrate_function(lambda x: x**2, 0, 3).value  # 9.0
solve_system(lambda v: [v[0] ** 2 + v[1] ** 2 - 1, v[0] - v[1]], [1.0, 0.5]).x
```

## Guarantees

| Guarantee | How |
|---|---|
| No silent non-convergence | Iterative routines raise `ConvergenceError`; results carry `converged`, `iterations`, `residual` |
| Solve, don't invert | Newton steps for systems use `engg_math.linear_systems.RobustSolver` (condition-checked) |
| Explicit tolerances | Every tolerance is a keyword argument with a documented default |
| Step sizes | Finite differences use eps^(1/3) (central) and eps^(1/2) (forward), scaled to `|x|` |
| No hidden extrapolation | `Interpolant` raises outside its data range unless `extrapolate=True` |

## API

| Area | Names |
|---|---|
| Roots | `bisect`, `brent`, `newton`, `solve_system` |
| Differentiation | `derivative`, `gradient`, `jacobian` (returns `DenseMatrix`) |
| Quadrature | `integrate_function`, `trapezoid`, `simpson` |
| Interpolation | `Interpolant` (`linear`, `cubic`, `pchip`) |
| Results/errors | `RootResult`, `SystemRootResult`, `QuadratureResult`, `ConvergenceError`, `BracketError`, `DomainError` |

Units are not tracked in 0.1: functions operate on plain floats and arrays.

## Development

```bash
uv sync && uv run pytest && uv run ruff check . && uv run mypy && uv run lint-imports
```
