"""Shared type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

# ndarray dtype/shape vary per call site; callers narrow where it matters.
Array = npt.NDArray[Any]
ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[Array], Array]
ScalarField = Callable[[Array], float]

FloatArray = npt.NDArray[np.float64]
