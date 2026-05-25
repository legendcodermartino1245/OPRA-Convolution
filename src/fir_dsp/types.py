from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .validation import ensure_1d_finite_array, ensure_non_negative_array


@dataclass(frozen=True)
class LinearMagnitude:
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        checked = np.asarray(ensure_non_negative_array("linear magnitude", self.values), dtype=np.float64).copy()
        object.__setattr__(self, "values", checked)

    def as_array(self) -> NDArray[np.float64]:
        return self.values.copy()


@dataclass(frozen=True)
class DbMagnitude:
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        checked = np.asarray(ensure_1d_finite_array("dB magnitude", self.values), dtype=np.float64).copy()
        object.__setattr__(self, "values", checked)

    def as_array(self) -> NDArray[np.float64]:
        return self.values.copy()


LinearMagnitudeLike = LinearMagnitude | NDArray[np.float64] | list[float]
DbMagnitudeLike = DbMagnitude | NDArray[np.float64] | list[float]


def coerce_linear_magnitude(values: LinearMagnitudeLike) -> LinearMagnitude:
    if isinstance(values, LinearMagnitude):
        return values
    if isinstance(values, DbMagnitude):
        raise TypeError("Expected linear-domain magnitude, received dB-domain magnitude")
    return LinearMagnitude(values)  # type: ignore[arg-type]


def coerce_db_magnitude(values: DbMagnitudeLike) -> DbMagnitude:
    if isinstance(values, DbMagnitude):
        return values
    if isinstance(values, LinearMagnitude):
        raise TypeError("Expected dB-domain magnitude, received linear-domain magnitude")
    return DbMagnitude(values)  # type: ignore[arg-type]

