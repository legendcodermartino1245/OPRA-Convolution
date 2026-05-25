from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import DbMagnitude, LinearMagnitude, coerce_linear_magnitude
from .validation import ensure_numeric_scalar

_MIN_PREAMP_DB = -120.0
_MAX_PREAMP_DB = 60.0


def validate_preamp_db(preamp_db: object | None) -> float | None:
    if preamp_db is None:
        return None
    preamp_db = ensure_numeric_scalar("preamp_db", preamp_db)
    if not (_MIN_PREAMP_DB <= preamp_db <= _MAX_PREAMP_DB):
        raise ValueError(
            f"preamp_db must be within the safe range [{_MIN_PREAMP_DB:.0f}, {_MAX_PREAMP_DB:.0f}] dB"
        )
    return preamp_db


def apply_preamp_db(values: LinearMagnitude | NDArray[np.float64] | list[float], preamp_db: float | None) -> NDArray[np.float64]:
    """Apply preamp in the linear domain.

    Contract:
    - ``values`` must already be linear-domain magnitudes.
    - ``DbMagnitude`` is rejected explicitly.
    - Preamp is applied exactly once by the FIR pipeline.
    """
    if isinstance(values, DbMagnitude):
        raise TypeError("apply_preamp_db only accepts linear-domain magnitude")

    try:
        linear_values = coerce_linear_magnitude(values)
    except ValueError as exc:
        raise ValueError("apply_preamp_db requires linear-domain magnitude with non-negative values") from exc
    checked_preamp_db = validate_preamp_db(preamp_db)
    array = linear_values.as_array()
    if checked_preamp_db is None:
        return array

    linear_gain = float(10.0 ** (checked_preamp_db / 20.0))
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = np.asarray(array * linear_gain, dtype=np.float64)
    if not np.all(np.isfinite(adjusted)):
        raise ValueError("preamp produces non-finite linear magnitude")
    return adjusted
