from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

ArrayLike1D = NDArray[np.float64] | list[float]


def _contains_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_bool(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_bool(item) for item in value)
    return False


def ensure_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")
    return bool(value)



def ensure_numeric_scalar(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value



def ensure_positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    if not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)



def ensure_non_negative_float(name: str, value: Any) -> float:
    numeric = ensure_numeric_scalar(name, value)
    if numeric < 0:
        raise ValueError(f"{name} must be >= 0")
    return numeric



def ensure_power_of_two(name: str, value: Any) -> int:
    checked = ensure_positive_int(name, value)
    if (checked & (checked - 1)) != 0:
        raise ValueError(f"{name} must be a power of two")
    return checked



def ensure_positive_sample_rate(sample_rate: Any) -> int:
    numeric = ensure_numeric_scalar("sample_rate", sample_rate)
    if numeric <= 0:
        raise ValueError("sample_rate must be positive")
    if not float(numeric).is_integer():
        raise ValueError("sample_rate must be an integer value")
    return int(numeric)



def ensure_choice(name: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return value



def ensure_1d_finite_array(name: str, values: ArrayLike1D, *, non_empty: bool = True) -> NDArray[np.float64]:
    if _contains_bool(values):
        raise TypeError(f"{name} must be numeric, not bool")
    raw_array = np.asarray(values)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if non_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array



def ensure_non_negative_array(name: str, values: ArrayLike1D, *, non_empty: bool = True) -> NDArray[np.float64]:
    array = ensure_1d_finite_array(name, values, non_empty=non_empty)
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array



def ensure_strictly_increasing_freqs(freqs_hz: ArrayLike1D) -> NDArray[np.float64]:
    freqs = ensure_non_negative_array("freqs_hz", freqs_hz)
    if freqs.size < 2:
        raise ValueError("At least two frequency points are required")
    if not np.all(np.diff(freqs) > 0):
        raise ValueError("freqs_hz must be strictly increasing")
    return freqs



def ensure_matching_lengths(name_a: str, a: NDArray[np.float64], name_b: str, b: NDArray[np.float64]) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must have the same shape")



def normalize_rates(rates: Iterable[int | float]) -> list[int]:
    normalized: list[int] = []
    for rate in rates:
        numeric = ensure_numeric_scalar("sample rate", rate)
        if numeric <= 0:
            raise ValueError("All sample rates must be positive")
        if not float(numeric).is_integer():
            raise ValueError("All sample rates must be integer values")
        normalized.append(int(numeric))
    if not normalized:
        raise ValueError("rates must contain at least one sample rate")
    return sorted(set(normalized))
