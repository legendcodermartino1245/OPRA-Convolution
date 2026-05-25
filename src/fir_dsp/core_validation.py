from __future__ import annotations

from typing import Any

import numpy as np

from .validation import (
    ensure_1d_finite_array,
    ensure_bool,
    ensure_non_negative_array,
    ensure_numeric_scalar,
    ensure_positive_int,
    ensure_positive_sample_rate,
)

EPS = 1e-20
SILENCE_FLOOR = 1e-12
RESPONSE_FLOOR_RELATIVE_TO_PEAK = 1e-10
RESPONSE_ERROR_DB_FLOOR_BELOW_PEAK = 120.0


def _as_float64_1d(h: np.ndarray | list[float]) -> np.ndarray:
    return ensure_1d_finite_array("h", h)


def db(x: float) -> float:
    x = ensure_numeric_scalar("x", x)
    return float(20.0 * np.log10(max(abs(x), EPS)))


def evaluate(
    name: str,
    value: float | None,
    good_threshold: float,
    warn_threshold: float,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    if value is None:
        return {"metric": name, "value": None, "status": "SKIPPED"}
    value = ensure_numeric_scalar(f"{name} value", value)
    good_threshold = ensure_numeric_scalar("good_threshold", good_threshold)
    warn_threshold = ensure_numeric_scalar("warn_threshold", warn_threshold)
    lower_is_better = ensure_bool("lower_is_better", lower_is_better)

    if lower_is_better:
        if not good_threshold < warn_threshold:
            raise ValueError("good_threshold must be < warn_threshold when lower_is_better=True")
        if value <= good_threshold:
            status = "PASS"
        elif value <= warn_threshold:
            status = "WARN"
        else:
            status = "FAIL"
    else:
        if not good_threshold > warn_threshold:
            raise ValueError("good_threshold must be > warn_threshold when lower_is_better=False")
        if value >= good_threshold:
            status = "PASS"
        elif value >= warn_threshold:
            status = "WARN"
        else:
            status = "FAIL"

    return {"metric": name, "value": float(value), "status": status}


def gain_linearity(h: np.ndarray) -> dict[str, Any]:
    h64 = _as_float64_1d(h)
    denom = float(np.max(np.abs(h64))) if h64.size else 0.0
    if denom < SILENCE_FLOOR:
        return {"metric": "gain_linearity_error", "value": None, "status": "SKIPPED"}

    scaled = h64 * 0.5
    ratio = float(np.max(np.abs(scaled)) / denom)
    return evaluate("gain_linearity_error", abs(ratio - 0.5), 1e-6, 1e-3, lower_is_better=True)


def energy_front_ratio(h: np.ndarray, sample_rate: int = 48_000) -> dict[str, Any]:
    h64 = _as_float64_1d(h)
    sample_rate = ensure_positive_sample_rate(sample_rate)
    split = min(int(sample_rate * (2.0 / 1000.0)), len(h64))
    peak = float(np.max(np.abs(h64))) if h64.size else 0.0
    if peak < SILENCE_FLOOR:
        return {"metric": "energy_front_ratio", "value": None, "status": "SKIPPED"}
    scaled = h64 / peak
    total_energy = float(np.sum(np.square(scaled)))
    if total_energy < SILENCE_FLOOR:
        return {"metric": "energy_front_ratio", "value": None, "status": "SKIPPED"}

    ratio = float(np.sum(np.square(scaled[:split])) / total_energy)
    return evaluate("energy_front_ratio", ratio, 0.85, 0.6, lower_is_better=False)


def peak_check(h: np.ndarray) -> dict[str, Any]:
    h64 = _as_float64_1d(h)
    peak = float(np.max(np.abs(h64))) if h64.size else 0.0
    return evaluate("peak_amplitude", peak, 1.001, 1.01, lower_is_better=True)


def frequency_response_error(h: np.ndarray, target_mag: np.ndarray | list[float], fft_size: int) -> dict[str, Any]:
    fft_size = ensure_positive_int("fft_size", fft_size)
    if fft_size < 2 or fft_size % 2 != 0:
        raise ValueError("fft_size must be an even integer >= 2")

    h64 = _as_float64_1d(h)
    if fft_size < h64.size:
        raise ValueError(f"fft_size must be >= FIR length ({h64.size})")
    target = ensure_non_negative_array("target_mag", target_mag)
    expected_len = fft_size // 2 + 1
    if target.size != expected_len:
        raise ValueError(
            f"target_mag length {target.size} does not match FFT size {expected_len}"
        )

    h_peak = float(np.max(np.abs(h64))) if h64.size else 0.0
    if h_peak > 0.0 and h_peak > np.finfo(np.float64).max / float(max(fft_size, 1)):
        raise ValueError("freq_response_error_db frequency response contains non-finite values")
    mag_raw = np.abs(np.fft.rfft(h64, n=fft_size))
    if not np.all(np.isfinite(mag_raw)):
        raise ValueError("freq_response_error_db frequency response contains non-finite values")
    target_peak = float(np.max(np.abs(target))) if target.size else 0.0
    mag_peak = float(np.max(mag_raw)) if mag_raw.size else 0.0
    if target_peak <= 0.0:
        if mag_peak <= SILENCE_FLOOR:
            return evaluate("freq_response_error_db", 0.0, 0.1, 0.5, lower_is_better=True)
        return {
            "metric": "freq_response_error_db",
            "value": 240.0,
            "status": "FAIL",
            "reason": "silent target produced non-silent response",
        }

    mag = np.maximum(mag_raw, EPS)
    target = np.maximum(target, EPS)
    peak_mag = float(max(np.max(mag), np.max(target), EPS))
    relevant_mask = np.maximum(mag, target) >= peak_mag * RESPONSE_FLOOR_RELATIVE_TO_PEAK
    if not np.any(relevant_mask):
        relevant_mask = np.ones_like(mag, dtype=bool)
    mag = mag[relevant_mask]
    target = target[relevant_mask]
    alignment_scale = max(float(np.max(mag)), float(np.max(target)), EPS)
    mag_scaled = mag / alignment_scale
    target_scaled = target / alignment_scale
    denom = float(np.dot(mag_scaled, mag_scaled))
    gain = float(np.dot(target_scaled, mag_scaled) / denom) if denom > 0.0 else 1.0
    if not np.isfinite(gain) or gain <= 0.0:
        gain = 1.0
    mag_aligned = mag * gain
    peak_mag = float(max(np.max(mag_aligned), np.max(target), EPS))
    db_floor = peak_mag * (10.0 ** (-RESPONSE_ERROR_DB_FLOOR_BELOW_PEAK / 20.0))
    mag_aligned = np.maximum(mag_aligned, db_floor)
    target = np.maximum(target, db_floor)
    error = float(np.max(np.abs(20.0 * np.log10(mag_aligned) - 20.0 * np.log10(target))))
    return evaluate("freq_response_error_db", error, 0.1, 0.5, lower_is_better=True)


def run_core_validations(
    h: np.ndarray | list[float],
    mode: str = "default",
    sample_rate: int = 48_000,
    target_mag: np.ndarray | list[float] | None = None,
    fft_size: int | None = None,
) -> list[dict[str, Any]]:
    h64 = _as_float64_1d(h)
    sample_rate = ensure_positive_sample_rate(sample_rate)
    results: list[dict[str, Any]] = [
        gain_linearity(h64),
        energy_front_ratio(h64, sample_rate=sample_rate),
        peak_check(h64),
    ]

    if target_mag is not None and fft_size is not None:
        results.append(frequency_response_error(h64, target_mag, fft_size))
    else:
        results.append(
            {
                "metric": "freq_response_error_db",
                "value": None,
                "status": "SKIPPED",
            }
        )

    for result in results:
        result["mode"] = mode

    return results
