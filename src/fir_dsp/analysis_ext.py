from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from .validation import (
    ensure_1d_finite_array,
    ensure_bool,
    ensure_non_negative_float,
    ensure_numeric_scalar,
    ensure_positive_int,
    ensure_positive_sample_rate,
)


def _as_float64_1d(h: np.ndarray) -> np.ndarray:
    h64 = ensure_1d_finite_array("h", h)
    if h64.ndim != 1:
        raise ValueError("h must be a 1D array")
    return h64


def _fft_convolve_same_numpy_length(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    full = fftconvolve(signal, h, mode="full")
    target_len = max(signal.size, h.size)
    start = (min(signal.size, h.size) - 1) // 2
    return np.asarray(full[start : start + target_len], dtype=np.float64)


def _scaled_square(h: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(h))) if h.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(h, dtype=np.float64)
    return np.square(h / peak)


def db(x: float) -> float:
    x = ensure_numeric_scalar("x", x)
    return float(20.0 * np.log10(max(abs(x), 1e-20)))


def evaluate(
    name: str,
    value: float,
    pass_threshold: float,
    warn_threshold: float,
    lower_is_better: bool = True,
) -> dict[str, float | str]:
    value = ensure_numeric_scalar("value", value)
    pass_threshold = ensure_numeric_scalar("pass_threshold", pass_threshold)
    warn_threshold = ensure_numeric_scalar("warn_threshold", warn_threshold)
    lower_is_better = ensure_bool("lower_is_better", lower_is_better)

    if lower_is_better:
        if not pass_threshold < warn_threshold:
            raise ValueError("pass_threshold must be < warn_threshold when lower_is_better=True")
        if value <= pass_threshold:
            status = "PASS"
        elif value <= warn_threshold:
            status = "WARN"
        else:
            status = "FAIL"
    else:
        if not pass_threshold > warn_threshold:
            raise ValueError("pass_threshold must be > warn_threshold when lower_is_better=False")
        if value >= pass_threshold:
            status = "PASS"
        elif value >= warn_threshold:
            status = "WARN"
        else:
            status = "FAIL"

    return {"metric": name, "value": value, "status": status}


def energy_decay(h: np.ndarray, fs: int) -> dict[str, float | str]:
    h = _as_float64_1d(h)
    fs = ensure_positive_sample_rate(fs)
    energy = np.cumsum(_scaled_square(h)[::-1])[::-1]
    max_energy = float(np.max(energy))
    if max_energy > 0.0:
        energy /= max_energy
    below = np.where(energy <= 0.01)[0]
    idx = int(below[0]) if below.size > 0 else len(h) - 1
    t_ms = (idx / float(fs)) * 1000.0
    return evaluate("energy_time_to_99_ms", t_ms, 5.0, 20.0, lower_is_better=True)


def sweep_stress_peak(
    h: np.ndarray,
    fs: int,
    *,
    duration_s: float = 0.1,
    oversample_factor: int = 1,
) -> dict[str, float | str]:
    h = _as_float64_1d(h)
    fs = ensure_positive_sample_rate(fs)
    duration_s = ensure_numeric_scalar("duration_s", duration_s)
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0")
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    n = max(1, int(fs * duration_s))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float64)
    f1 = 20.0
    f2 = 0.45 * float(fs)
    sweep = np.sin(2.0 * np.pi * (f1 * t + ((f2 - f1) / (2.0 * duration_s)) * np.square(t)))
    y = _fft_convolve_same_numpy_length(sweep, h)
    if oversample_factor > 1:
        y = resample_poly(y, oversample_factor, 1)
    peak_db = db(np.max(np.abs(y)))
    return evaluate("sweep_stress_peak_dbfs", peak_db, -1.0, 0.0, lower_is_better=True)


def energy_window(h: np.ndarray, fs: int, window_ms: float) -> float:
    h = _as_float64_1d(h)
    fs = ensure_positive_sample_rate(fs)
    window_ms = ensure_non_negative_float("window_ms", window_ms)
    n = min(len(h), int(fs * window_ms / 1000.0))
    sq = _scaled_square(h)
    total = float(np.sum(sq))
    window = float(np.sum(sq[:n]))
    return (window / total) * 100.0 if total > 0.0 else 0.0


def analyze_ext(h: np.ndarray, fs: int, mode: str = "fast") -> list[dict[str, float | str]]:
    if mode not in {"fast", "strict"}:
        raise ValueError("mode must be 'fast' or 'strict'")
    fs = ensure_positive_sample_rate(fs)
    h64 = _as_float64_1d(h)

    energy_1ms = energy_window(h64, fs, 1.0)
    energy_0_1ms = energy_window(h64, fs, 0.1)
    stress_metric = sweep_stress_peak(
        h64,
        fs,
        duration_s=0.1,
        oversample_factor=4 if mode == "strict" else 1,
    )

    return [
        energy_decay(h64, fs),
        stress_metric,
        {"metric": "energy_within_1ms_pct", "value": energy_1ms, "status": "INFO"},
        {"metric": "energy_within_0_1ms_pct", "value": energy_0_1ms, "status": "INFO"},
    ]
