from __future__ import annotations

from typing import Any

import numpy as np

from .preamp import validate_preamp_db
from .validation import ensure_non_negative_array, ensure_numeric_scalar, ensure_positive_sample_rate

_MIN_BAND_GAIN_DB = -120.0
_MAX_BAND_GAIN_DB = 60.0
_SUPPORTED_FILTER_SLOPES_DB_PER_OCTAVE = {12.0}


def _as_freq_grid(freqs_hz: np.ndarray | list[float]) -> np.ndarray:
    return ensure_non_negative_array("freqs_hz", freqs_hz)


def _validate_sample_rate(sample_rate: int | float) -> float:
    return float(ensure_positive_sample_rate(sample_rate))


def _validate_band_params(fc: Any, q: Any, fs: float) -> tuple[float, float]:
    fc = ensure_numeric_scalar("band frequency", fc)
    q = ensure_numeric_scalar("band q", q)
    if fc <= 0.0:
        raise ValueError("band frequency must be a positive finite value")
    if fc >= fs / 2.0:
        raise ValueError(f"band frequency must be below Nyquist ({fs / 2.0} Hz)")
    if q <= 0.0:
        raise ValueError("band q must be a positive finite value")
    return fc, q


def _validate_band_gain_db(gain_db: Any) -> float:
    gain = ensure_numeric_scalar("band gain_db", gain_db)
    if not (_MIN_BAND_GAIN_DB <= gain <= _MAX_BAND_GAIN_DB):
        raise ValueError(
            f"band gain_db must be within the safe range [{_MIN_BAND_GAIN_DB:.0f}, {_MAX_BAND_GAIN_DB:.0f}] dB"
        )
    return gain


def _validate_filter_slope(slope: Any) -> float:
    checked_slope = ensure_numeric_scalar("band slope", slope)
    if checked_slope not in _SUPPORTED_FILTER_SLOPES_DB_PER_OCTAVE:
        supported = ", ".join(f"{value:.0f}" for value in sorted(_SUPPORTED_FILTER_SLOPES_DB_PER_OCTAVE))
        raise ValueError(f"band slope must be one of {{{supported}}} dB/oct")
    return checked_slope


def _biquad_response(b: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)
    return (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)


def _normalize_biquad(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a0 = float(a[0])
    if not np.isfinite(a0) or abs(a0) < 1e-20:
        raise ValueError("biquad denominator is degenerate")
    return np.asarray(b / a0, dtype=np.float64), np.asarray(a / a0, dtype=np.float64)


def _peak(fc: float, gain_db: float, q: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    fc, q = _validate_band_params(fc, q, fs)
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q)

    b = np.array([1.0 + alpha * A, -2.0 * np.cos(w0), 1.0 - alpha * A], dtype=np.float64)
    a = np.array([1.0 + alpha / A, -2.0 * np.cos(w0), 1.0 - alpha / A], dtype=np.float64)
    return _normalize_biquad(b, a)


def _low_shelf(fc: float, gain_db: float, q: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    fc, q = _validate_band_params(fc, q, fs)
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q)
    cosw = np.cos(w0)
    sqrtA = np.sqrt(A)

    b = np.array(
        [
            A * ((A + 1.0) - (A - 1.0) * cosw + 2.0 * sqrtA * alpha),
            2.0 * A * ((A - 1.0) - (A + 1.0) * cosw),
            A * ((A + 1.0) - (A - 1.0) * cosw - 2.0 * sqrtA * alpha),
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            (A + 1.0) + (A - 1.0) * cosw + 2.0 * sqrtA * alpha,
            -2.0 * ((A - 1.0) + (A + 1.0) * cosw),
            (A + 1.0) + (A - 1.0) * cosw - 2.0 * sqrtA * alpha,
        ],
        dtype=np.float64,
    )
    return _normalize_biquad(b, a)


def _high_shelf(fc: float, gain_db: float, q: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    fc, q = _validate_band_params(fc, q, fs)
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q)
    cosw = np.cos(w0)
    sqrtA = np.sqrt(A)

    b = np.array(
        [
            A * ((A + 1.0) + (A - 1.0) * cosw + 2.0 * sqrtA * alpha),
            -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw),
            A * ((A + 1.0) + (A - 1.0) * cosw - 2.0 * sqrtA * alpha),
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            (A + 1.0) - (A - 1.0) * cosw + 2.0 * sqrtA * alpha,
            2.0 * ((A - 1.0) - (A + 1.0) * cosw),
            (A + 1.0) - (A - 1.0) * cosw - 2.0 * sqrtA * alpha,
        ],
        dtype=np.float64,
    )
    return _normalize_biquad(b, a)


def _low_pass(fc: float, slope: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    fc = ensure_numeric_scalar("band frequency", fc)
    if fc <= 0.0:
        raise ValueError("band frequency must be a positive finite value")
    if fc >= fs / 2.0:
        raise ValueError(f"band frequency must be below Nyquist ({fs / 2.0} Hz)")
    _validate_filter_slope(slope)

    # OPRA currently uses 12 dB/oct low-pass sections, which map to a
    # standard second-order Butterworth-style biquad.
    q = 1.0 / np.sqrt(2.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q)
    cosw = np.cos(w0)

    b = np.array(
        [
            (1.0 - cosw) / 2.0,
            1.0 - cosw,
            (1.0 - cosw) / 2.0,
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            1.0 + alpha,
            -2.0 * cosw,
            1.0 - alpha,
        ],
        dtype=np.float64,
    )
    return _normalize_biquad(b, a)


def eq_json_to_native_magnitude(
    eq: dict[str, Any],
    freqs_hz: np.ndarray | list[float],
    sample_rate: int | float,
) -> np.ndarray:
    if not isinstance(eq, dict):
        raise TypeError("eq must be a dictionary")
    if "data" not in eq or "parameters" not in eq["data"]:
        raise ValueError("eq must contain data.parameters")

    fs = _validate_sample_rate(sample_rate)
    freqs = _as_freq_grid(freqs_hz)
    nyquist = fs / 2.0
    if np.any(freqs > nyquist):
        raise ValueError(f"freqs_hz must not exceed Nyquist ({nyquist} Hz)")
    params = eq["data"]["parameters"]
    bands = params.get("bands", [])
    if not isinstance(bands, list):
        raise ValueError("eq data.parameters.bands must be a list")

    w = 2.0 * np.pi * freqs / fs
    H = np.ones_like(freqs, dtype=np.complex128)

    for band in bands:
        if not isinstance(band, dict):
            raise ValueError("each band must be a dictionary")
        band_type = band["type"]
        fc = band["frequency"]

        if band_type == "peak_dip":
            gain_db = _validate_band_gain_db(band.get("gain_db", 0.0))
            q = band.get("q", 0.7)
            b, a = _peak(fc, gain_db, q, fs)
        elif band_type == "low_shelf":
            gain_db = _validate_band_gain_db(band.get("gain_db", 0.0))
            q = band.get("q", 0.7)
            b, a = _low_shelf(fc, gain_db, q, fs)
        elif band_type == "high_shelf":
            gain_db = _validate_band_gain_db(band.get("gain_db", 0.0))
            q = band.get("q", 0.7)
            b, a = _high_shelf(fc, gain_db, q, fs)
        elif band_type == "low_pass":
            b, a = _low_pass(fc, band.get("slope", 12.0), fs)
        else:
            raise ValueError(f"Unsupported band type: {band_type}")

        with np.errstate(over="ignore", invalid="ignore"):
            H *= _biquad_response(b, a, w)
        if not np.all(np.isfinite(H)):
            raise ValueError("eq response produces non-finite magnitudes")

    source_preamp_db = validate_preamp_db(params.get("gain_db", 0.0))
    with np.errstate(over="ignore", invalid="ignore"):
        H *= 10.0 ** (float(source_preamp_db or 0.0) / 20.0)
        magnitude = np.asarray(np.abs(H), dtype=np.float64)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("eq response produces non-finite magnitudes")
    return magnitude
