from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import build_fft_freq_grid, interpolate_log_frequency_response, linear_to_db
from .validation import ensure_1d_finite_array, ensure_non_negative_float, ensure_positive_int, ensure_positive_sample_rate

EPS = 1e-12


@dataclass(frozen=True)
class TargetValidationBandSummary:
    label: str
    min_freq_hz: float
    max_freq_hz: float
    bins_compared: int
    max_abs_error_db: float
    mean_abs_error_db: float
    rms_error_db: float
    p95_abs_error_db: float
    max_error_freq_hz: float


@dataclass(frozen=True)
class TargetValidationSummary:
    min_freq_hz: float
    max_freq_hz: float
    bins_compared: int
    max_abs_error_db: float
    mean_abs_error_db: float
    rms_error_db: float
    p95_abs_error_db: float
    max_error_freq_hz: float
    listening_band_summary: TargetValidationBandSummary | None = None
    band_summaries: tuple[TargetValidationBandSummary, ...] = ()


def _target_values_to_db(target_values: np.ndarray, target_scale: str) -> np.ndarray:
    target_values = ensure_1d_finite_array("target_values", target_values)
    if target_scale == "db":
        return np.asarray(target_values, dtype=np.float64)
    if target_scale == "linear":
        return linear_to_db(np.asarray(target_values, dtype=np.float64))
    raise ValueError("target_scale must be 'db' or 'linear'")


def _interpolate_target_to_fft_db(
    target_freqs_hz: np.ndarray,
    target_values: np.ndarray,
    target_scale: str,
    fir_freqs_hz: np.ndarray,
) -> np.ndarray:
    target_values = ensure_1d_finite_array("target_values", target_values)
    if target_scale == "db":
        return np.asarray(
            interpolate_log_frequency_response(target_freqs_hz, target_values, fir_freqs_hz),
            dtype=np.float64,
        )
    if target_scale == "linear":
        return linear_to_db(
            interpolate_log_frequency_response(target_freqs_hz, target_values, fir_freqs_hz)
        )
    raise ValueError("target_scale must be 'db' or 'linear'")


def _summarize_validation_band(
    *,
    label: str,
    freqs_hz: np.ndarray,
    error_db: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> TargetValidationBandSummary | None:
    band_mask = (freqs_hz >= lower_hz) & (freqs_hz <= upper_hz)
    if not np.any(band_mask):
        return None

    band_freqs_hz = freqs_hz[band_mask]
    band_error_db = error_db[band_mask]
    band_abs_error_db = np.abs(band_error_db)
    max_index = int(np.argmax(band_abs_error_db))
    return TargetValidationBandSummary(
        label=label,
        min_freq_hz=float(band_freqs_hz[0]),
        max_freq_hz=float(band_freqs_hz[-1]),
        bins_compared=int(band_mask.sum()),
        max_abs_error_db=float(np.max(band_abs_error_db)),
        mean_abs_error_db=float(np.mean(band_abs_error_db)),
        rms_error_db=float(np.sqrt(np.mean(np.square(band_error_db)))),
        p95_abs_error_db=float(np.percentile(band_abs_error_db, 95)),
        max_error_freq_hz=float(band_freqs_hz[max_index]),
    )


def validate_fir_against_target(
    fir: np.ndarray,
    sample_rate: int,
    target_freqs_hz: np.ndarray,
    target_values: np.ndarray,
    *,
    target_scale: str = "db",
    n_fft: int | None = None,
    min_freq_hz: float = 20.0,
    max_freq_hz: float | None = None,
) -> TargetValidationSummary:
    fir = ensure_1d_finite_array("fir", fir)
    sample_rate = ensure_positive_sample_rate(sample_rate)
    target_freqs_hz = ensure_1d_finite_array("target_freqs_hz", target_freqs_hz)
    target_values_db = _target_values_to_db(target_values, target_scale)

    if target_freqs_hz.shape != target_values_db.shape:
        raise ValueError("target_freqs_hz and target_values must have the same length")

    if n_fft is None:
        n_fft = int(fir.size)
    n_fft = ensure_positive_int("n_fft", n_fft)
    if n_fft < fir.size:
        raise ValueError(f"n_fft must be >= FIR length ({fir.size})")

    fir_freqs_hz = build_fft_freq_grid(n_fft, sample_rate)
    fir_peak = float(np.max(np.abs(fir))) if fir.size else 0.0
    if fir_peak > 0.0 and fir_peak > np.finfo(np.float64).max / float(max(n_fft, 1)):
        raise ValueError("fir frequency response contains non-finite values")
    fir_magnitude = np.abs(np.fft.rfft(fir, n=n_fft))
    if not np.all(np.isfinite(fir_magnitude)):
        raise ValueError("fir frequency response contains non-finite values")
    fir_mag_db = linear_to_db(fir_magnitude)

    target_interp_db = _interpolate_target_to_fft_db(
        target_freqs_hz,
        target_values,
        target_scale,
        fir_freqs_hz,
    )

    nyquist = float(sample_rate) / 2.0
    lower = ensure_non_negative_float("min_freq_hz", min_freq_hz)
    upper = nyquist if max_freq_hz is None else min(ensure_non_negative_float("max_freq_hz", max_freq_hz), nyquist)
    if lower > upper:
        raise ValueError("min_freq_hz must be <= max_freq_hz")

    mask = (fir_freqs_hz >= lower) & (fir_freqs_hz <= upper)
    if not np.any(mask):
        raise ValueError("No FFT bins fall inside the requested validation band")

    fir_mag_db = fir_mag_db[mask]
    target_interp_db = target_interp_db[mask]
    masked_freqs_hz = fir_freqs_hz[mask]

    gain_offset_db = float(np.mean(target_interp_db - fir_mag_db))
    fir_mag_db_aligned = fir_mag_db + gain_offset_db
    error_db = fir_mag_db_aligned - target_interp_db
    abs_error_db = np.abs(error_db)

    max_index = int(np.argmax(abs_error_db))
    band_edges_hz = (
        ("sub_bass_to_low", 20.0, 200.0),
        ("midrange", 200.0, 2_000.0),
        ("presence", 2_000.0, 10_000.0),
        ("air", 10_000.0, upper),
    )
    band_summaries = tuple(
        summary
        for summary in (
            _summarize_validation_band(
                label=label,
                freqs_hz=masked_freqs_hz,
                error_db=error_db,
                lower_hz=max(lower, band_lower),
                upper_hz=min(upper, band_upper),
            )
            for label, band_lower, band_upper in band_edges_hz
        )
        if summary is not None
    )
    listening_band_summary = _summarize_validation_band(
        label="listening_band",
        freqs_hz=masked_freqs_hz,
        error_db=error_db,
        lower_hz=max(lower, 20.0),
        upper_hz=min(upper, 10_000.0),
    )
    return TargetValidationSummary(
        min_freq_hz=lower,
        max_freq_hz=upper,
        bins_compared=int(mask.sum()),
        max_abs_error_db=float(np.max(abs_error_db)),
        mean_abs_error_db=float(np.mean(abs_error_db)),
        rms_error_db=float(np.sqrt(np.mean(np.square(error_db)))),
        p95_abs_error_db=float(np.percentile(abs_error_db, 95)),
        max_error_freq_hz=float(masked_freqs_hz[max_index]),
        listening_band_summary=listening_band_summary,
        band_summaries=band_summaries,
    )
