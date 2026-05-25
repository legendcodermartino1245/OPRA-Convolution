from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import resample_poly
from scipy.signal import firls

from .models import LatencySummary, ResponseErrorSummary, VerificationSummary, WindowSpec
from .validation import (
    ensure_1d_finite_array,
    ensure_bool,
    ensure_matching_lengths,
    ensure_non_negative_array,
    ensure_non_negative_float,
    ensure_numeric_scalar,
    ensure_positive_int,
    ensure_positive_sample_rate,
    ensure_strictly_increasing_freqs,
)

WINDOW_PRESETS: dict[str, WindowSpec] = {
    "safe": WindowSpec(name="kaiser", beta=8.6, preset="safe"),
    "sharp": WindowSpec(name="kaiser", beta=14.0, preset="sharp"),
    "minimal_ringing": WindowSpec(name="hann", beta=None, preset="minimal_ringing"),
}

EPS = 1e-12
LOG_EPS_HZ = 1e-6
TRUE_PEAK_MARGIN_TOLERANCE_DB = 1e-6
FLOAT64_MAX_DB = float(20.0 * np.log10(np.finfo(np.float64).max))
MINIMUM_PHASE_RELATIVE_MAGNITUDE_FLOOR = EPS
MINIMUM_PHASE_MAGNITUDE_RTOL = 1e-7


@lru_cache(maxsize=512)
def _cached_fft_freq_grid(fft_size: int, sample_rate: int) -> np.ndarray:
    freqs = np.asarray(np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate)), dtype=np.float64)
    freqs.setflags(write=False)
    return freqs


def build_fft_freq_grid(fft_size: int, sample_rate: int) -> np.ndarray:
    fft_size = ensure_positive_int("fft_size", fft_size)
    if fft_size % 2 != 0:
        raise ValueError("fft_size must be a positive even integer")
    sample_rate = ensure_positive_sample_rate(sample_rate)
    return _cached_fft_freq_grid(fft_size, sample_rate)


def db_to_linear(db_values: np.ndarray) -> np.ndarray:
    db_values = ensure_1d_finite_array("db_values", db_values)
    if db_values.size and float(np.max(db_values)) > FLOAT64_MAX_DB:
        raise ValueError("db_values produce non-finite linear magnitudes")
    linear = np.asarray(10.0 ** (db_values / 20.0), dtype=np.float64)
    if not np.all(np.isfinite(linear)):
        raise ValueError("db_values produce non-finite linear magnitudes")
    return linear


def linear_to_db(magnitude: np.ndarray, eps: float = EPS) -> np.ndarray:
    magnitude = ensure_non_negative_array("magnitude", magnitude)
    eps = ensure_numeric_scalar("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    db = np.asarray(20.0 * np.log10(np.maximum(magnitude, eps)), dtype=np.float64)
    if not np.all(np.isfinite(db)):
        raise ValueError("magnitude produces non-finite dB values")
    return db


def _validate_response_axes(freqs_hz: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs_hz = ensure_strictly_increasing_freqs(freqs_hz)
    values = ensure_1d_finite_array("values", values)
    if len(freqs_hz) != len(values):
        raise ValueError("freqs_hz and values must have the same length")
    return freqs_hz, values


def _validate_nyquist_coverage(freqs_hz: np.ndarray, sample_rate: int) -> None:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    nyquist = float(sample_rate) / 2.0
    if freqs_hz[-1] < nyquist:
        raise ValueError(f"Input response must extend to Nyquist ({nyquist} Hz), got {freqs_hz[-1]} Hz")


def _safe_log10_freqs(freqs_hz: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(freqs_hz, dtype=np.float64), LOG_EPS_HZ))


def interpolate_log_frequency_response(
    freqs_hz: np.ndarray,
    values: np.ndarray,
    target_freqs_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate onto a target frequency grid using a log-frequency axis.

    This preserves low-frequency detail better than linear interpolation.
    """
    freqs_hz, values = _validate_response_axes(freqs_hz, values)
    target_freqs_hz = ensure_non_negative_array("target_freqs_hz", target_freqs_hz)

    interp_values = np.interp(
        _safe_log10_freqs(target_freqs_hz),
        _safe_log10_freqs(freqs_hz),
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )
    return np.asarray(interp_values, dtype=np.float64)


def interpolate_magnitude_response(
    freqs_hz: np.ndarray,
    magnitude: np.ndarray,
    fft_size: int,
    sample_rate: int,
) -> np.ndarray:
    freqs_hz, magnitude = _validate_response_axes(freqs_hz, magnitude)
    if np.any(magnitude < 0):
        raise ValueError("magnitude must be non-negative")

    _validate_nyquist_coverage(freqs_hz, sample_rate)
    fft_freqs = build_fft_freq_grid(fft_size, sample_rate)
    return np.asarray(
        np.interp(fft_freqs, freqs_hz, magnitude, left=float(magnitude[0]), right=float(magnitude[-1])),
        dtype=np.float64,
    )


def interpolate_db_response(
    freqs_hz: np.ndarray,
    db_values: np.ndarray,
    fft_size: int,
    sample_rate: int,
) -> np.ndarray:
    freqs_hz, db_values = _validate_response_axes(freqs_hz, db_values)
    _validate_nyquist_coverage(freqs_hz, sample_rate)

    fft_freqs = build_fft_freq_grid(fft_size, sample_rate)
    interp_db = np.asarray(
        np.interp(fft_freqs, freqs_hz, db_values, left=float(db_values[0]), right=float(db_values[-1])),
        dtype=np.float64,
    )
    return db_to_linear(interp_db)


def _validate_mag_spectrum(magnitude: np.ndarray, fft_size: int) -> np.ndarray:
    fft_size = ensure_positive_int("fft_size", fft_size)
    if fft_size < 2 or fft_size % 2 != 0:
        raise ValueError("fft_size must be an even integer >= 2")
    magnitude = ensure_non_negative_array("Magnitude", magnitude)
    expected_len = fft_size // 2 + 1
    if len(magnitude) != expected_len:
        raise ValueError(f"Magnitude length must be fft_size//2 + 1 ({expected_len}), got {len(magnitude)}")
    return magnitude


def _full_even_spectrum_from_rfft_bins(values: np.ndarray, fft_size: int) -> np.ndarray:
    midpoint = fft_size // 2
    full = np.empty(fft_size, dtype=np.float64)
    full[: midpoint + 1] = values
    full[midpoint + 1 :] = values[1:midpoint][::-1]
    return full


def _full_hermitian_spectrum_from_rfft_bins(values: np.ndarray, fft_size: int) -> np.ndarray:
    midpoint = fft_size // 2
    positive = np.asarray(values, dtype=np.complex128)
    expected_len = midpoint + 1
    if positive.size != expected_len:
        raise ValueError(f"Spectrum length must be fft_size//2 + 1 ({expected_len}), got {positive.size}")
    if positive[0].imag != 0.0 or positive[midpoint].imag != 0.0:
        raise ValueError("Hermitian spectrum requires real DC and Nyquist bins")

    full = np.empty(fft_size, dtype=np.complex128)
    full[: midpoint + 1] = positive
    full[midpoint + 1 :] = np.conj(positive[1:midpoint][::-1])
    return full


def _ifft_hermitian_to_real(spectrum: np.ndarray) -> np.ndarray:
    time = np.fft.ifft(spectrum)
    imag_limit = np.finfo(np.float64).eps * np.linalg.norm(spectrum)
    imag_energy = np.linalg.norm(time.imag)

    if imag_energy > imag_limit:
        raise ValueError("Hermitian spectrum produced non-real FIR values")
    if np.max(np.abs(time.imag)) > imag_limit:
        raise ValueError("Hermitian spectrum produced non-real FIR values")
    return np.asarray(time.real, dtype=np.float64)

def minimum_phase_from_mag(magnitude: np.ndarray, fft_size: int) -> np.ndarray:
    fft_size = ensure_positive_int("fft_size", fft_size)
    if fft_size % 2 != 0:
        raise RuntimeError("minimum_phase_from_mag requires even fft_size")
    magnitude = _validate_mag_spectrum(magnitude, fft_size)
    if np.any(magnitude <= 0.0):
        raise ValueError("minimum-phase design requires strictly positive magnitudes")

    reference_gain = float(np.max(magnitude))
    if not np.isfinite(reference_gain) or reference_gain <= 0.0:
        raise ValueError("minimum-phase design requires a positive finite magnitude peak")
    normalized_magnitude = np.asarray(magnitude / reference_gain, dtype=np.float64)
    if np.any(normalized_magnitude < MINIMUM_PHASE_RELATIVE_MAGNITUDE_FLOOR):
        raise ValueError(
            "minimum-phase design requires relative magnitudes >= "
            f"{MINIMUM_PHASE_RELATIVE_MAGNITUDE_FLOOR:g}; increase the notch floor or use a larger FFT target"
        )

    log_mag = _full_even_spectrum_from_rfft_bins(np.log(normalized_magnitude), fft_size)
    cepstrum = _ifft_hermitian_to_real(log_mag.astype(np.complex128))

    cepstrum_min = np.zeros_like(cepstrum)
    midpoint = fft_size // 2
    cepstrum_min[0] = cepstrum[0]
    cepstrum_min[1:midpoint] = 2 * cepstrum[1:midpoint]

    cepstrum_min[midpoint] = cepstrum[midpoint]

    spectrum_min = np.exp(np.fft.fft(cepstrum_min, n=fft_size))
    fir_min = _ifft_hermitian_to_real(spectrum_min)
    achieved = np.asarray(np.abs(np.fft.rfft(fir_min, n=fft_size)), dtype=np.float64)
    relative_error = np.max(
        np.abs(achieved - normalized_magnitude)
        / np.maximum(normalized_magnitude, MINIMUM_PHASE_RELATIVE_MAGNITUDE_FLOOR)
    )
    if not np.isfinite(relative_error) or relative_error > MINIMUM_PHASE_MAGNITUDE_RTOL:
        raise RuntimeError("minimum-phase reconstruction did not preserve the requested FFT-bin magnitude")
    fir_min = np.asarray(fir_min * reference_gain, dtype=np.float64)
    if not np.all(np.isfinite(fir_min)):
        raise ValueError("magnitude produces non-finite FIR values")
    return np.asarray(fir_min, dtype=np.float64)


def minimum_phase_from_fir(fir: np.ndarray) -> np.ndarray:
    fir = ensure_1d_finite_array("FIR", fir)
    magnitude = np.asarray(np.abs(np.fft.rfft(fir)), dtype=np.float64)
    return minimum_phase_from_mag(magnitude, len(fir))


def design_fir_from_mag_fft(
    magnitude: np.ndarray,
    fft_size: int,
    minimum_phase: bool = True,
) -> np.ndarray:
    """
    FFT-based FIR design (scales to very large filters).

    - magnitude: linear magnitude (rfft domain, size = fft_size//2 + 1)
    - fft_size: full FFT size
    - minimum_phase: convert to minimum phase (recommended)
    """

    magnitude = _validate_mag_spectrum(magnitude, fft_size)
    minimum_phase = ensure_bool("minimum_phase", minimum_phase)
    max_mag = float(np.max(magnitude)) if magnitude.size else 0.0
    if max_mag > 0.0 and max_mag > np.finfo(np.float64).max / float(max(magnitude.size, 1)):
        raise ValueError("magnitude produces non-finite FIR values")

    if minimum_phase:
        return minimum_phase_from_mag(magnitude, fft_size)

    # ---- Step 1: build complex spectrum (zero phase) ----
    spectrum = _full_hermitian_spectrum_from_rfft_bins(magnitude.astype(np.complex128), fft_size)

    # ---- Step 2: IFFT → linear-phase FIR ----
    fir = _ifft_hermitian_to_real(spectrum)
    if not np.all(np.isfinite(fir)):
        raise ValueError("magnitude produces non-finite FIR values")

    # ---- Step 3: center impulse (linear phase alignment) ----
    fir = np.roll(fir, -fft_size // 2)

    return np.asarray(fir, dtype=np.float64)

def design_fir_from_mag_firls(
    magnitude: np.ndarray,
    fft_size: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Least-squares FIR design from target magnitude.

    - Uses firls for globally optimal L2 fit
    - Works on linear magnitude
    - Returns linear-phase FIR (will be converted later if needed)
    """

    n_taps = fft_size if fft_size % 2 == 1 else fft_size - 1
    if n_taps < 1:
        raise ValueError("FIRLS requires at least one tap")
    nyq = sample_rate / 2.0

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    num_points = min(len(freqs), 4097)
    indices = np.unique(
        np.round(np.linspace(0, len(freqs) - 1, num_points)).astype(np.int64)
    )

    bands_ds = np.clip(freqs[indices] / nyq, 0.0, 1.0)
    bands_ds = np.unique(bands_ds)

    if len(bands_ds) < 2:
        raise ValueError("Insufficient unique frequency points for FIRLS")

    bands_ds[0] = 0.0
    bands_ds[-1] = 1.0
    interp_mag = np.interp(bands_ds, freqs / nyq, magnitude)

    weights = np.ones_like(bands_ds[:-1])

    if len(weights) != len(bands_ds) - 1:
        raise RuntimeError("FIRLS weight mismatch")

    # 🔧 REQUIRED FIX
    bands = np.repeat(bands_ds, 2)[1:-1]
    desired = np.repeat(interp_mag, 2)[1:-1]

    fir = firls(
        numtaps=n_taps,
        bands=bands,
        desired=desired,
        weight=weights,
        fs=2.0
    )

    # Optional but recommended
    return fir.astype(np.float64)


def oversample_signal(x: np.ndarray, factor: int) -> np.ndarray:
    x = ensure_1d_finite_array("Signal", x)
    factor = ensure_positive_int("factor", factor)
    if factor == 1:
        return x
    return np.asarray(resample_poly(x, up=factor, down=1), dtype=np.float64)

def downsample_fir(
    fir_high: np.ndarray,
    high_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    Downsample FIR from high design rate to target rate.
    Uses polyphase filtering (high quality).
    """

    fir_high = ensure_1d_finite_array("fir_high", fir_high)
    high_rate = ensure_positive_sample_rate(high_rate)
    target_rate = ensure_positive_sample_rate(target_rate)

    if high_rate == target_rate:
        return fir_high

    ratio = high_rate / target_rate

    if not ratio.is_integer():
        raise ValueError(
            f"Oversampling ratio must be integer (got {high_rate}/{target_rate})"
        )

    down = int(ratio)

    fir_final = resample_poly(
        fir_high,
        up=1,
        down=down,
        window=("kaiser", 8.6),
    )

    return np.asarray(fir_final, dtype=np.float64)

def compute_true_peak(signal: np.ndarray, oversample_factor: int = 8) -> float:
    signal = ensure_1d_finite_array("signal", signal)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    oversampled = oversample_signal(signal, oversample_factor)
    return float(np.max(np.abs(oversampled)))


def _deprecated_normalize_fir(*args, **kwargs):
    raise RuntimeError("normalize_fir is deprecated — use apply_tap_scaling")

def apply_tap_scaling(
    fir: np.ndarray,
    headroom_db: float,
    true_peak: bool = False,
    oversample_factor: int = 8,
    mode: str = "neutral",
) -> np.ndarray:
    """
    Tap scaling with optional true-peak awareness.
    This replaces normalize_fir in advanced pipelines.
    """

    fir = ensure_1d_finite_array("FIR", fir)
    headroom_db = ensure_non_negative_float("headroom_db", headroom_db)
    true_peak = ensure_bool("true_peak", true_peak)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)

    # Measure peak (respect your existing true-peak logic)
    measured_peak = (
        compute_true_peak(fir, oversample_factor)
        if true_peak
        else float(np.max(np.abs(fir)))
    )

    if measured_peak == 0:
        raise ValueError("Cannot scale silent FIR")

    target_peak = 10 ** (-headroom_db / 20.0)

    # Base normalization (same as normalize_fir)
    fir = (fir / measured_peak) * target_peak

    if mode == "neutral":
        return np.asarray(fir, dtype=np.float64)

    # Profile modes are intentionally contract-preserving aliases for now.
    # Any future shaping must happen before a final normalization pass.
    return np.asarray(fir, dtype=np.float64)


def normalize_fir(*args, **kwargs):
    return apply_tap_scaling(*args, **kwargs)


def verify_headroom(
    fir: np.ndarray,
    headroom_db: float,
    true_peak: bool = False,
    oversample_factor: int = 8,
    tolerance: float = 1e-9,
) -> None:
    fir = ensure_1d_finite_array("FIR", fir)
    headroom_db = ensure_non_negative_float("headroom_db", headroom_db)
    true_peak = ensure_bool("true_peak", true_peak)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)

    target_peak = 10 ** (-float(headroom_db) / 20.0)
    measured_peak = compute_true_peak(fir, oversample_factor) if true_peak else float(np.max(np.abs(fir)))
    if measured_peak > target_peak + tolerance:
        peak_type = "true peak" if true_peak else "sample peak"
        raise ValueError(
            f"Headroom verification failed: measured {peak_type} {measured_peak:.12f} exceeds target {target_peak:.12f}"
        )


def resolve_window(window_type: str | None = None, beta: float = 8.6, preset: str | None = None) -> WindowSpec:
    if preset is not None:
        if preset not in WINDOW_PRESETS:
            raise ValueError(f"window_preset must be one of {sorted(WINDOW_PRESETS)}")
        return WINDOW_PRESETS[preset]
    if window_type is None:
        return WindowSpec(name=None, beta=None, preset=None)
    if window_type not in {"hann", "kaiser", "blackman"}:
        raise ValueError("window_type must be None, 'hann', 'kaiser', or 'blackman'")
    if window_type == "kaiser":
        checked_beta = ensure_numeric_scalar("beta", beta)
        if checked_beta <= 0:
            raise ValueError("beta must be > 0 for kaiser window")
        return WindowSpec(name=window_type, beta=checked_beta, preset=None)
    return WindowSpec(name=window_type, beta=None, preset=None)


def _build_causal_window_taper(length: int, window: WindowSpec) -> np.ndarray:
    length = ensure_positive_int("length", length)
    if window.name is None or length == 1:
        return np.ones(length, dtype=np.float64)

    full_length = (2 * length) - 1
    if window.name == "hann":
        full_window = np.hanning(full_length)
    elif window.name == "blackman":
        full_window = np.blackman(full_length)
    elif window.name == "kaiser":
        beta = 8.6 if window.beta is None else float(window.beta)
        full_window = np.kaiser(full_length, beta)
    else:
        raise ValueError(f"Unsupported window '{window.name}'")

    taper = np.asarray(full_window[length - 1 :], dtype=np.float64)
    peak = float(taper[0])
    if peak <= 0.0:
        raise ValueError("Window taper must begin with a positive gain")
    return np.asarray(taper / peak, dtype=np.float64)


def apply_window_to_fir(fir: np.ndarray, window: WindowSpec, minimum_phase: bool = True) -> np.ndarray:
    if not minimum_phase:
        if window.name is None:
            return fir
        full_window = np.hanning(len(fir))
        return fir * full_window

    fir = ensure_1d_finite_array("FIR", fir)
    if window.name is None or fir.size <= 1:
        return np.asarray(fir, dtype=np.float64)

    peak_index = int(np.argmax(np.abs(fir)))
    if peak_index >= fir.size - 1:
        return np.asarray(fir, dtype=np.float64)

    taper = _build_causal_window_taper(fir.size - peak_index, window)
    windowed = np.asarray(fir, dtype=np.float64).copy()
    windowed[peak_index:] *= taper
    return np.asarray(windowed, dtype=np.float64)


def compute_frequency_response(fir: np.ndarray, fft_size: int | None = None) -> np.ndarray:
    fir = ensure_1d_finite_array("FIR", fir)
    n = fft_size if fft_size is not None else len(fir)
    n = ensure_positive_int("fft_size", n)
    if n < fir.size:
        raise ValueError(f"fft_size must be >= FIR length ({fir.size})")
    max_abs = float(np.max(np.abs(fir))) if fir.size else 0.0
    if max_abs > 0.0 and max_abs > np.finfo(np.float64).max / float(max(n, 1)):
        raise ValueError("frequency response contains non-finite values")
    response = np.asarray(np.abs(np.fft.rfft(fir, n=n)), dtype=np.float64)
    if not np.all(np.isfinite(response)):
        raise ValueError("frequency response contains non-finite values")
    return response


def _safe_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return 0.0
    if not np.isfinite(max_abs):
        raise ValueError("response error summary contains non-finite values")
    normalized = values / max_abs
    rms = float(max_abs * np.sqrt(np.mean(np.square(normalized))))
    if not np.isfinite(rms):
        raise ValueError("response error summary contains non-finite values")
    return rms


def summarize_response_error(target_magnitude: np.ndarray, actual_magnitude: np.ndarray) -> ResponseErrorSummary:
    target = ensure_non_negative_array("target_magnitude", target_magnitude)
    actual = ensure_non_negative_array("actual_magnitude", actual_magnitude)
    ensure_matching_lengths("target_magnitude", target, "actual_magnitude", actual)

    # Align away any single global gain offset before computing response error.
    # The designed FIR is intentionally normalized to the requested headroom,
    # while the target magnitude remains at its original absolute gain. Without
    # this step the reported dB error is dominated by that deliberate level shift
    # rather than by true response-shape mismatch.
    denom = float(np.dot(actual, actual))
    gain = float(np.dot(target, actual) / denom) if denom > 0.0 else 1.0
    if not np.isfinite(gain) or gain <= 0.0:
        gain = 1.0
    actual_aligned = np.asarray(actual * gain, dtype=np.float64)

    abs_error = np.abs(actual_aligned - target)
    target_db = linear_to_db(target)
    actual_db = linear_to_db(actual_aligned)
    error_db = np.abs(actual_db - target_db)
    summary = ResponseErrorSummary(
        aligned_max_abs_error=float(np.max(abs_error)),
        aligned_rms_error=_safe_rms(actual_aligned - target),
        aligned_mean_abs_error=float(np.mean(abs_error)),
        max_abs_error_db=float(np.max(error_db)),
        rms_error_db=_safe_rms(actual_db - target_db),
        p95_abs_error_db=float(np.percentile(error_db, 95.0)),
    )
    if not all(np.isfinite(value) for value in summary.__dict__.values()):
        raise ValueError("response error summary contains non-finite values")
    return summary


def summarize_latency(fir: np.ndarray, sample_rate: int) -> LatencySummary:
    fir = ensure_1d_finite_array("FIR", fir)
    sample_rate = ensure_positive_sample_rate(sample_rate)

    abs_weights = np.abs(fir)
    peak_abs = float(np.max(abs_weights)) if abs_weights.size else 0.0
    scaled_abs_weights = abs_weights / peak_abs if peak_abs > 0.0 else np.zeros_like(fir)
    energy_weights = np.square(scaled_abs_weights)
    idx = np.arange(len(fir), dtype=np.float64)
    abs_weight_sum = float(np.sum(scaled_abs_weights))
    energy_weight_sum = float(np.sum(energy_weights))
    abs_centroid = float(np.sum(idx * scaled_abs_weights) / abs_weight_sum) if abs_weight_sum > 0.0 else 0.0
    energy_centroid = float(np.sum(idx * energy_weights) / energy_weight_sum) if energy_weight_sum > 0.0 else 0.0
    peak_index = int(np.argmax(abs_weights))
    sr = float(sample_rate)
    return LatencySummary(
        sample_rate=int(sample_rate),
        taps=int(len(fir)),
        peak_index=peak_index,
        peak_latency_ms=float((peak_index / sr) * 1000.0),
        nominal_linear_phase_latency_ms=float((((len(fir) - 1) / 2.0) / sr) * 1000.0),
        abs_centroid_ms=float((abs_centroid / sr) * 1000.0),
        energy_centroid_ms=float((energy_centroid / sr) * 1000.0),
    )


def coeff_hash_sha256(fir: np.ndarray) -> str:
    fir = ensure_1d_finite_array("fir", fir)
    canonical = np.asarray(fir, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _reject_non_finite_json_numbers(value: Any, path: str = "payload") -> None:
    if isinstance(value, (bool, str)) or value is None:
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            raise ValueError(f"{path} contains non-finite numeric value")
        return
    if isinstance(value, np.ndarray):
        _reject_non_finite_json_numbers(value.tolist(), path)
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_non_finite_json_numbers(key, f"{path} key")
            _reject_non_finite_json_numbers(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_non_finite_json_numbers(nested, f"{path}[{index}]")
        return


def request_fingerprint(payload: dict[str, Any]) -> str:
    _reject_non_finite_json_numbers(payload)
    ready = dataclass_to_json_ready(payload)
    _reject_non_finite_json_numbers(ready)
    encoded = json.dumps(ready, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_verification(
    fir: np.ndarray,
    target_magnitude: np.ndarray,
    oversample_factor: int,
    request_payload: dict[str, Any],
    *,
    true_peak_target_dbfs: float,
    true_peak_min_safe_margin_db: float = 0.1,
    gain_stage_preamp_source: str = "none",
    preamp_applied_as_gain_stage: bool = False,
    source_preamp_db: float | None = None,
    source_preamp_present: bool = False,
    source_preamp_used_for_target_shape: bool = False,
    source_preamp_origin: str | None = None,
    final_baked_headroom_db: float = 0.0,
    final_gain_policy: str = "normalized_to_oversampled_peak_estimate_target",
    true_peak_policy: str = "normalize_to_oversampled_peak_estimate",
    true_peak_target_is_baked_safety_ceiling: bool = True,
    true_peak_margin_warning_enabled: bool = True,
) -> VerificationSummary:
    fir = ensure_1d_finite_array("fir", fir)
    target_magnitude = ensure_non_negative_array("target_magnitude", target_magnitude)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    true_peak_target_dbfs = ensure_numeric_scalar("true_peak_target_dbfs", true_peak_target_dbfs)
    true_peak_min_safe_margin_db = ensure_non_negative_float(
        "true_peak_min_safe_margin_db", true_peak_min_safe_margin_db
    )
    final_baked_headroom_db = ensure_non_negative_float("final_baked_headroom_db", final_baked_headroom_db)
    preamp_applied_as_gain_stage = ensure_bool("preamp_applied_as_gain_stage", preamp_applied_as_gain_stage)
    source_preamp_present = ensure_bool("source_preamp_present", source_preamp_present)
    source_preamp_used_for_target_shape = ensure_bool(
        "source_preamp_used_for_target_shape", source_preamp_used_for_target_shape
    )
    true_peak_target_is_baked_safety_ceiling = ensure_bool(
        "true_peak_target_is_baked_safety_ceiling", true_peak_target_is_baked_safety_ceiling
    )
    true_peak_margin_warning_enabled = ensure_bool(
        "true_peak_margin_warning_enabled", true_peak_margin_warning_enabled
    )
    checked_source_preamp_db = (
        None if source_preamp_db is None else ensure_numeric_scalar("source_preamp_db", source_preamp_db)
    )
    fir_peak_linear = float(np.max(np.abs(fir)))
    fir_true_peak_linear = compute_true_peak(fir, oversample_factor=oversample_factor)
    true_peak_target_linear = float(10.0 ** (true_peak_target_dbfs / 20.0))
    true_peak_margin_linear = float(true_peak_target_linear - fir_true_peak_linear)
    fir_true_peak_dbfs = float(20.0 * np.log10(max(fir_true_peak_linear, EPS)))
    true_peak_margin_db = float(true_peak_target_dbfs - fir_true_peak_dbfs)
    return VerificationSummary(
        fir_peak_linear=fir_peak_linear,
        fir_peak_dbfs=float(20.0 * np.log10(max(fir_peak_linear, EPS))),
        fir_true_peak_linear=fir_true_peak_linear,
        fir_true_peak_dbfs=fir_true_peak_dbfs,
        true_peak_target_linear=true_peak_target_linear,
        true_peak_target_dbfs=float(true_peak_target_dbfs),
        true_peak_margin_linear=true_peak_margin_linear,
        true_peak_margin_db=true_peak_margin_db,
        true_peak_margin_warning=bool(
            true_peak_margin_warning_enabled
            and true_peak_margin_db + TRUE_PEAK_MARGIN_TOLERANCE_DB < true_peak_min_safe_margin_db
        ),
        true_peak_min_safe_margin_db=true_peak_min_safe_margin_db,
        target_hash_sha256=coeff_hash_sha256(target_magnitude),
        coeff_hash_sha256=coeff_hash_sha256(fir),
        request_fingerprint_sha256=request_fingerprint(request_payload),
        gain_stage_preamp_source=gain_stage_preamp_source,
        preamp_applied_as_gain_stage=preamp_applied_as_gain_stage,
        source_preamp_db=checked_source_preamp_db,
        source_preamp_present=source_preamp_present,
        source_preamp_used_for_target_shape=source_preamp_used_for_target_shape,
        source_preamp_origin=source_preamp_origin,
        final_baked_headroom_db=final_baked_headroom_db,
        final_gain_policy=final_gain_policy,
        true_peak_policy=true_peak_policy,
        true_peak_target_is_baked_safety_ceiling=true_peak_target_is_baked_safety_ceiling,
    )


def dataclass_to_json_ready(value: object) -> dict[str, Any] | list[Any] | Any:
    if is_dataclass(value):
        return {k: dataclass_to_json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return dataclass_to_json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON-ready payload contains non-finite numeric value")
        return value
    if isinstance(value, (list, tuple)):
        return [dataclass_to_json_ready(v) for v in value]
    if isinstance(value, dict):
        ready_dict: dict[str, Any] = {}
        for key, nested in value.items():
            ready_key = dataclass_to_json_ready(key)
            json_key = str(ready_key)
            if json_key in ready_dict:
                raise ValueError("JSON-ready payload contains duplicate key after string conversion")
            ready_dict[json_key] = dataclass_to_json_ready(nested)
        return ready_dict
    return value
