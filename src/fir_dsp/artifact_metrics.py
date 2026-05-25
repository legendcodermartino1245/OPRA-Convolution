from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from .core import _safe_rms, compute_true_peak
from .validation import (
    ensure_1d_finite_array,
    ensure_non_negative_array,
    ensure_non_negative_float,
    ensure_positive_int,
    ensure_positive_sample_rate,
)

EPS = 1e-12
FLOAT32_DENORMAL_THRESHOLD = float(np.finfo(np.float32).tiny)
FLOAT32_MAX_ABS = float(np.finfo(np.float32).max)


def _fft_convolve_same_numpy_length(signal: np.ndarray, fir: np.ndarray) -> np.ndarray:
    full = fftconvolve(signal, fir, mode="full")
    target_len = max(signal.size, fir.size)
    start = (min(signal.size, fir.size) - 1) // 2
    return np.asarray(full[start : start + target_len], dtype=np.float64)


@dataclass(frozen=True)
class ArtifactContract:
    wav_subtype: str
    wav_bits_per_sample: int
    channels: int
    frames: int
    sample_rate: int
    duration_ms: float


@dataclass(frozen=True)
class ExportParitySummary:
    wav_txt_max_abs_diff: float
    wav_txt_rms_diff: float
    wav_txt_allclose: bool


@dataclass(frozen=True)
class EffectiveLengthSummary:
    energy_50_ms: float
    energy_90_ms: float
    energy_99_ms: float
    energy_999_ms: float
    last_tap_above_minus_120db: int


@dataclass(frozen=True)
class GainSummary:
    dc_gain_db: float
    nyquist_gain_db: float
    max_gain_db: float
    max_gain_freq_hz: float
    min_gain_db: float
    min_gain_freq_hz: float


@dataclass(frozen=True)
class ReconstructionErrorSummary:
    max_abs_error_db: float
    rms_error_db: float
    p95_abs_error_db: float


@dataclass(frozen=True)
class FloatSafetySummary:
    has_nan: bool
    has_inf: bool
    has_denormals: bool
    min_nonzero_abs_coeff: float
    zero_tap_count: int


@dataclass(frozen=True)
class CrossRateBandSummary:
    label: str
    min_freq_hz: float
    max_freq_hz: float
    max_delta_db: float
    p95_delta_db: float
    rms_delta_db: float
    worst_frequency_hz: float


@dataclass(frozen=True)
class CrossRateConsistencySummary:
    max_response_delta_between_rates_db: float
    p95_response_delta_between_rates_db: float
    rms_response_delta_between_rates_db: float
    worst_pair: str
    worst_frequency_hz: float
    all_rates_pass: bool
    comparison_mode: str
    min_freq_hz: float
    max_freq_hz: float
    reference_rate: int
    alignment_min_freq_hz: float
    alignment_max_freq_hz: float
    strict_max_response_delta_between_rates_db: float
    strict_all_rates_pass: bool
    extended_max_response_delta_between_rates_db: float
    extended_warning: bool
    cross_rate_gain_alignment_offsets_db: dict[str, float]
    band_summaries: list[CrossRateBandSummary]


@dataclass(frozen=True)
class TargetErrorBandSummary:
    label: str
    min_freq_hz: float
    max_freq_hz: float
    max_abs_error_db: float
    p95_abs_error_db: float
    rms_error_db: float
    mean_abs_error_db: float
    worst_frequency_hz: float


@dataclass(frozen=True)
class PerceptualWeightedErrorSummary:
    weighted_mean_abs_error_db: float
    weighted_rms_error_db: float
    max_abs_error_db: float
    worst_frequency_hz: float


@dataclass(frozen=True)
class StressProbeSummary:
    worst_probe: str
    worst_true_peak_dbfs: float
    true_peak_target_dbfs: float
    passes_target: bool
    probe_true_peak_dbfs: dict[str, float]


def canonical_wav_array(signal: np.ndarray) -> np.ndarray:
    arr = ensure_1d_finite_array("signal", signal, non_empty=False)
    if arr.size and float(np.max(np.abs(arr))) > FLOAT32_MAX_ABS:
        raise ValueError("signal values must fit in finite float32 WAV range")
    converted = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError("signal values must remain finite after float32 conversion")
    return converted


def artifact_contract(signal: np.ndarray, sample_rate: int) -> ArtifactContract:
    arr = ensure_1d_finite_array("signal", signal, non_empty=False)
    if arr.ndim != 1:
        raise ValueError("artifact_contract expects a mono 1D signal")
    sample_rate = ensure_positive_sample_rate(sample_rate)
    frames = int(arr.size)
    return ArtifactContract(
        wav_subtype="FLOAT",
        wav_bits_per_sample=32,
        channels=1,
        frames=frames,
        sample_rate=sample_rate,
        duration_ms=float((frames / float(sample_rate)) * 1000.0),
    )


def export_parity(reference: np.ndarray, exported: np.ndarray) -> ExportParitySummary:
    ref = ensure_1d_finite_array("reference", reference, non_empty=False)
    exp = ensure_1d_finite_array("exported", exported, non_empty=False)
    if ref.shape != exp.shape:
        raise ValueError("reference and exported must have the same shape")
    diff = ref - exp
    return ExportParitySummary(
        wav_txt_max_abs_diff=float(np.max(np.abs(diff))) if diff.size else 0.0,
        wav_txt_rms_diff=_safe_rms(diff),
        wav_txt_allclose=bool(np.allclose(ref, exp, rtol=1e-6, atol=1e-8)),
    )


def effective_length(signal: np.ndarray, sample_rate: int) -> EffectiveLengthSummary:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    fir = ensure_1d_finite_array("signal", signal, non_empty=False)
    peak = float(np.max(np.abs(fir))) if fir.size else 0.0
    scaled = fir / peak if peak > 0.0 else np.zeros_like(fir)
    sq = np.square(scaled)
    total_energy = float(np.sum(sq))
    cumulative = np.cumsum(sq)

    def _energy_time_ms(target_fraction: float) -> float:
        if total_energy <= EPS or cumulative.size == 0:
            return 0.0
        target = total_energy * float(target_fraction)
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(idx, max(fir.size - 1, 0))
        return float((idx / float(sample_rate)) * 1000.0)

    threshold = peak * (10.0 ** (-120.0 / 20.0))
    if peak <= 0.0:
        last_above = -1
    else:
        significant = np.flatnonzero(np.abs(fir) >= threshold)
        last_above = int(significant[-1]) if significant.size else -1

    return EffectiveLengthSummary(
        energy_50_ms=_energy_time_ms(0.50),
        energy_90_ms=_energy_time_ms(0.90),
        energy_99_ms=_energy_time_ms(0.99),
        energy_999_ms=_energy_time_ms(0.999),
        last_tap_above_minus_120db=last_above,
    )


def gain_summary(signal: np.ndarray, sample_rate: int, fft_size: int | None = None) -> GainSummary:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    fir = ensure_1d_finite_array("signal", signal, non_empty=False)
    n_fft = ensure_positive_int("fft_size", fft_size) if fft_size is not None else max(fir.size, 1)
    n_fft = int(max(n_fft, fir.size or 1))
    max_abs = float(np.max(np.abs(fir))) if fir.size else 0.0
    if max_abs > 0.0 and max_abs > np.finfo(np.float64).max / float(max(n_fft, 1)):
        raise ValueError("gain_summary frequency response contains non-finite values")
    spectrum = np.fft.rfft(fir, n=n_fft)
    magnitude = np.abs(spectrum)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("gain_summary frequency response contains non-finite values")
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, EPS))
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))

    max_idx = int(np.argmax(magnitude_db))
    min_idx = int(np.argmin(magnitude_db))
    return GainSummary(
        dc_gain_db=float(magnitude_db[0]),
        nyquist_gain_db=float(magnitude_db[-1]),
        max_gain_db=float(magnitude_db[max_idx]),
        max_gain_freq_hz=float(freqs_hz[max_idx]),
        min_gain_db=float(magnitude_db[min_idx]),
        min_gain_freq_hz=float(freqs_hz[min_idx]),
    )


def reconstruction_error(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    sample_rate: int,
    fft_size: int | None = None,
) -> ReconstructionErrorSummary:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    ref = ensure_1d_finite_array("reference", reference, non_empty=False)
    rec = ensure_1d_finite_array("reconstructed", reconstructed, non_empty=False)
    n_fft = ensure_positive_int("fft_size", fft_size) if fft_size is not None else max(ref.size, rec.size, 1)
    n_fft = int(max(n_fft, ref.size or 1, rec.size or 1))
    ref_peak = float(np.max(np.abs(ref))) if ref.size else 0.0
    rec_peak = float(np.max(np.abs(rec))) if rec.size else 0.0
    max_fft_input = max(ref_peak, rec_peak)
    if max_fft_input > 0.0 and max_fft_input > np.finfo(np.float64).max / float(max(n_fft, 1)):
        raise ValueError("reconstruction_error frequency response contains non-finite values")
    ref_mag = np.abs(np.fft.rfft(ref, n=n_fft))
    rec_mag = np.abs(np.fft.rfft(rec, n=n_fft))
    if not np.all(np.isfinite(ref_mag)) or not np.all(np.isfinite(rec_mag)):
        raise ValueError("reconstruction_error frequency response contains non-finite values")
    peak_mag = float(max(np.max(ref_mag), np.max(rec_mag), EPS))
    mask = np.maximum(ref_mag, rec_mag) >= peak_mag * 1e-10
    if not np.any(mask):
        mask = np.ones_like(ref_mag, dtype=bool)
    diff_db = np.abs(
        20.0 * np.log10(np.maximum(ref_mag[mask], EPS))
        - 20.0 * np.log10(np.maximum(rec_mag[mask], EPS))
    )
    if not np.all(np.isfinite(diff_db)):
        raise ValueError("reconstruction_error summary contains non-finite values")
    return ReconstructionErrorSummary(
        max_abs_error_db=float(np.max(diff_db)) if diff_db.size else 0.0,
        rms_error_db=float(np.sqrt(np.mean(np.square(diff_db)))) if diff_db.size else 0.0,
        p95_abs_error_db=float(np.percentile(diff_db, 95.0)) if diff_db.size else 0.0,
    )


def float_safety(signal: np.ndarray) -> FloatSafetySummary:
    source = ensure_1d_finite_array("signal", signal, non_empty=False)
    overflows_float32 = bool(source.size and float(np.max(np.abs(source))) > FLOAT32_MAX_ABS)
    clipped_source = (
        np.zeros_like(source, dtype=np.float64)
        if overflows_float32
        else source
    )
    arr = np.asarray(clipped_source, dtype=np.float32)
    finite_abs = np.abs(arr[np.isfinite(arr)])
    nonzero = finite_abs[finite_abs > 0.0]
    return FloatSafetySummary(
        has_nan=bool(np.isnan(arr).any()),
        has_inf=bool(overflows_float32 or np.isinf(arr).any()),
        has_denormals=bool(np.any((nonzero < FLOAT32_DENORMAL_THRESHOLD))) if nonzero.size else False,
        min_nonzero_abs_coeff=float(np.min(nonzero)) if nonzero.size else 0.0,
        zero_tap_count=int(np.count_nonzero(arr == 0.0)),
    )


def _extract_cross_rate_response(result: Any, sample_rate: int) -> tuple[int, np.ndarray]:
    fft_size = getattr(result, "fft_size", None)
    actual_magnitude = getattr(result, "actual_magnitude", None)
    if fft_size is None or actual_magnitude is None:
        raise TypeError(
            "cross_rate_consistency expects PipelineResult-like values with fft_size and actual_magnitude"
        )

    fft_size = ensure_positive_int("fft_size", fft_size)
    magnitude = ensure_non_negative_array("actual_magnitude", actual_magnitude)
    expected_len = fft_size // 2 + 1
    if magnitude.size != expected_len:
        raise ValueError(
            f"cross_rate_consistency expected actual_magnitude length {expected_len} for sample_rate={sample_rate}, got {magnitude.size}"
        )
    return fft_size, magnitude


def cross_rate_consistency(results: dict[int, Any]) -> CrossRateConsistencySummary | None:
    if len(results) < 2:
        return None

    sorted_rates = sorted(ensure_positive_sample_rate(sr) for sr in results)
    reference_rate = 44_100 if 44_100 in results else sorted_rates[0]
    min_nyquist_hz = min(float(sr) / 2.0 for sr in results)
    extended_min_freq_hz = 20.0
    extended_max_freq_hz = min(20_000.0, min_nyquist_hz * 0.999)
    strict_min_freq_hz = 20.0
    strict_max_freq_hz = min(18_000.0, extended_max_freq_hz)
    alignment_min_freq_hz = 100.0
    alignment_max_freq_hz = min(10_000.0, strict_max_freq_hz)

    if extended_max_freq_hz <= extended_min_freq_hz:
        extended_max_freq_hz = min_nyquist_hz * 0.9
        extended_min_freq_hz = max(1.0, extended_min_freq_hz / 20.0)
    if strict_max_freq_hz <= strict_min_freq_hz:
        strict_max_freq_hz = extended_max_freq_hz
    if alignment_max_freq_hz <= alignment_min_freq_hz:
        alignment_min_freq_hz = strict_min_freq_hz
        alignment_max_freq_hz = strict_max_freq_hz

    probe_freqs = np.geomspace(extended_min_freq_hz, extended_max_freq_hz, num=4096)

    interpolated: dict[int, np.ndarray] = {}
    for sr, result in results.items():
        fft_size, actual_magnitude = _extract_cross_rate_response(result, int(sr))
        freqs_hz = np.fft.rfftfreq(fft_size, d=1.0 / float(sr))
        magnitude_db = 20.0 * np.log10(np.maximum(actual_magnitude, EPS))
        interpolated[int(sr)] = np.interp(probe_freqs, freqs_hz, magnitude_db)

    strict_mask = (probe_freqs >= strict_min_freq_hz) & (probe_freqs <= strict_max_freq_hz)
    extended_mask = (probe_freqs >= extended_min_freq_hz) & (probe_freqs <= extended_max_freq_hz)
    alignment_mask = (probe_freqs >= alignment_min_freq_hz) & (probe_freqs <= alignment_max_freq_hz)
    if not np.any(alignment_mask):
        alignment_mask = strict_mask
    if not np.any(strict_mask):
        strict_mask = extended_mask

    band_ranges = [
        ("20_60_hz", 20.0, 60.0),
        ("60_200_hz", 60.0, 200.0),
        ("200_1000_hz", 200.0, 1000.0),
        ("1_4_khz", 1000.0, 4000.0),
        ("4_10_khz", 4000.0, 10000.0),
        ("10_16_khz", 10000.0, 16000.0),
        ("16_20_khz", 16000.0, 20000.0),
    ]
    band_stats = {
        label: {"max": 0.0, "worst_frequency_hz": band_min, "deltas": []}
        for label, band_min, _ in band_ranges
    }

    reference = interpolated[reference_rate]
    strict_worst_delta = -1.0
    worst_pair = ""
    worst_frequency_hz = strict_min_freq_hz
    extended_worst_delta = -1.0
    strict_delta_collector: list[np.ndarray] = []
    alignment_offsets_db: dict[str, float] = {}
    for compared_rate in sorted_rates:
        if compared_rate == reference_rate:
            continue
        compared = interpolated[compared_rate]
        gain_offset_db = float(np.mean(reference[alignment_mask] - compared[alignment_mask]))
        alignment_offsets_db[str(compared_rate)] = gain_offset_db
        aligned_delta = reference - (compared + gain_offset_db)
        strict_abs_delta = np.abs(aligned_delta[strict_mask])
        extended_abs_delta = np.abs(aligned_delta[extended_mask])
        strict_delta_collector.append(strict_abs_delta)
        strict_delta = float(np.max(strict_abs_delta))
        extended_delta = float(np.max(extended_abs_delta))
        if strict_delta > strict_worst_delta:
            strict_worst_delta = strict_delta
            worst_pair = f"{reference_rate}_vs_{compared_rate}"
            strict_probe_freqs = probe_freqs[strict_mask]
            worst_frequency_hz = float(strict_probe_freqs[int(np.argmax(strict_abs_delta))])
        if extended_delta > extended_worst_delta:
            extended_worst_delta = extended_delta

        for label, band_min, band_max in band_ranges:
            band_max_capped = min(band_max, extended_max_freq_hz)
            if band_max_capped <= band_min:
                continue
            band_mask = (probe_freqs >= band_min) & (probe_freqs <= band_max_capped)
            if not np.any(band_mask):
                continue
            band_abs_delta = np.abs(aligned_delta[band_mask])
            band_stats[label]["deltas"].append(band_abs_delta)
            band_delta = float(np.max(band_abs_delta))
            if band_delta > float(band_stats[label]["max"]):
                band_stats[label]["max"] = band_delta
                band_probe_freqs = probe_freqs[band_mask]
                band_stats[label]["worst_frequency_hz"] = float(
                    band_probe_freqs[int(np.argmax(band_abs_delta))]
                )

    if strict_worst_delta < 0.0:
        strict_worst_delta = 0.0
    if extended_worst_delta < 0.0:
        extended_worst_delta = 0.0
    if strict_delta_collector:
        all_strict_deltas = np.concatenate(strict_delta_collector)
        p95_strict_delta = float(np.percentile(all_strict_deltas, 95.0))
        rms_strict_delta = float(np.sqrt(np.mean(np.square(all_strict_deltas))))
    else:
        p95_strict_delta = 0.0
        rms_strict_delta = 0.0

    strict_all_rates_pass = bool(strict_worst_delta <= 0.30)

    band_summaries = [
        CrossRateBandSummary(
            label=label,
            min_freq_hz=band_min,
            max_freq_hz=min(band_max, extended_max_freq_hz),
            max_delta_db=float(band_stats[label]["max"]),
            p95_delta_db=float(np.percentile(np.concatenate(band_stats[label]["deltas"]), 95.0))
            if band_stats[label]["deltas"]
            else 0.0,
            rms_delta_db=float(
                np.sqrt(np.mean(np.square(np.concatenate(band_stats[label]["deltas"]))))
            )
            if band_stats[label]["deltas"]
            else 0.0,
            worst_frequency_hz=float(band_stats[label]["worst_frequency_hz"]),
        )
        for label, band_min, band_max in band_ranges
        if min(band_max, extended_max_freq_hz) > band_min
    ]

    return CrossRateConsistencySummary(
        max_response_delta_between_rates_db=strict_worst_delta,
        p95_response_delta_between_rates_db=p95_strict_delta,
        rms_response_delta_between_rates_db=rms_strict_delta,
        worst_pair=worst_pair,
        worst_frequency_hz=worst_frequency_hz,
        all_rates_pass=strict_all_rates_pass,
        comparison_mode="gain_aligned_shape_audible_band",
        min_freq_hz=float(strict_min_freq_hz),
        max_freq_hz=float(strict_max_freq_hz),
        reference_rate=int(reference_rate),
        alignment_min_freq_hz=float(alignment_min_freq_hz),
        alignment_max_freq_hz=float(alignment_max_freq_hz),
        strict_max_response_delta_between_rates_db=strict_worst_delta,
        strict_all_rates_pass=strict_all_rates_pass,
        extended_max_response_delta_between_rates_db=extended_worst_delta,
        extended_warning=bool(extended_worst_delta > 0.50),
        cross_rate_gain_alignment_offsets_db=alignment_offsets_db,
        band_summaries=band_summaries,
    )


def _aligned_error_db(
    target_magnitude: np.ndarray,
    actual_magnitude: np.ndarray,
    sample_rate: int,
    fft_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    fft_size = ensure_positive_int("fft_size", fft_size)
    target = ensure_non_negative_array("target_magnitude", target_magnitude)
    actual = ensure_non_negative_array("actual_magnitude", actual_magnitude)
    if target.shape != actual.shape:
        raise ValueError("target_magnitude and actual_magnitude must have the same shape")
    expected_len = fft_size // 2 + 1
    if target.size != expected_len:
        raise ValueError(
            f"target_magnitude length {target.size} does not match fft_size {fft_size}"
        )
    freqs_hz = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
    target_db = 20.0 * np.log10(np.maximum(target, EPS))
    actual_db = 20.0 * np.log10(np.maximum(actual, EPS))
    gain_offset_db = float(np.mean(target_db - actual_db))
    error_db = (actual_db + gain_offset_db) - target_db
    return freqs_hz, error_db


def target_error_band(
    target_magnitude: np.ndarray,
    actual_magnitude: np.ndarray,
    *,
    sample_rate: int,
    fft_size: int,
    min_freq_hz: float,
    max_freq_hz: float,
    label: str,
) -> TargetErrorBandSummary:
    freqs_hz, error_db = _aligned_error_db(target_magnitude, actual_magnitude, sample_rate, fft_size)
    return target_error_band_from_aligned_error(
        freqs_hz,
        error_db,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        label=label,
    )


def target_error_band_from_aligned_error(
    freqs_hz: np.ndarray,
    error_db: np.ndarray,
    *,
    min_freq_hz: float,
    max_freq_hz: float,
    label: str,
) -> TargetErrorBandSummary:
    freqs_hz = ensure_non_negative_array("freqs_hz", freqs_hz)
    error_db = ensure_1d_finite_array("error_db", error_db)
    if freqs_hz.shape != error_db.shape:
        raise ValueError("freqs_hz and error_db must have the same shape")
    lower = ensure_non_negative_float("min_freq_hz", min_freq_hz)
    upper = ensure_non_negative_float("max_freq_hz", max_freq_hz)
    if lower > upper:
        raise ValueError("min_freq_hz must be <= max_freq_hz")
    mask = (freqs_hz >= lower) & (freqs_hz <= upper)
    band_freqs = freqs_hz[mask]
    band_error = error_db[mask]
    abs_error = np.abs(band_error)
    if band_freqs.size == 0:
        return TargetErrorBandSummary(label, lower, upper, 0.0, 0.0, 0.0, 0.0, lower)
    max_idx = int(np.argmax(abs_error))
    return TargetErrorBandSummary(
        label=label,
        min_freq_hz=float(band_freqs[0]),
        max_freq_hz=float(band_freqs[-1]),
        max_abs_error_db=float(np.max(abs_error)),
        p95_abs_error_db=float(np.percentile(abs_error, 95.0)),
        rms_error_db=float(np.sqrt(np.mean(np.square(band_error)))),
        mean_abs_error_db=float(np.mean(abs_error)),
        worst_frequency_hz=float(band_freqs[max_idx]),
    )


def perceptual_weighted_error(
    target_magnitude: np.ndarray,
    actual_magnitude: np.ndarray,
    *,
    sample_rate: int,
    fft_size: int,
) -> PerceptualWeightedErrorSummary:
    freqs_hz, error_db = _aligned_error_db(target_magnitude, actual_magnitude, sample_rate, fft_size)
    return perceptual_weighted_error_from_aligned_error(
        freqs_hz,
        error_db,
        sample_rate=sample_rate,
    )


def perceptual_weighted_error_from_aligned_error(
    freqs_hz: np.ndarray,
    error_db: np.ndarray,
    *,
    sample_rate: int,
) -> PerceptualWeightedErrorSummary:
    freqs_hz = ensure_non_negative_array("freqs_hz", freqs_hz)
    error_db = ensure_1d_finite_array("error_db", error_db)
    if freqs_hz.shape != error_db.shape:
        raise ValueError("freqs_hz and error_db must have the same shape")
    sample_rate = ensure_positive_sample_rate(sample_rate)
    mask = (freqs_hz >= 20.0) & (freqs_hz <= min(18_000.0, float(sample_rate) / 2.0))
    freqs = freqs_hz[mask]
    errors = error_db[mask]
    abs_errors = np.abs(errors)
    if freqs.size == 0:
        return PerceptualWeightedErrorSummary(0.0, 0.0, 0.0, 20.0)
    weights = np.full(freqs.shape, 0.5, dtype=np.float64)
    weights[(freqs >= 100.0) & (freqs < 1000.0)] = 1.0
    weights[(freqs >= 1000.0) & (freqs <= 6000.0)] = 2.0
    weights[(freqs > 6000.0) & (freqs <= 10000.0)] = 1.0
    weighted_mean = float(np.sum(abs_errors * weights) / np.sum(weights))
    weighted_rms = float(np.sqrt(np.sum(np.square(errors) * weights) / np.sum(weights)))
    max_idx = int(np.argmax(abs_errors))
    return PerceptualWeightedErrorSummary(
        weighted_mean_abs_error_db=weighted_mean,
        weighted_rms_error_db=weighted_rms,
        max_abs_error_db=float(np.max(abs_errors)),
        worst_frequency_hz=float(freqs[max_idx]),
    )


def _normalize_probe_peak(signal: np.ndarray, peak_linear: float) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        raise ValueError("Probe signal must not be silent")
    return np.asarray((signal / peak) * peak_linear, dtype=np.float64)


def _build_program_probes(length: int, peak_linear: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260401)
    t = np.arange(length, dtype=np.float64)
    sweep_phase = 2.0 * np.pi * (20.0 * t / length + ((20_000.0 - 20.0) / (2.0 * length * length)) * t * t)
    multitone = (
        np.sin(2.0 * np.pi * 53.0 * t / length)
        + np.sin(2.0 * np.pi * 997.0 * t / length)
        + np.sin(2.0 * np.pi * 5021.0 * t / length)
    )
    probes = {
        "step": np.ones(length, dtype=np.float64),
        "alternating": np.where((t.astype(np.int64) % 2) == 0, 1.0, -1.0),
        "multitone": multitone,
        "sweep": np.sin(sweep_phase),
        "noise": rng.standard_normal(length),
    }
    return {name: _normalize_probe_peak(signal, peak_linear) for name, signal in probes.items()}


def true_peak_stress_summary(
    fir: np.ndarray,
    *,
    requested_headroom_db: float,
    oversample_factor: int,
    probe_length: int = 32768,
) -> StressProbeSummary:
    fir = ensure_1d_finite_array("fir", fir)
    requested_headroom_db = ensure_non_negative_float("requested_headroom_db", requested_headroom_db)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    probe_length = ensure_positive_int("probe_length", probe_length)
    playback_peak_linear = 10.0 ** (-requested_headroom_db / 20.0)
    probes = _build_program_probes(probe_length, playback_peak_linear)
    peaks: dict[str, float] = {}
    for name, signal in probes.items():
        convolved = _fft_convolve_same_numpy_length(signal, fir)
        peaks[name] = float(
            20.0 * np.log10(max(compute_true_peak(convolved, oversample_factor=oversample_factor), EPS))
        )
    worst_probe = max(peaks, key=peaks.get)
    worst_tp = float(peaks[worst_probe])
    target_dbfs = -float(requested_headroom_db)
    return StressProbeSummary(
        worst_probe=worst_probe,
        worst_true_peak_dbfs=worst_tp,
        true_peak_target_dbfs=target_dbfs,
        passes_target=bool(worst_tp <= target_dbfs + 0.01),
        probe_true_peak_dbfs=peaks,
    )


def audible_target_verdict(
    listening_band_10k: TargetErrorBandSummary,
    listening_band_18k: TargetErrorBandSummary,
) -> str:
    if listening_band_18k.p95_abs_error_db <= 0.05 and listening_band_18k.max_abs_error_db <= 0.10:
        return "transparent"
    if listening_band_18k.p95_abs_error_db <= 0.10 and listening_band_18k.max_abs_error_db <= 0.30:
        return "very_close"
    if listening_band_10k.p95_abs_error_db <= 0.15 and listening_band_18k.max_abs_error_db <= 0.50:
        return "acceptable"
    return "review"
