from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .core import (
    TRUE_PEAK_MARGIN_TOLERANCE_DB,
    apply_tap_scaling,
    build_fft_freq_grid,
    coeff_hash_sha256,
    compute_frequency_response,
    compute_true_peak,
    design_fir_from_mag_fft,
    interpolate_log_frequency_response,
    db_to_linear,
    linear_to_db,
    resolve_window,
    summarize_latency,
    summarize_response_error,
    summarize_verification,
    verify_headroom,
    apply_window_to_fir,
    minimum_phase_from_mag,
)
from .core_validation import run_core_validations
from .eq_to_magnitude_native import eq_json_to_native_magnitude
from .logging_utils import logger
from .models import GainTraceabilitySummary, PipelineResult, PipelineSpec
from .preamp import apply_preamp_db, validate_preamp_db
from .profiles import PipelineProfile, resolve_profile
from .system_validation import validate_system
from .types import DbMagnitude, LinearMagnitude, coerce_db_magnitude, coerce_linear_magnitude
from .validation import (
    ensure_bool,
    ensure_choice,
    ensure_non_negative_float,
    ensure_positive_int,
    ensure_positive_sample_rate,
    normalize_rates,
)

_PREAMP_SOURCE_NONE = "none"
_PREAMP_SOURCE_PEQ = "peq"
_ALLOWED_PREAMP_SOURCES = {_PREAMP_SOURCE_NONE, _PREAMP_SOURCE_PEQ}
_MIN_SAFE_TRUE_PEAK_MARGIN_DB = 0.0
_TARGET_SILENCE_FLOOR = 0.0
_MAX_TARGET_MAGNITUDE = 1e6
_PARALLEL_MULTI_RATE_MIN_FFT_SIZE = 8192
_TRUE_PEAK_ESTIMATE_POLICY = "normalize_to_oversampled_peak_estimate"
_TRUE_PEAK_GAIN_POLICY = "normalized_to_oversampled_peak_estimate_target"
_TARGET_MEASUREMENT_DOMAIN = "continuous_source_eq"
_TARGET_PROJECTION_STAGE = "pre_design"
_TARGET_PROJECTION_GRID = "canonical_union_fft_bin_centers"


def _log_stage(name: str):
    logger.info("=== [%s] ===", name)


def _is_fatal_core_validation_result(result: dict[str, Any]) -> bool:
    metric = result.get("metric")
    status = result.get("status")
    if status != "FAIL":
        return bool(metric == "freq_response_error_db" and status == "WARN")
    if metric == "energy_front_ratio":
        return False
    return True


def _enforce_true_peak_oversampling(true_peak: bool, oversample_factor: int):
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    if true_peak and oversample_factor < 8:
        raise ValueError(
            f"true_peak=True requires oversample_factor >= 8 (got {oversample_factor})"
        )


def is_power_of_two(n: int) -> bool:
    if isinstance(n, (bool, np.bool_)):
        return False
    if not isinstance(n, (int, np.integer)):
        return False
    checked = int(n)
    return (checked & (checked - 1)) == 0 and checked != 0


def _validate_fft_size_contract(fft_size: Any) -> int:
    checked = ensure_positive_int("fft_size", fft_size)
    if checked < 2 or checked % 2 != 0:
        raise ValueError("fft_size must be an even integer >= 2")
    if not is_power_of_two(checked):
        raise ValueError("fft_size must be a power of two")
    return checked


def _validate_pipeline_inputs(
    magnitude: NDArray[np.float64] | list[float] | LinearMagnitude | DbMagnitude,
    fft_size: int,
    headroom_db: float,
    sample_rate: int,
    input_scale: str,
    design_oversample: int,
    oversample_factor: int,
    freqs_hz: NDArray[np.float64] | list[float] | None,
) -> LinearMagnitude | DbMagnitude:
    if magnitude is None:
        raise ValueError("Magnitude cannot be None")
    if isinstance(headroom_db, str):
        raise TypeError("headroom_db must be numeric")

    if freqs_hz is not None:
        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        if freqs_hz.ndim != 1 or freqs_hz.size == 0:
            raise ValueError("freqs_hz must be a non-empty 1D array")

    fft_size = _validate_fft_size_contract(fft_size)
    ensure_non_negative_float("headroom_db", headroom_db)
    ensure_positive_sample_rate(sample_rate)
    ensure_choice("input_scale", input_scale, {"linear", "db"})
    design_oversample = ensure_positive_int("design_oversample", design_oversample)
    if design_oversample != 1:
        raise ValueError("design_oversample values greater than 1 are not currently supported")
    ensure_positive_int("oversample_factor", oversample_factor)
    if freqs_hz is None and input_scale == "db":
        raise ValueError("FFT-binned input does not support input_scale='db'; provide freqs_hz for dB interpolation")

    return coerce_db_magnitude(magnitude) if input_scale == "db" else coerce_linear_magnitude(magnitude)


def _normalize_native_eq_json_input(
    magnitude: Any,
    *,
    fft_size: int,
    sample_rate: int,
    freqs_hz: NDArray[np.float64] | list[float] | None,
    input_scale: str,
) -> tuple[Any, NDArray[np.float64] | list[float] | None, str]:
    if not isinstance(magnitude, dict) or magnitude.get("type") != "eq_json":
        return magnitude, freqs_hz, input_scale

    if "eq" not in magnitude:
        raise ValueError("eq_json input must include an 'eq' payload")

    freqs = build_fft_freq_grid(fft_size, sample_rate)
    native_magnitude = eq_json_to_native_magnitude(
        magnitude["eq"],
        freqs,
        sample_rate,
    )
    return native_magnitude, None, "linear"


def _is_eq_json_input(magnitude: Any) -> bool:
    return isinstance(magnitude, dict) and magnitude.get("type") == "eq_json" and "eq" in magnitude


def _build_canonical_multi_rate_freq_grid(rates: tuple[int, ...], fft_size: int) -> np.ndarray:
    grids = [build_fft_freq_grid(fft_size, sample_rate) for sample_rate in rates]
    return np.asarray(np.unique(np.concatenate(grids)), dtype=np.float64)


def _resolve_reference_target_rate(rates: tuple[int, ...], target_sample_rate: int | None) -> int:
    return max(rates) if target_sample_rate is None else ensure_positive_sample_rate(target_sample_rate)


def _target_projection_metadata(
    *,
    rates: tuple[int, ...],
    target_sample_rate: int | None,
    design_sample_rate: int | None = None,
    interpolation_mode: str = "log",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "measurement_domain": _TARGET_MEASUREMENT_DOMAIN,
        "target_sample_rate": None if target_sample_rate is None else ensure_positive_sample_rate(target_sample_rate),
        "reference_target_rate": _resolve_reference_target_rate(rates, target_sample_rate),
        "projection_stage": _TARGET_PROJECTION_STAGE,
        "projection_grid": _TARGET_PROJECTION_GRID,
        "interpolation_mode": interpolation_mode,
    }
    if design_sample_rate is None:
        metadata["design_sample_rates"] = [int(rate) for rate in rates]
    else:
        metadata["design_sample_rate"] = ensure_positive_sample_rate(design_sample_rate)
    return metadata


def _prepare_unified_multi_rate_eq_json_target(
    magnitude: dict[str, Any],
    *,
    rates: tuple[int, ...],
    fft_size: int,
    target_sample_rate: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if "eq" not in magnitude:
        raise ValueError("eq_json input must include an 'eq' payload")
    reference_target_rate = _resolve_reference_target_rate(rates, target_sample_rate)
    master_freqs_hz = _build_canonical_multi_rate_freq_grid(rates, fft_size)
    evaluation_freqs_hz = np.minimum(master_freqs_hz, float(reference_target_rate) / 2.0)
    master_magnitude = eq_json_to_native_magnitude(
        magnitude["eq"],
        evaluation_freqs_hz,
        reference_target_rate,
    )
    return (
        np.asarray(master_magnitude, dtype=np.float64),
        np.asarray(master_freqs_hz, dtype=np.float64),
    )


def _extract_source_preamp_from_input(
    magnitude: Any,
    *,
    checked_preamp_db: float | None,
    preamp_source: str,
) -> tuple[float | None, bool, bool, str | None]:
    if isinstance(magnitude, dict) and magnitude.get("type") == "eq_json" and "eq" in magnitude:
        params = magnitude["eq"].get("data", {}).get("parameters", {})
        if "gain_db" in params:
            return validate_preamp_db(params["gain_db"]), True, True, "opra_eq_json"
        return None, False, False, "opra_eq_json"

    if checked_preamp_db is not None and preamp_source == _PREAMP_SOURCE_PEQ:
        return float(checked_preamp_db), True, False, "gain_stage"

    return None, False, False, None


def _resolve_pipeline_profile(profile: str | PipelineProfile | None, *, true_peak: bool) -> PipelineProfile:
    resolved = resolve_profile(profile)
    if resolved.true_peak_required and not true_peak:
        raise ValueError(f"Profile '{resolved.name}' requires true_peak=True")
    return resolved


def _enforce_window_contract(*, profile: PipelineProfile, window) -> None:
    if profile.enforce_no_windowing and window is not None and getattr(window, "name", None) is not None:
        raise ValueError(f"Profile '{profile.name}' forbids windowing")


def _validate_preamp_contract(
    *,
    preamp_db: float | None,
    preamp_source: str,
    preamp_already_applied: bool,
    profile: PipelineProfile,
) -> float | None:
    preamp_already_applied = ensure_bool("preamp_already_applied", preamp_already_applied)
    ensure_choice("preamp_source", preamp_source, _ALLOWED_PREAMP_SOURCES)
    checked_preamp_db = validate_preamp_db(preamp_db)
    if preamp_already_applied and checked_preamp_db is not None:
        raise ValueError("Preamp would be applied twice: input is already marked as preamped")
    if checked_preamp_db is None and preamp_source != _PREAMP_SOURCE_NONE:
        raise ValueError("preamp_source must be 'none' when preamp_db is not provided")
    if profile.forbid_manual_preamp and checked_preamp_db is not None and preamp_source != _PREAMP_SOURCE_PEQ:
        raise ValueError(
            "This profile only allows preamp sourced from trusted pipeline data."
        )
    if profile.enforce_peq_preamp and preamp_source not in {_PREAMP_SOURCE_NONE, _PREAMP_SOURCE_PEQ}:
        raise ValueError(f"Profile '{profile.name}' only permits 'none' or 'peq' preamp sources")
    return checked_preamp_db


def _verify_preamped_target_contract(
    target_magnitude: np.ndarray,
    *,
    preamp_db: float | None,
    source_preamp_used: bool,
    enforce_absolute_peak: bool,
    headroom_db: float,
    tolerance: float = 1e-9,
) -> None:
    if not enforce_absolute_peak:
        return

    peak_linear = float(np.max(target_magnitude))
    if peak_linear <= 0.0:
        return

    if peak_linear > 1.0 + tolerance:
        peak_db = 20.0 * np.log10(peak_linear)
        raise ValueError(
            "Target/preamp contract failed: preamp is insufficient; post-preamp target peak "
            f"{peak_db:.3f} dB exceeds 0 dBFS safety before FIR normalization. "
            "Lower the PEQ preamp, reduce boost, or use the default shape-only profile."
        )

    requested_headroom_linear = 10.0 ** (-float(headroom_db) / 20.0)
    logger.info(
        "Post-preamp target contract: peak=%.6f (%.3f dBFS), requested_output_headroom_peak=%.6f (%.2f dBFS)",
        peak_linear,
        20.0 * np.log10(max(peak_linear, np.finfo(np.float64).tiny)),
        requested_headroom_linear,
        -float(headroom_db),
    )


def _verify_target_is_not_silent(target_magnitude: np.ndarray) -> None:
    peak_linear = float(np.max(np.asarray(target_magnitude, dtype=np.float64)))
    if peak_linear <= _TARGET_SILENCE_FLOOR:
        raise ValueError(
            "Target magnitude is silent; refusing to normalize a silent FIR target"
        )


def _verify_target_magnitude_is_practical(target_magnitude: np.ndarray) -> None:
    peak_linear = float(np.max(np.asarray(target_magnitude, dtype=np.float64)))
    if peak_linear > _MAX_TARGET_MAGNITUDE:
        peak_db = 20.0 * np.log10(peak_linear)
        limit_db = 20.0 * np.log10(_MAX_TARGET_MAGNITUDE)
        raise ValueError(
            "Target magnitude is outside the practical generation range: "
            f"peak {peak_db:.3f} dB exceeds {limit_db:.1f} dB"
        )


def _normalize_target_shape(target_magnitude: np.ndarray) -> np.ndarray:
    target_magnitude = np.asarray(target_magnitude, dtype=np.float64)
    peak_linear = float(np.max(target_magnitude))
    if not np.isfinite(peak_linear) or peak_linear <= 0.0:
        raise ValueError("Target magnitude is silent; refusing to normalize a silent FIR target")
    normalized = np.asarray(target_magnitude / peak_linear, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Target magnitude normalization produced non-finite values")
    return normalized


def _verify_final_true_peak_contract(
    fir: np.ndarray,
    *,
    true_peak_target_dbfs: float,
    oversample_factor: int,
    tolerance_db: float = 0.01,
    min_safe_margin_db: float = _MIN_SAFE_TRUE_PEAK_MARGIN_DB,
) -> None:
    true_peak_target_dbfs = -float(ensure_non_negative_float("true_peak_target_headroom_db", -true_peak_target_dbfs))
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    min_safe_margin_db = ensure_non_negative_float("min_safe_margin_db", min_safe_margin_db)
    measured_true_peak_linear = compute_true_peak(fir, oversample_factor=oversample_factor)
    measured_true_peak_dbfs = 20.0 * np.log10(max(measured_true_peak_linear, np.finfo(np.float64).tiny))
    target_linear = 10.0 ** (true_peak_target_dbfs / 20.0)
    allowed_linear = target_linear * (10.0 ** (float(tolerance_db) / 20.0))
    margin_linear = target_linear - measured_true_peak_linear
    margin_db = true_peak_target_dbfs - measured_true_peak_dbfs

    logger.info(
        "Final FIR true-peak contract: measured=%.6f (%.3f dBFS), target=%.6f (%.3f dBFS), margin=%.6f (%.3f dB)",
        measured_true_peak_linear,
        measured_true_peak_dbfs,
        target_linear,
        true_peak_target_dbfs,
        margin_linear,
        margin_db,
    )

    if measured_true_peak_linear > allowed_linear:
        raise ValueError(
            "True-peak contract failed: final FIR true peak "
            f"{measured_true_peak_dbfs:.3f} dBFS exceeds target {true_peak_target_dbfs:.3f} dBFS. "
            "Enable true_peak normalization or increase output headroom."
        )
    if margin_db < -float(tolerance_db):
        raise ValueError(
            "Negative true-peak margin detected beyond verification tolerance. "
            "This indicates an internal verification inconsistency."
        )
    if margin_db + TRUE_PEAK_MARGIN_TOLERANCE_DB < float(min_safe_margin_db):
        logger.warning(
            "True-peak margin is very low: %.4f dB (target %.2f dBFS, minimum safe margin %.2f dB). "
            "Filter is technically valid but fragile to reconstruction or rounding.",
            margin_db,
            true_peak_target_dbfs,
            float(min_safe_margin_db),
        )


def _prepare_target_magnitude(
    magnitude: LinearMagnitude | DbMagnitude,
    fft_size: int,
    sample_rate: int,
    freqs_hz: NDArray[np.float64] | list[float] | None,
    input_scale: str,
    interpolation_mode: str = "log",
) -> np.ndarray:
    values = magnitude.as_array()

    # ---- FFT BINNED INPUT ----
    if freqs_hz is None:
        expected_len = fft_size // 2 + 1
        if values.size != expected_len:
            raise ValueError(
                f"Magnitude length must be fft_size//2 + 1 ({expected_len}), got {values.size}"
            )
        if isinstance(magnitude, DbMagnitude):
            raise TypeError(
                "FFT-binned dB input is not supported; provide freqs_hz for dB interpolation"
            )
        return np.asarray(values, dtype=np.float64)

    # ---- INTERPOLATED INPUT ----
    fft_freqs = build_fft_freq_grid(fft_size, sample_rate)
    freqs_array = np.asarray(freqs_hz, dtype=np.float64)
    if freqs_array.shape == fft_freqs.shape and np.array_equal(freqs_array, fft_freqs):
        return np.asarray(
            db_to_linear(values) if isinstance(magnitude, DbMagnitude) else values,
            dtype=np.float64,
        )

    nyquist = sample_rate / 2.0
    if freqs_hz[-1] < nyquist:
        raise ValueError(
            f"Input response must extend to Nyquist ({nyquist} Hz), got {freqs_hz[-1]} Hz"
        )

    if input_scale == "linear":
        if interpolation_mode == "log":
            target_magnitude = interpolate_log_frequency_response(
                freqs_hz,
                values,
                fft_freqs
            )
        else:
            target_magnitude = np.interp(
                fft_freqs,
                freqs_hz,
                values
            )
    else:
        interp_db = interpolate_log_frequency_response(
            freqs_hz,
            values,
            fft_freqs
        )
        target_magnitude = db_to_linear(interp_db)

    return np.asarray(target_magnitude, dtype=np.float64)



def _request_payload(
    magnitude: np.ndarray,
    spec: PipelineSpec,
    freqs_hz: NDArray[np.float64] | list[float] | None,
) -> dict:
    payload = {
        "fft_size": int(spec.fft_size),
        "requested_headroom_db": float(spec.requested_headroom_db),
        "normalization_headroom_db": float(spec.normalization_headroom_db),
        "sample_rate": int(spec.sample_rate),
        "input_scale": spec.input_scale,
        "minimum_phase": bool(spec.minimum_phase),
        "true_peak": bool(spec.true_peak),
        "oversample_factor": int(spec.oversample_factor),
        "window_type": None if spec.window is None else spec.window.name,
        "window_beta": None if spec.window is None or spec.window.beta is None else float(spec.window.beta),
        "window_preset": None if spec.window is None else spec.window.preset,
        "design_oversample": int(spec.design_oversample),
        "design_method": "minimum_phase_cepstrum_from_target_magnitude" if spec.minimum_phase else "fft_linear_phase_from_target_magnitude",
        "target_shape_normalization": "unit_peak_before_design",
        "interpolation_mode": spec.interpolation_mode,
        "true_peak_measurement": {
            "method": "polyphase_oversampled_peak_estimate",
            "oversample_factor": int(spec.oversample_factor),
            "exact": False,
        },
        "freqs_hz": None if freqs_hz is None else np.asarray(freqs_hz, dtype=np.float64).tolist(),
        "magnitude": np.asarray(magnitude, dtype=np.float64).tolist(),
        "gain_stage_preamp_source": spec.gain_stage_preamp_source,
        "preamp_applied_as_gain_stage": bool(spec.preamp_applied_as_gain_stage),
        "source_preamp_db": None if spec.source_preamp_db is None else float(spec.source_preamp_db),
        "source_preamp_present": bool(spec.source_preamp_present),
        "source_preamp_used_for_target_shape": bool(spec.source_preamp_used_for_target_shape),
        "source_preamp_origin": spec.source_preamp_origin,
        "post_scale_attenuation_db": float(spec.post_scale_attenuation_db),
        "profile": spec.profile.name,
        "minimum_true_peak_margin_db": float(spec.minimum_true_peak_margin_db),
    }
    if spec.target_projection is not None:
        payload["target_projection"] = spec.target_projection
    return payload


def _build_pipeline_spec(
    *,
    sample_rate: int,
    fft_size: int,
    input_scale: str,
    requested_headroom_db: float,
    true_peak: bool,
    oversample_factor: int,
    design_oversample: int,
    window,
    gain_stage_preamp_source: str,
    preamp_applied_as_gain_stage: bool,
    source_preamp_db: float | None = None,
    source_preamp_present: bool = False,
    source_preamp_used_for_target_shape: bool = False,
    source_preamp_origin: str | None = None,
    profile: PipelineProfile,
    minimum_phase: bool,
    interpolation_mode: str,
    target_projection: dict[str, Any] | None = None,
) -> PipelineSpec:
    return PipelineSpec.from_profile(
        sample_rate=sample_rate,
        fft_size=fft_size,
        input_scale=input_scale,
        requested_headroom_db=requested_headroom_db,
        true_peak=true_peak,
        oversample_factor=oversample_factor,
        design_oversample=design_oversample,
        window=window,
        gain_stage_preamp_source=gain_stage_preamp_source,
        preamp_applied_as_gain_stage=preamp_applied_as_gain_stage,
        source_preamp_db=source_preamp_db,
        source_preamp_present=source_preamp_present,
        source_preamp_used_for_target_shape=source_preamp_used_for_target_shape,
        source_preamp_origin=source_preamp_origin,
        profile=profile,
        minimum_phase=minimum_phase,
        interpolation_mode=interpolation_mode,
        target_projection=target_projection,
    )


def generate_fir_from_spec(
    spec: PipelineSpec,
    magnitude,
    *,
    freqs_hz=None,
    preamp_db=None,
    return_details=False,
    raw_input_magnitude=None,
    preamp_already_applied=False,
):
    spec = spec.copy()

    if isinstance(spec.requested_headroom_db, str):
        raise TypeError("PipelineSpec.requested_headroom_db must be numeric")

    return_details = ensure_bool("return_details", return_details)
    preamp_already_applied = ensure_bool("preamp_already_applied", preamp_already_applied)

    # -------------------------------------------------
    # NORMALIZE INPUT (eq_json → native magnitude)
    # -------------------------------------------------
    raw_input_magnitude = magnitude if raw_input_magnitude is None else raw_input_magnitude

    resolved_profile = _resolve_pipeline_profile(
        spec.profile,
        true_peak=spec.true_peak
    )
    _enforce_window_contract(profile=resolved_profile, window=spec.window)

    magnitude, freqs_hz, _ = _normalize_native_eq_json_input(
        magnitude,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        freqs_hz=freqs_hz,
        input_scale=spec.input_scale,
    )

    # -------------------------------------------------
    # ALWAYS RESOLVE PROFILE (single source of truth)
    # -------------------------------------------------

    _log_stage("INPUT VALIDATION")
    validated_magnitude = _validate_pipeline_inputs(
        magnitude=magnitude,
        fft_size=spec.fft_size,
        headroom_db=spec.requested_headroom_db,
        sample_rate=spec.sample_rate,
        input_scale=spec.input_scale,
        design_oversample=spec.design_oversample,
        oversample_factor=spec.oversample_factor,
        freqs_hz=freqs_hz,
    )
    _log_stage("PROFILE RESOLVED")
    _enforce_true_peak_oversampling(spec.true_peak, spec.oversample_factor)
    if float(spec.minimum_true_peak_margin_db) < _MIN_SAFE_TRUE_PEAK_MARGIN_DB:
        raise ValueError(
            f"PipelineSpec minimum_true_peak_margin_db={spec.minimum_true_peak_margin_db:.3f} is below the global minimum safety floor"
        )

    checked_preamp_db = _validate_preamp_contract(
        preamp_db=preamp_db,
        preamp_source=spec.gain_stage_preamp_source,
        preamp_already_applied=preamp_already_applied,
        profile=resolved_profile
    )
    source_preamp_db, source_preamp_present, source_preamp_used_for_target_shape, source_preamp_origin = _extract_source_preamp_from_input(
        raw_input_magnitude,
        checked_preamp_db=checked_preamp_db,
        preamp_source=spec.gain_stage_preamp_source,
    )
    _log_stage("PIPELINE START")
    start_time = time.perf_counter()

    phase_mode = "minimum" if spec.minimum_phase else "linear"

    logger.info(
        "Generating FIR: phase_mode=%s profile=%s sample_rate=%s fft_size=%s requested_headroom_db=%.2f normalization_headroom_db=%.2f true_peak=%s design_oversample=%s preamp_db=%s preamp_source=%s",
        phase_mode,
        resolved_profile.name,
        spec.sample_rate,
        spec.fft_size,
        float(spec.requested_headroom_db),
        float(spec.normalization_headroom_db),
        spec.true_peak,
        spec.design_oversample,
        "none" if checked_preamp_db is None else f"{float(checked_preamp_db):.2f}",
        spec.gain_stage_preamp_source,
    )
    _log_stage("TARGET PREPARATION")
    target_magnitude = _prepare_target_magnitude(
        magnitude=validated_magnitude,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        freqs_hz=freqs_hz,
        input_scale=spec.input_scale,
        interpolation_mode=spec.interpolation_mode,
    )
    _log_stage("PREAMP APPLICATION")
    target_magnitude = apply_preamp_db(LinearMagnitude(target_magnitude), checked_preamp_db)

    _verify_target_is_not_silent(target_magnitude)
    target_magnitude = _normalize_target_shape(target_magnitude)

    _verify_preamped_target_contract(
        target_magnitude,
        preamp_db=checked_preamp_db,
        source_preamp_used=source_preamp_used_for_target_shape,
        enforce_absolute_peak=resolved_profile.enforce_absolute_peak,
        headroom_db=spec.requested_headroom_db,
    )
    _verify_target_magnitude_is_practical(target_magnitude)

    target_magnitude = np.asarray(target_magnitude, dtype=np.float64)
    preamp_applied_as_gain_stage = bool(checked_preamp_db is not None)
    spec = replace(
        spec,
        source_preamp_db=source_preamp_db,
        source_preamp_present=bool(source_preamp_present),
        source_preamp_used_for_target_shape=bool(source_preamp_used_for_target_shape),
        source_preamp_origin=source_preamp_origin,
        preamp_applied_as_gain_stage=preamp_applied_as_gain_stage,
    )

    _log_stage("FIR DESIGN")

    # -----------------------------
    # 1. FIR DESIGN (pure shape)
    # -----------------------------
    fir_linear: np.ndarray | None = None
    if spec.minimum_phase:
        fir_phase = minimum_phase_from_mag(target_magnitude, spec.fft_size)
    else:
        fir_linear = design_fir_from_mag_fft(
            magnitude=target_magnitude,
            fft_size=spec.fft_size,
            minimum_phase=False,
        )

    # -----------------------------
    # 2. PHASE CONTROL (single path)
    # -----------------------------
    if not spec.minimum_phase:
        if fir_linear is None:
            raise RuntimeError("linear-phase design did not produce FIR taps")
        fir_linear = np.asarray(fir_linear, dtype=np.float64)
        fir_phase = fir_linear

    fir_phase = np.asarray(fir_phase, dtype=np.float64)
    fir_linear_details = fir_phase if fir_linear is None else fir_linear

    # -----------------------------
    # 3. WINDOW (causal-safe)
    # -----------------------------
    if spec.window is None or spec.window.name is None:
        fir_windowed = fir_phase
    else:
        fir_windowed = apply_window_to_fir(fir_phase, spec.window, minimum_phase=spec.minimum_phase)

    fir_windowed = np.asarray(fir_windowed, dtype=np.float64)

    # -----------------------------
    # 4. FINAL SCALING (ONLY ONCE)
    # -----------------------------
    fir_final = apply_tap_scaling(
        fir_windowed,
        headroom_db=spec.normalization_headroom_db,
        true_peak=spec.true_peak,
        oversample_factor=spec.oversample_factor,
        mode="neutral"
    )

    fir_final = np.asarray(fir_final, dtype=np.float64)

    # -----------------------------
    # 5. RESPONSE ANALYSIS
    # -----------------------------
    pre_window_response = compute_frequency_response(
        fir_phase,
        fft_size=spec.fft_size
    )
    pre_window_error = summarize_response_error(
        target_magnitude,
        pre_window_response
    )

    windowed_response = compute_frequency_response(
        fir_windowed,
        fft_size=spec.fft_size
    )
    windowed_error = summarize_response_error(
        target_magnitude,
        windowed_response
    )

    final_response = compute_frequency_response(
        fir_final,
        fft_size=spec.fft_size
    )
    final_error = summarize_response_error(
        target_magnitude,
        final_response
    )

    # -----------------------------
    # 6. LOGGING (critical insight)
    # -----------------------------
    logger.info(
        "Pre-window error: max=%.6f dB rms=%.6f dB",
        pre_window_error.max_abs_error_db,
        pre_window_error.rms_error_db
    )

    logger.info(
        "Windowed error: max=%.6f dB rms=%.6f dB",
        windowed_error.max_abs_error_db,
        windowed_error.rms_error_db
    )

    logger.info(
        "Final error: max=%.6f dB rms=%.6f dB",
        final_error.max_abs_error_db,
        final_error.rms_error_db
    )

    # -----------------------------
    # 7. CONSISTENT OUTPUT STATE
    # -----------------------------
    _log_stage("HEADROOM CHECK")
    verify_headroom(
        fir_final,
        headroom_db=spec.normalization_headroom_db,
        true_peak=spec.true_peak,
        oversample_factor=spec.oversample_factor,
    )
    if spec.true_peak:
        _log_stage("TRUE PEAK CHECK")
        _verify_final_true_peak_contract(
            fir_final,
            true_peak_target_dbfs=spec.true_peak_target_dbfs,
            oversample_factor=spec.oversample_factor,
            min_safe_margin_db=float(spec.minimum_true_peak_margin_db),
        )
    else:
        logger.info("Skipping true-peak contract check because true_peak=False")

    if not np.all(np.isfinite(fir_final)):
        raise ValueError("FIR contains NaN or Inf")

    _log_stage("CORE VALIDATION")
    results = run_core_validations(
        fir_final,
        mode="eq",
        sample_rate=spec.sample_rate,
        target_mag=target_magnitude,
        fft_size=spec.fft_size,
    )

    logger.info("=== CORE VALIDATION RESULTS ===")
    logger.info(json.dumps(results, indent=2))

    if any(_is_fatal_core_validation_result(r) for r in results):
        raise RuntimeError(
            f"FIR validation failed:\n{json.dumps(results, indent=2)}"
        )

    fingerprint = coeff_hash_sha256(fir_final)
    logger.info("FIR fingerprint: %s", fingerprint)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info("FIR generation finished in %.2f ms", elapsed_ms)

    actual_magnitude = final_response
    target_magnitude_db = linear_to_db(target_magnitude)
    actual_magnitude_db = linear_to_db(actual_magnitude)
    error = final_error
    latency = summarize_latency(fir_final, int(spec.sample_rate))
    verification = summarize_verification(
        fir=fir_final,
        target_magnitude=target_magnitude,
        oversample_factor=spec.oversample_factor,
        true_peak_target_dbfs=spec.true_peak_target_dbfs,
        true_peak_min_safe_margin_db=float(spec.minimum_true_peak_margin_db),
        gain_stage_preamp_source=spec.gain_stage_preamp_source,
        preamp_applied_as_gain_stage=preamp_applied_as_gain_stage,
        source_preamp_db=source_preamp_db,
        source_preamp_present=source_preamp_present,
        source_preamp_used_for_target_shape=source_preamp_used_for_target_shape,
        source_preamp_origin=source_preamp_origin,
        final_baked_headroom_db=float(spec.normalization_headroom_db),
        final_gain_policy=(_TRUE_PEAK_GAIN_POLICY if spec.true_peak else "normalized_to_sample_peak_target"),
        true_peak_policy=(_TRUE_PEAK_ESTIMATE_POLICY if spec.true_peak else "measure_only"),
        true_peak_target_is_baked_safety_ceiling=bool(spec.true_peak),
        true_peak_margin_warning_enabled=bool(spec.true_peak),
        request_payload=_request_payload(
            magnitude=validated_magnitude.as_array(),
            spec=spec,
            freqs_hz=freqs_hz,
        ),
    )
    gain_traceability = GainTraceabilitySummary(
        source_preamp_db=source_preamp_db,
        source_preamp_present=bool(source_preamp_present),
        source_preamp_used_for_target_shape=bool(source_preamp_used_for_target_shape),
        source_preamp_origin=source_preamp_origin,
        gain_stage_preamp_source=spec.gain_stage_preamp_source,
        preamp_applied_as_gain_stage=bool(preamp_applied_as_gain_stage),
        final_baked_headroom_db=float(spec.normalization_headroom_db),
        final_gain_policy=(_TRUE_PEAK_GAIN_POLICY if spec.true_peak else "normalized_to_sample_peak_target"),
        true_peak_policy=(_TRUE_PEAK_ESTIMATE_POLICY if spec.true_peak else "measure_only"),
        true_peak_target_is_baked_safety_ceiling=bool(spec.true_peak),
    )
    _log_stage("SYSTEM VALIDATION")
    system_result = validate_system(
        fir_final,
        sample_rate=spec.sample_rate,
        target_magnitude=target_magnitude,
        fft_size=spec.fft_size,
        true_peak_enforced=bool(spec.true_peak),
        true_peak_target_dbfs=float(spec.true_peak_target_dbfs),
        profile=resolved_profile
    )

    logger.info("=== SYSTEM VALIDATION ===")
    logger.info("Status: %s", system_result.status)

    for v in system_result.violations:
        logger.error("VIOLATION: %s", v)

    for w in system_result.warnings:
        logger.warning("WARNING: %s", w)

    if system_result.status == "FAIL":
        raise RuntimeError("System validation failed")

    if not return_details:
        return fir_final

    logger.info(
        "Verification summary: sample_peak=%.3f dBFS true_peak=%.3f dBFS tp_margin=%.3f dB max_error=%.4f dB",
        verification.fir_peak_dbfs,
        verification.fir_true_peak_dbfs,
        verification.true_peak_margin_db,
        error.max_abs_error_db,
    )
    if verification.true_peak_margin_warning:
        logger.warning(
            "Verification summary warning: true-peak margin %.4f dB is below safe threshold %.2f dB.",
            verification.true_peak_margin_db,
            verification.true_peak_min_safe_margin_db,
        )


    return PipelineResult(
        sample_rate=int(spec.sample_rate),
        fft_size=int(spec.fft_size),
        input_scale=spec.input_scale,
        headroom_db=float(spec.requested_headroom_db),
        true_peak=bool(spec.true_peak),
        oversample_factor=int(spec.oversample_factor),
        design_oversample=int(spec.design_oversample),
        window=spec.window,
        spec=spec,
        profile=resolved_profile,
        target_magnitude=np.asarray(target_magnitude, dtype=np.float64),
        fir_linear=np.asarray(fir_linear_details, dtype=np.float64),
        fir_windowed=np.asarray(fir_windowed, dtype=np.float64),
        fir_final=np.asarray(fir_final, dtype=np.float64),
        actual_magnitude=np.asarray(actual_magnitude, dtype=np.float64),
        actual_magnitude_db=np.asarray(actual_magnitude_db, dtype=np.float64),
        target_magnitude_db=np.asarray(target_magnitude_db, dtype=np.float64),
        error=error,
        latency=latency,
        verification=verification,
        gain_traceability=gain_traceability,
        system_validation=system_result,
    )


def generate_fir_pipeline(
    magnitude: NDArray[np.float64] | list[float] | LinearMagnitude | DbMagnitude | dict[str, Any],
    fft_size: int,
    headroom_db: float,
    sample_rate: int = 48000,
    freqs_hz: NDArray[np.float64] | list[float] | None = None,
    input_scale: str = "linear",
    true_peak: bool = True,
    oversample_factor: int = 8,
    window_type: str | None = None,
    window_beta: float = 8.6,
    window_preset: str | None = None,
    design_oversample: int = 1,
    preamp_db: float | None = None,
    return_details: bool = False,
    *,
    preamp_source: str = _PREAMP_SOURCE_NONE,
    preamp_already_applied: bool = False,
    profile: str | PipelineProfile | None = None,
    _raw_input_magnitude: Any | None = None,
    _target_projection: dict[str, Any] | None = None,
    minimum_phase: bool = True,
    interpolation_mode: str = "log"
) -> np.ndarray | PipelineResult:
    true_peak = ensure_bool("true_peak", true_peak)
    return_details = ensure_bool("return_details", return_details)
    preamp_already_applied = ensure_bool("preamp_already_applied", preamp_already_applied)
    if isinstance(headroom_db, (bool, np.bool_)):
        raise TypeError("headroom_db must be numeric, not bool")
    if isinstance(headroom_db, str):
        raise TypeError("headroom_db must be numeric")
    if isinstance(fft_size, (bool, np.bool_)):
        raise TypeError("fft_size must be an integer, not bool")

    resolved_profile = _resolve_pipeline_profile(profile, true_peak=true_peak)
    _enforce_true_peak_oversampling(true_peak, oversample_factor)
    if window_type is None and window_preset is None:
        window_spec = None
    else:
        window_spec = resolve_window(window_type=window_type, beta=window_beta, preset=window_preset)
    _enforce_window_contract(profile=resolved_profile, window=window_spec)
    checked_preamp_db = _validate_preamp_contract(
        preamp_db=preamp_db,
        preamp_source=preamp_source,
        preamp_already_applied=preamp_already_applied,
        profile=resolved_profile,
    )
    spec = _build_pipeline_spec(
        sample_rate=sample_rate,
        fft_size=fft_size,
        input_scale=input_scale,
        requested_headroom_db=headroom_db,
        true_peak=true_peak,
        oversample_factor=oversample_factor,
        design_oversample=design_oversample,
        window=window_spec,
        gain_stage_preamp_source=preamp_source,
        preamp_applied_as_gain_stage=False,
        source_preamp_db=None,
        source_preamp_present=False,
        source_preamp_used_for_target_shape=False,
        source_preamp_origin=None,
        profile=resolved_profile,
        minimum_phase=minimum_phase,
        interpolation_mode=interpolation_mode,
        target_projection=_target_projection,
    )
    return generate_fir_from_spec(
        spec.copy(),
        magnitude,
        freqs_hz=freqs_hz,
        preamp_db=checked_preamp_db,
        return_details=return_details,
        raw_input_magnitude=_raw_input_magnitude,
        preamp_already_applied=preamp_already_applied,
    )


def generate_fir_multi_rate(
    magnitude: NDArray[np.float64] | list[float] | LinearMagnitude | DbMagnitude | dict[str, Any],
    rates: Iterable[int | float],
    fft_size: int,
    headroom_db: float,
    freqs_hz: NDArray[np.float64] | list[float] | None = None,
    input_scale: str = "linear",
    true_peak: bool = True,
    oversample_factor: int = 8,
    window_type: str | None = None,
    window_beta: float = 8.6,
    window_preset: str | None = None,
    design_oversample: int = 1,
    preamp_db: float | None = None,
    return_details: bool = False,
    *,
    preamp_source: str = _PREAMP_SOURCE_NONE,
    preamp_already_applied: bool = False,
    profile: str | PipelineProfile | None = None,
    interpolation_mode: str = "log",
    target_sample_rate: int | None = None,
) -> dict[int, np.ndarray | PipelineResult]:
    true_peak = ensure_bool("true_peak", true_peak)
    return_details = ensure_bool("return_details", return_details)
    preamp_already_applied = ensure_bool("preamp_already_applied", preamp_already_applied)
    if isinstance(headroom_db, (bool, np.bool_)):
        raise TypeError("headroom_db must be numeric, not bool")
    if isinstance(fft_size, (bool, np.bool_)):
        raise TypeError("fft_size must be an integer, not bool")
    normalized_rates = normalize_rates(rates)
    shared_magnitude = magnitude
    shared_freqs_hz = freqs_hz
    shared_input_scale = input_scale
    raw_input_magnitude = None
    target_projection: dict[str, Any] | None = None
    if _is_eq_json_input(magnitude) and (len(normalized_rates) > 1 or target_sample_rate is not None):
        shared_magnitude, shared_freqs_hz = _prepare_unified_multi_rate_eq_json_target(
            magnitude,
            rates=tuple(normalized_rates),
            fft_size=_validate_fft_size_contract(fft_size),
            target_sample_rate=target_sample_rate,
        )
        shared_input_scale = "linear"
        raw_input_magnitude = magnitude
        target_projection = _target_projection_metadata(
            rates=tuple(normalized_rates),
            target_sample_rate=target_sample_rate,
            interpolation_mode=interpolation_mode,
        )

    def build_rate(sr: int) -> np.ndarray | PipelineResult:
        return generate_fir_pipeline(
            magnitude=shared_magnitude,
            freqs_hz=shared_freqs_hz,
            fft_size=fft_size,
            sample_rate=sr,
            headroom_db=headroom_db,
            input_scale=shared_input_scale,
            true_peak=true_peak,
            oversample_factor=oversample_factor,
            window_type=window_type,
            window_beta=window_beta,
            window_preset=window_preset,
            design_oversample=design_oversample,
            preamp_db=preamp_db,
            return_details=return_details,
            preamp_source=preamp_source,
            preamp_already_applied=preamp_already_applied,
            profile=profile,
            _raw_input_magnitude=raw_input_magnitude,
            _target_projection=None if target_projection is None else _target_projection_metadata(
                rates=tuple(normalized_rates),
                target_sample_rate=target_sample_rate,
                design_sample_rate=sr,
                interpolation_mode=interpolation_mode,
            ),
            interpolation_mode=interpolation_mode,
        )

    if len(normalized_rates) <= 1 or int(fft_size) < _PARALLEL_MULTI_RATE_MIN_FFT_SIZE:
        return {sr: build_rate(sr) for sr in normalized_rates}

    results_by_rate: dict[int, np.ndarray | PipelineResult] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(normalized_rates))) as executor:
        future_to_rate = {executor.submit(build_rate, sr): sr for sr in normalized_rates}
        for future in as_completed(future_to_rate):
            results_by_rate[future_to_rate[future]] = future.result()
    return {sr: results_by_rate[sr] for sr in normalized_rates}
