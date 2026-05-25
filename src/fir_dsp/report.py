from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .artifact_metrics import (
    _aligned_error_db,
    artifact_contract,
    audible_target_verdict,
    canonical_wav_array,
    effective_length,
    export_parity,
    float_safety,
    gain_summary,
    perceptual_weighted_error_from_aligned_error,
    reconstruction_error,
    target_error_band_from_aligned_error,
    true_peak_stress_summary,
)
from .core import dataclass_to_json_ready
from .validation import ensure_bool, ensure_non_negative_float, ensure_positive_int


def _round_floats_for_reproducibility(value: object, precision: int = 9) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata contains a non-finite float")
        return round(value, precision)
    if isinstance(value, list):
        return [_round_floats_for_reproducibility(item, precision=precision) for item in value]
    if isinstance(value, tuple):
        return [_round_floats_for_reproducibility(item, precision=precision) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _round_floats_for_reproducibility(nested, precision=precision)
            for key, nested in value.items()
        }
    return value


def metadata_json_dumps(
    payload: object,
    *,
    reproducible: bool = False,
    precision: int = 9,
) -> str:
    reproducible = ensure_bool("reproducible", reproducible)
    precision = ensure_positive_int("precision", precision)
    ready = dataclass_to_json_ready(payload)
    if reproducible:
        ready = _round_floats_for_reproducibility(ready, precision=precision)
    return json.dumps(
        ready,
        indent=2,
        sort_keys=reproducible,
        allow_nan=False,
    )


def _runtime_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "byte_order": sys.byteorder,
        "float_dtype": "float64",
    }


def _spec_metadata(result: Any, mode: str | None = None) -> dict[str, object]:
    spec = dataclass_to_json_ready(result.spec)
    assert isinstance(spec, dict)
    verification = result.verification
    gain_traceability = result.gain_traceability
    spec["headroom_db"] = float(result.spec.requested_headroom_db)
    spec["requested_headroom_db"] = float(result.spec.requested_headroom_db)
    spec["normalization_headroom_db"] = float(result.spec.normalization_headroom_db)
    spec["final_baked_headroom_db"] = float(
        getattr(verification, "final_baked_headroom_db", gain_traceability.final_baked_headroom_db)
    )
    spec["final_gain_policy"] = str(
        getattr(verification, "final_gain_policy", gain_traceability.final_gain_policy)
    )
    spec["true_peak_policy"] = str(
        getattr(verification, "true_peak_policy", gain_traceability.true_peak_policy)
    )
    spec["true_peak_target_is_baked_safety_ceiling"] = bool(
        getattr(
            verification,
            "true_peak_target_is_baked_safety_ceiling",
            gain_traceability.true_peak_target_is_baked_safety_ceiling,
        )
    )
    spec["minimum_phase"] = True
    spec["phase_mode"] = "minimum"
    spec["mode"] = mode
    spec["num_taps"] = int(len(result.fir_final))
    spec["taps"] = int(len(result.fir_final))

    window_meta = dataclass_to_json_ready(result.window)
    if window_meta is None:
        window_meta = {"name": None, "beta": None, "preset": None}
    assert isinstance(window_meta, dict)
    spec["window_type"] = window_meta.get("name")
    spec["window_beta"] = window_meta.get("beta")
    spec["window_preset"] = window_meta.get("preset")
    return spec


def pipeline_metadata(
    result: Any,
    elapsed_ms: float | None = None,
    mode: str | None = None,
    target_validation: Any | None = None,
    cross_rate_consistency: dict[str, object] | None = None,
    include_stress_probes: bool = False,
) -> dict[str, object]:
    include_stress_probes = ensure_bool("include_stress_probes", include_stress_probes)
    exported_wav = canonical_wav_array(result.fir_final)
    aligned_freqs_hz, aligned_error_db = _aligned_error_db(
        result.target_magnitude,
        result.actual_magnitude,
        sample_rate=result.sample_rate,
        fft_size=result.fft_size,
    )
    listening_band_target_error_10k = target_error_band_from_aligned_error(
        aligned_freqs_hz,
        aligned_error_db,
        min_freq_hz=20.0,
        max_freq_hz=min(10_000.0, float(result.sample_rate) / 2.0),
        label="20_10000_hz",
    )
    listening_band_target_error_18k = target_error_band_from_aligned_error(
        aligned_freqs_hz,
        aligned_error_db,
        min_freq_hz=20.0,
        max_freq_hz=min(18_000.0, float(result.sample_rate) / 2.0),
        label="20_18000_hz",
    )
    num_taps = int(len(result.fir_final))
    window_meta = dataclass_to_json_ready(result.window)
    if window_meta is None:
        window_meta = {"name": None, "beta": None, "preset": None}
    assert isinstance(window_meta, dict)
    metadata: dict[str, object] = {
        "sample_rate": result.sample_rate,
        "fft_size": result.fft_size,
        "num_taps": num_taps,
        "taps": num_taps,
        "window_type": window_meta.get("name"),
        "window_beta": window_meta.get("beta"),
        "window_preset": window_meta.get("preset"),
        "input_scale": result.input_scale,
        "interpolation_mode": result.spec.interpolation_mode,
        "headroom_db": result.headroom_db,
        "requested_headroom_db": result.spec.requested_headroom_db,
        "normalization_headroom_db": result.spec.normalization_headroom_db,
        "minimum_phase": True,
        "phase_mode": "minimum",
        "locked_profile": ("Mart Reference" if mode == "mart_reference" else None),
        "true_peak": result.true_peak,
        "oversample_factor": result.oversample_factor,
        "design_oversample": result.design_oversample,
        "mode": mode,
        "design_method": "minimum_phase_cepstrum_from_target_magnitude" if result.spec.minimum_phase else "fft_linear_phase_from_target_magnitude",
        "target_shape_normalization": "unit_peak_before_design",
        "true_peak_measurement": {
            "method": "polyphase_oversampled_peak_estimate",
            "oversample_factor": int(result.oversample_factor),
            "exact": False,
        },
        "determinism_scope": "bit-identical within the same Python/NumPy/SciPy/runtime stack; exported metadata records the stack for cross-system audits",
        "runtime": _runtime_metadata(),
        "window": window_meta,
        "spec": _spec_metadata(result, mode=mode),
        "profile": getattr(result.profile, "name", result.profile),
        "latency": dataclass_to_json_ready(result.latency),
        "error": dataclass_to_json_ready(result.error),
        "verification": dataclass_to_json_ready(result.verification),
        "gain_traceability": dataclass_to_json_ready(result.gain_traceability),
        "artifact_contract": dataclass_to_json_ready(artifact_contract(exported_wav, result.sample_rate)),
        "export_parity": dataclass_to_json_ready(export_parity(result.fir_final, exported_wav)),
        "effective_length": dataclass_to_json_ready(effective_length(result.fir_final, result.sample_rate)),
        "gain_summary": dataclass_to_json_ready(gain_summary(result.fir_final, result.sample_rate, fft_size=result.fft_size)),
        "exported_wav_reconstruction_error": dataclass_to_json_ready(
            reconstruction_error(
                result.fir_final,
                exported_wav,
                sample_rate=result.sample_rate,
                fft_size=result.fft_size,
            )
        ),
        "float_safety": dataclass_to_json_ready(float_safety(exported_wav)),
        "listening_band_target_error": {
            "20_10000_hz": dataclass_to_json_ready(listening_band_target_error_10k),
            "20_18000_hz": dataclass_to_json_ready(listening_band_target_error_18k),
        },
        "perceptual_weighted_error": dataclass_to_json_ready(
            perceptual_weighted_error_from_aligned_error(
                aligned_freqs_hz,
                aligned_error_db,
                sample_rate=result.sample_rate,
            )
        ),
        "audible_target_verdict": audible_target_verdict(
            listening_band_target_error_10k,
            listening_band_target_error_18k,
        ),
    }
    if result.spec.target_projection is not None:
        metadata["target_projection"] = dataclass_to_json_ready(result.spec.target_projection)
    if elapsed_ms is not None:
        metadata["generation_time_ms"] = ensure_non_negative_float("elapsed_ms", elapsed_ms)
    if target_validation is not None:
        metadata["target_validation"] = dataclass_to_json_ready(target_validation)
    if cross_rate_consistency is not None:
        metadata["cross_rate_consistency"] = cross_rate_consistency
    if include_stress_probes:
        metadata["program_material_stress_summary"] = dataclass_to_json_ready(
            true_peak_stress_summary(
                result.fir_final,
                requested_headroom_db=result.spec.requested_headroom_db,
                oversample_factor=result.oversample_factor,
            )
        )
    return metadata


def write_pipeline_report(
    base_out: Path,
    result: Any,
    export_json: bool,
    plot: bool,
    elapsed_ms: float | None = None,
    mode: str | None = None,
    target_validation: Any | None = None,
    cross_rate_consistency: dict[str, object] | None = None,
    include_stress_probes: bool = False,
    reproducible: bool = False,
) -> None:
    export_json = ensure_bool("export_json", export_json)
    plot = ensure_bool("plot", plot)
    include_stress_probes = ensure_bool("include_stress_probes", include_stress_probes)
    reproducible = ensure_bool("reproducible", reproducible)
    if export_json:
        base_out.with_suffix(".json").write_text(
            metadata_json_dumps(
                pipeline_metadata(
                    result,
                    elapsed_ms=None if reproducible else elapsed_ms,
                    mode=mode,
                    target_validation=target_validation,
                    cross_rate_consistency=cross_rate_consistency,
                    include_stress_probes=include_stress_probes,
                ),
                reproducible=reproducible,
            ),
            encoding="utf-8",
        )
    if plot:
        from .visualize import plot_frequency_response, plot_impulse

        plot_frequency_response(
            result.fir_final,
            fs=result.sample_rate,
            target_magnitude=result.target_magnitude,
            save_path=base_out.with_name(f"{base_out.stem}_response.png"),
        )
        plot_impulse(result.fir_final, save_path=base_out.with_name(f"{base_out.stem}_impulse.png"))
