import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from fir_dsp.artifact_metrics import (
    _aligned_error_db,
    artifact_contract,
    cross_rate_consistency,
    effective_length,
    export_parity,
    gain_summary,
    perceptual_weighted_error,
    perceptual_weighted_error_from_aligned_error,
    reconstruction_error,
    target_error_band,
    target_error_band_from_aligned_error,
    true_peak_stress_summary,
)
import fir_dsp.api as api_module
from fir_dsp.api import generate_fir_multi_rate, generate_fir_pipeline, is_power_of_two
from fir_dsp.profiles import PipelineProfile
from fir_dsp.core_validation import frequency_response_error, run_core_validations
from fir_dsp.validation import ensure_1d_finite_array, ensure_bool, ensure_numeric_scalar, ensure_positive_sample_rate
from fir_dsp.core import (
    apply_tap_scaling,
    build_fft_freq_grid,
    coeff_hash_sha256,
    compute_frequency_response,
    compute_true_peak,
    dataclass_to_json_ready,
    db_to_linear,
    design_fir_from_mag_fft,
    interpolate_log_frequency_response,
    linear_to_db,
    minimum_phase_from_mag,
    normalize_fir,
    request_fingerprint,
    summarize_response_error,
    summarize_latency,
    summarize_verification,
    verify_headroom,
)
from fir_dsp.cli import save_fir
import fir_dsp.report as report_module
from fir_dsp.report import metadata_json_dumps, pipeline_metadata, write_pipeline_report

RELAXED_TEST_PROFILE = PipelineProfile(name="test_relaxed")


def test_pipeline_is_deterministic():
    mag = np.linspace(0.8, 1.2, 513, dtype=np.float64)
    result_a = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=3.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )
    result_b = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=3.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )
    assert np.array_equal(result_a.fir_final, result_b.fir_final)
    assert result_a.verification.coeff_hash_sha256 == result_b.verification.coeff_hash_sha256
    assert result_a.verification.request_fingerprint_sha256 == result_b.verification.request_fingerprint_sha256


def test_request_fingerprint_tracks_interpolation_mode():
    freqs = np.array([0.0, 60.0, 800.0, 5000.0, 24000.0], dtype=np.float64)
    values = np.array([1.0, 0.55, 1.4, 0.8, 1.1], dtype=np.float64)

    log_result = generate_fir_pipeline(
        magnitude=values,
        freqs_hz=freqs,
        fft_size=1024,
        headroom_db=6.0,
        input_scale="linear",
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        interpolation_mode="log",
        return_details=True,
    )
    linear_result = generate_fir_pipeline(
        magnitude=values,
        freqs_hz=freqs,
        fft_size=1024,
        headroom_db=6.0,
        input_scale="linear",
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        interpolation_mode="linear",
        return_details=True,
    )

    assert log_result.spec.interpolation_mode == "log"
    assert linear_result.spec.interpolation_mode == "linear"
    assert log_result.verification.request_fingerprint_sha256 != linear_result.verification.request_fingerprint_sha256


def test_multi_rate_output_contract_hashes_and_margins():
    expected = {
        44_100: {
            "coeff": "19240addadd6463b43eae6a2113099a1dc163bf5bd092978137c85ebbee6b5d5",
            "request": "39a6a41b60d076fb21d733603f2359119f7ab275a7c84068e3492995a64bfbc7",
        },
        48_000: {
            "coeff": "19240addadd6463b43eae6a2113099a1dc163bf5bd092978137c85ebbee6b5d5",
            "request": "712d5de06544874af2655be45f4903ea77dd8708eafb671f3a5bf6dbb945d2b2",
        },
        96_000: {
            "coeff": "19240addadd6463b43eae6a2113099a1dc163bf5bd092978137c85ebbee6b5d5",
            "request": "345aceabddafa0284dd3c2e8d56fd4871974ae7cee8c37ce658fece4ef809dd7",
        },
    }
    results_a = generate_fir_multi_rate(
        magnitude=np.ones(513, dtype=np.float64),
        rates=list(expected),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        return_details=True,
    )
    results_b = generate_fir_multi_rate(
        magnitude=np.ones(513, dtype=np.float64),
        rates=list(reversed(expected)),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        return_details=True,
    )

    assert list(results_a) == sorted(expected)
    assert list(results_b) == sorted(expected)
    for sample_rate, contract in expected.items():
        result_a = results_a[sample_rate]
        result_b = results_b[sample_rate]
        assert np.array_equal(result_a.fir_final, result_b.fir_final)
        assert result_a.verification.coeff_hash_sha256 == contract["coeff"]
        assert result_a.verification.request_fingerprint_sha256 == contract["request"]
        assert result_b.verification.coeff_hash_sha256 == contract["coeff"]
        assert result_b.verification.request_fingerprint_sha256 == contract["request"]
        assert result_a.verification.true_peak_margin_db >= 1.0 - 1e-12
        assert result_a.error.max_abs_error_db <= 1e-12


def test_saved_fir_text_round_trips_to_reported_coeff_hash(tmp_path: Path):
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        return_details=True,
    )
    out_path = tmp_path / "fir.txt"

    save_fir(out_path, result.fir_final)
    loaded = np.loadtxt(out_path, dtype=np.float64)

    assert np.array_equal(loaded, result.fir_final)
    assert coeff_hash_sha256(loaded) == result.verification.coeff_hash_sha256


def test_save_fir_rejects_invalid_signal_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="fir must contain only finite values"):
        save_fir(tmp_path / "bad_nan.txt", np.array([np.nan], dtype=np.float64))

    with pytest.raises(TypeError, match="fir must be numeric, not bool"):
        save_fir(tmp_path / "bad_bool.txt", [True, 0.5])

    with pytest.raises(ValueError, match="fir must be a 1D array"):
        save_fir(tmp_path / "bad_2d.txt", np.ones((2, 2), dtype=np.float64))


def test_response_error_summary_remains_finite_for_huge_finite_values():
    summary = summarize_response_error(
        np.ones(513, dtype=np.float64) * 1e300,
        np.ones(513, dtype=np.float64) * 0.5,
    )

    assert np.isfinite(summary.aligned_rms_error)
    assert np.isfinite(summary.aligned_max_abs_error)
    assert np.isfinite(summary.max_abs_error_db)


def test_hash_helpers_reject_non_finite_values():
    with pytest.raises(ValueError, match="fir must contain only finite values"):
        coeff_hash_sha256(np.array([np.nan], dtype=np.float64))

    with pytest.raises(ValueError, match="payload.x contains non-finite numeric value"):
        request_fingerprint({"x": float("nan")})

    with pytest.raises(ValueError, match=r"payload.x\[0\] contains non-finite numeric value"):
        request_fingerprint({"x": np.array([np.nan], dtype=np.float64)})

    with pytest.raises(ValueError, match="payload key contains non-finite numeric value"):
        request_fingerprint({float("nan"): "bad"})

    with pytest.raises(ValueError, match="duplicate key after string conversion"):
        request_fingerprint({1: "numeric", "1": "string"})


def test_json_fingerprint_helpers_normalize_numpy_scalar_inputs():
    assert request_fingerprint({"x": np.int64(1), "y": np.array([2.0, 3.0])}) == request_fingerprint(
        {"x": 1, "y": [2.0, 3.0]}
    )
    assert metadata_json_dumps({"x": np.int64(1), "y": np.array([2.0, 3.0])}) == '{\n  "x": 1,\n  "y": [\n    2.0,\n    3.0\n  ]\n}'


def test_json_ready_helper_rejects_non_finite_public_values():
    with pytest.raises(ValueError, match="JSON-ready payload contains non-finite numeric value"):
        dataclass_to_json_ready(float("nan"))

    with pytest.raises(ValueError, match="JSON-ready payload contains non-finite numeric value"):
        dataclass_to_json_ready(np.array([1.0, np.inf], dtype=np.float64))

    with pytest.raises(ValueError, match="JSON-ready payload contains non-finite numeric value"):
        metadata_json_dumps({np.float64("inf"): "bad"})

    with pytest.raises(ValueError, match="duplicate key after string conversion"):
        metadata_json_dumps({1: "numeric", "1": "string"})


def test_shared_numeric_validators_accept_numpy_scalar_numbers():
    assert ensure_numeric_scalar("x", np.int64(2)) == 2.0
    assert ensure_numeric_scalar("x", np.float64(1.5)) == 1.5
    assert ensure_positive_sample_rate(np.int64(48_000)) == 48_000

    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=np.int64(1024),
        headroom_db=np.float64(6.0),
        sample_rate=np.int64(48_000),
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )
    assert result.sample_rate == 48_000


def test_shared_bool_validator_accepts_numpy_bool_scalars():
    assert ensure_bool("flag", np.bool_(True)) is True
    assert ensure_bool("flag", np.bool_(False)) is False

    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=np.bool_(False),
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )
    assert result.true_peak is False

    scaled = normalize_fir(np.ones(4, dtype=np.float64), 6.0, true_peak=np.bool_(False))
    assert np.max(np.abs(scaled)) == pytest.approx(10 ** (-6.0 / 20.0))


def test_public_power_of_two_helper_handles_bool_and_numpy_ints():
    assert is_power_of_two(True) is False
    assert is_power_of_two(np.bool_(True)) is False
    assert is_power_of_two(np.int64(1024)) is True
    assert is_power_of_two(np.int64(1000)) is False


def test_shared_array_validator_rejects_mixed_boolean_values():
    with pytest.raises(TypeError, match="x must be numeric, not bool"):
        ensure_1d_finite_array("x", [True, 1.0])

    with pytest.raises(TypeError, match="x must be numeric, not bool"):
        ensure_1d_finite_array("x", np.array([False, 1.0], dtype=object))

    with pytest.raises(TypeError, match="fir must be numeric, not bool"):
        coeff_hash_sha256([True, 0.5])

    with pytest.raises(TypeError, match="signal must be numeric, not bool"):
        artifact_contract([False, 1.0], 48_000)


def test_public_core_helpers_reject_non_finite_outputs():
    with pytest.raises(ValueError, match="non-finite linear magnitudes"):
        db_to_linear(np.array([10_000.0], dtype=np.float64))

    with pytest.raises(TypeError, match="eps must be numeric, not bool"):
        linear_to_db(np.ones(2, dtype=np.float64), eps=True)

    with pytest.raises(ValueError, match="eps must be > 0"):
        linear_to_db(np.array([0.0], dtype=np.float64), eps=0.0)

    with pytest.raises(ValueError, match="frequency response contains non-finite values"):
        compute_frequency_response(np.array([1e308, 1e308], dtype=np.float64))

    with pytest.raises(TypeError, match="true_peak must be a bool"):
        normalize_fir(np.ones(4, dtype=np.float64), 6.0, true_peak="false")

    with pytest.raises(TypeError, match="true_peak must be a bool"):
        verify_headroom(np.ones(4, dtype=np.float64) * 0.5, 6.0, true_peak="false")

    with pytest.raises(TypeError, match="minimum_phase must be a bool"):
        design_fir_from_mag_fft(np.ones(513, dtype=np.float64), 1024, minimum_phase="false")

    with pytest.raises(ValueError, match="magnitude produces non-finite FIR values"):
        design_fir_from_mag_fft(np.ones(513, dtype=np.float64) * 1e308, 1024, minimum_phase=False)


def test_latency_summary_remains_finite_for_huge_finite_values():
    summary = summarize_latency(np.array([1e308, 1e308], dtype=np.float64), 48_000)

    assert np.isfinite(summary.abs_centroid_ms)
    assert np.isfinite(summary.energy_centroid_ms)
    assert np.isfinite(summary.peak_latency_ms)


def test_summarize_verification_rejects_bool_policy_scalars():
    with pytest.raises(TypeError, match="true_peak_min_safe_margin_db must be numeric, not bool"):
        summarize_verification(
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            1,
            {},
            true_peak_target_dbfs=-6.0,
            true_peak_min_safe_margin_db=True,
        )


def test_export_parity_rms_remains_finite_for_huge_finite_values():
    summary = export_parity(
        np.ones(10, dtype=np.float64) * 1e300,
        np.zeros(10, dtype=np.float64),
    )

    assert summary.wav_txt_max_abs_diff == pytest.approx(1e300)
    assert np.isfinite(summary.wav_txt_rms_diff)


def test_artifact_metric_sample_rates_reject_bool():
    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        artifact_contract(np.ones(10, dtype=np.float64), True)

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        effective_length(np.ones(10, dtype=np.float64), True)

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        gain_summary(np.ones(10, dtype=np.float64), True)

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        target_error_band(
            np.ones(513, dtype=np.float64),
            np.ones(513, dtype=np.float64),
            sample_rate=True,
            fft_size=1024,
            min_freq_hz=20.0,
            max_freq_hz=20_000.0,
            label="bad",
        )

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        perceptual_weighted_error(
            np.ones(513, dtype=np.float64),
            np.ones(513, dtype=np.float64),
            sample_rate=True,
            fft_size=1024,
        )


def test_aligned_error_helpers_match_public_artifact_metrics():
    target = np.linspace(0.6, 1.4, 513, dtype=np.float64)
    actual = target * (1.0 + 0.01 * np.sin(np.linspace(0.0, 8.0, 513, dtype=np.float64)))
    freqs_hz, error_db = _aligned_error_db(target, actual, sample_rate=48_000, fft_size=1024)

    assert target_error_band_from_aligned_error(
        freqs_hz,
        error_db,
        min_freq_hz=20.0,
        max_freq_hz=10_000.0,
        label="20_10000_hz",
    ) == target_error_band(
        target,
        actual,
        sample_rate=48_000,
        fft_size=1024,
        min_freq_hz=20.0,
        max_freq_hz=10_000.0,
        label="20_10000_hz",
    )
    assert perceptual_weighted_error_from_aligned_error(
        freqs_hz,
        error_db,
        sample_rate=48_000,
    ) == perceptual_weighted_error(
        target,
        actual,
        sample_rate=48_000,
        fft_size=1024,
    )


def test_core_validation_rejects_non_finite_and_bool_contract_values():
    with pytest.raises(ValueError, match="h must contain only finite values"):
        run_core_validations([np.nan, 1.0])

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        run_core_validations([1.0, 0.0], sample_rate=True)

    with pytest.raises(TypeError, match="fft_size must be an integer, not bool"):
        frequency_response_error([1.0, 0.0], [1.0], True)

    with pytest.raises(ValueError, match="fft_size must be >= FIR length"):
        frequency_response_error(np.ones(8, dtype=np.float64), np.ones(3, dtype=np.float64), 4)

    with pytest.raises(ValueError, match="frequency response contains non-finite values"):
        frequency_response_error(np.ones(1024, dtype=np.float64) * 1e308, np.ones(513), 1024)


def test_core_validation_scaled_metrics_do_not_emit_non_finite_values():
    results = run_core_validations(
        np.ones(1024, dtype=np.float64) * 1e300,
        sample_rate=48_000,
        target_mag=np.ones(513, dtype=np.float64) * 1e300,
        fft_size=1024,
    )

    values = [result["value"] for result in results if result["value"] is not None]
    assert values
    assert all(np.isfinite(value) for value in values)


def test_artifact_metrics_reject_non_finite_public_arrays():
    with pytest.raises(ValueError, match="signal must contain only finite values"):
        gain_summary(np.array([np.nan, 1.0], dtype=np.float64), 48_000)

    with pytest.raises(ValueError, match="reference must contain only finite values"):
        reconstruction_error(np.array([np.inf, 1.0], dtype=np.float64), np.ones(2), 48_000)

    with pytest.raises(ValueError, match="target_magnitude must contain only finite values"):
        target_error_band(
            np.array([np.nan], dtype=np.float64),
            np.ones(1, dtype=np.float64),
            sample_rate=48_000,
            fft_size=1,
            min_freq_hz=0.0,
            max_freq_hz=1.0,
            label="bad",
        )

    with pytest.raises(ValueError, match="gain_summary frequency response contains non-finite values"):
        gain_summary(np.array([1e308, 1e308], dtype=np.float64), 48_000)

    with pytest.raises(ValueError, match="reconstruction_error frequency response contains non-finite values"):
        reconstruction_error(np.array([1e308, 1e308], dtype=np.float64), np.ones(2), 48_000)

    with pytest.raises(TypeError, match="min_freq_hz must be numeric, not bool"):
        target_error_band(
            np.ones(513, dtype=np.float64),
            np.ones(513, dtype=np.float64),
            sample_rate=48_000,
            fft_size=1024,
            min_freq_hz=True,
            max_freq_hz=1_000.0,
            label="bad",
        )

    with pytest.raises(ValueError, match="min_freq_hz must be finite"):
        target_error_band(
            np.ones(513, dtype=np.float64),
            np.ones(513, dtype=np.float64),
            sample_rate=48_000,
            fft_size=1024,
            min_freq_hz=float("nan"),
            max_freq_hz=1_000.0,
            label="bad",
        )

    with pytest.raises(ValueError, match="min_freq_hz must be <= max_freq_hz"):
        target_error_band(
            np.ones(513, dtype=np.float64),
            np.ones(513, dtype=np.float64),
            sample_rate=48_000,
            fft_size=1024,
            min_freq_hz=2_000.0,
            max_freq_hz=1_000.0,
            label="bad",
        )


def test_effective_length_remains_finite_for_huge_finite_values():
    summary = effective_length(np.ones(10, dtype=np.float64) * 1e300, 48_000)

    assert np.isfinite(summary.energy_50_ms)
    assert np.isfinite(summary.energy_90_ms)
    assert np.isfinite(summary.energy_99_ms)
    assert np.isfinite(summary.energy_999_ms)


def test_true_peak_stress_summary_rejects_bool_contract_values():
    with pytest.raises(TypeError, match="requested_headroom_db must be numeric, not bool"):
        true_peak_stress_summary(
            np.ones(4, dtype=np.float64),
            requested_headroom_db=True,
            oversample_factor=1,
            probe_length=16,
        )

    with pytest.raises(TypeError, match="oversample_factor must be an integer, not bool"):
        true_peak_stress_summary(
            np.ones(4, dtype=np.float64),
            requested_headroom_db=6.0,
            oversample_factor=True,
            probe_length=16,
        )


def test_true_peak_stress_summary_handles_fir_longer_than_probe():
    fir = np.zeros(1024, dtype=np.float64)
    fir[0] = 1.0

    summary = true_peak_stress_summary(
        fir,
        requested_headroom_db=6.0,
        oversample_factor=1,
        probe_length=64,
    )

    assert summary.worst_true_peak_dbfs > -100.0
    assert all(value > -100.0 for value in summary.probe_true_peak_dbfs.values())


def test_apply_tap_scaling_rejects_invalid_public_scalars():
    with pytest.raises(ValueError, match="headroom_db must be >= 0"):
        apply_tap_scaling(np.ones(10, dtype=np.float64), -6.0, true_peak=False)

    with pytest.raises(TypeError, match="true_peak must be a bool"):
        apply_tap_scaling(np.ones(10, dtype=np.float64), 6.0, true_peak="false")


def test_true_peak_headroom_is_enforced():
    mag = np.ones(513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=1.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )
    expected_peak = 10 ** (-1.0 / 20.0)
    assert compute_true_peak(result.fir_final, oversample_factor=8) <= expected_peak + 1e-3


def test_minimum_phase_rejects_unresolvable_relative_notches():
    magnitude = np.ones(513, dtype=np.float64)
    magnitude[128] = 1e-13

    with pytest.raises(ValueError, match="relative magnitudes"):
        minimum_phase_from_mag(magnitude, 1024)


def test_minimum_phase_is_absolute_level_invariant_for_tiny_valid_targets():
    tiny = np.ones(513, dtype=np.float64) * 1e-40
    full = np.ones(513, dtype=np.float64)

    tiny_fir = minimum_phase_from_mag(tiny, 1024)
    full_fir = minimum_phase_from_mag(full, 1024)

    assert np.all(np.isfinite(tiny_fir))
    assert np.allclose(tiny_fir / np.max(np.abs(tiny_fir)), full_fir / np.max(np.abs(full_fir)))


def test_sample_peak_mode_does_not_enforce_true_peak_contract():
    mag = np.ones(513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=6.0,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    assert result.verification.true_peak_policy == "measure_only"
    assert result.verification.true_peak_target_is_baked_safety_ceiling is False
    assert result.verification.true_peak_margin_warning is False


def test_sample_peak_mode_does_not_fail_system_validation_on_true_peak_overshoot():
    mag = np.ones(513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=0.0,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    assert result.system_validation.status != "FAIL"


def test_pipeline_raises_on_system_validation_failure(monkeypatch):
    import fir_dsp.api as api
    from fir_dsp.system_validation import SystemValidationResult

    def fail_validation(*args, **kwargs):
        return SystemValidationResult(
            status="FAIL",
            violations=["forced failure"],
            warnings=[],
        )

    monkeypatch.setattr(api, "validate_system", fail_validation)

    with pytest.raises(RuntimeError, match="System validation failed"):
        generate_fir_pipeline(
            magnitude=np.ones(513, dtype=np.float64),
            fft_size=1024,
            headroom_db=6.0,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
        )


def test_multi_rate_returns_sorted_unique_keys():
    mag = np.ones(513, dtype=np.float64)
    results = generate_fir_multi_rate(
        magnitude=mag,
        rates=[96000, 48000, 48000, 44100],
        fft_size=1024,
        headroom_db=6.0,
    )
    assert list(results.keys()) == [44100, 48000, 96000]


def test_multi_rate_generation_dispatches_independent_rates_concurrently(monkeypatch):
    barrier = threading.Barrier(2, timeout=1.0)
    lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_generate_fir_pipeline(*_args, **kwargs):
        nonlocal active_calls, max_active_calls
        sample_rate = int(kwargs["sample_rate"])
        with lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        finally:
            with lock:
                active_calls -= 1
        return np.array([float(sample_rate)], dtype=np.float64)

    monkeypatch.setattr(api_module, "generate_fir_pipeline", fake_generate_fir_pipeline)

    results = generate_fir_multi_rate(
        magnitude=np.ones(3, dtype=np.float64),
        rates=[48_000, 44_100],
        fft_size=8192,
        headroom_db=6.0,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
    )

    assert max_active_calls >= 2
    assert list(results) == [44_100, 48_000]
    assert np.array_equal(results[44_100], np.array([44_100.0], dtype=np.float64))
    assert np.array_equal(results[48_000], np.array([48_000.0], dtype=np.float64))


def test_multi_rate_default_matches_single_rate_true_peak_policy():
    mag = np.ones(513, dtype=np.float64)

    single = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=6.0,
        return_details=True,
    )
    multi = generate_fir_multi_rate(
        magnitude=mag,
        rates=[48_000],
        fft_size=1024,
        headroom_db=6.0,
        return_details=True,
    )[48_000]

    assert multi.true_peak is True
    assert multi.verification.true_peak_policy == single.verification.true_peak_policy
    assert np.array_equal(multi.fir_final, single.fir_final)


def test_single_rate_eq_json_multi_rate_matches_pipeline_path():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -4.0,
                    "bands": [
                        {"type": "peak_dip", "frequency": 12_000.0, "gain_db": 5.0, "q": 1.8},
                        {"type": "high_shelf", "frequency": 7_500.0, "gain_db": -3.0, "q": 0.7},
                    ],
                },
            },
        },
    }

    single = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=2048,
        headroom_db=6.0,
        sample_rate=48_000,
        true_peak=False,
        window_preset=None,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )
    multi = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[48_000],
        fft_size=2048,
        headroom_db=6.0,
        true_peak=False,
        window_preset=None,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )[48_000]

    assert np.array_equal(multi.target_magnitude, single.target_magnitude)
    assert np.array_equal(multi.fir_final, single.fir_final)
    assert multi.verification.target_hash_sha256 == single.verification.target_hash_sha256


def test_run_core_validations_uses_provided_sample_rate_for_energy_front_ratio():
    fir = np.zeros(384, dtype=np.float64)
    fir[:192] = 1.0

    results_48k = {result["metric"]: result for result in run_core_validations(fir, sample_rate=48_000)}
    results_96k = {result["metric"]: result for result in run_core_validations(fir, sample_rate=96_000)}

    assert results_48k["energy_front_ratio"]["value"] == 0.5
    assert results_48k["energy_front_ratio"]["status"] == "FAIL"
    assert results_96k["energy_front_ratio"]["value"] == 1.0
    assert results_96k["energy_front_ratio"]["status"] == "PASS"


def test_core_validation_checks_aligned_frequency_response_when_target_is_provided():
    target = np.ones(513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=target,
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    validation = {
        item["metric"]: item
        for item in run_core_validations(
            result.fir_final,
            sample_rate=48_000,
            target_mag=target,
            fft_size=1024,
        )
    }

    assert validation["freq_response_error_db"]["status"] == "PASS"
    assert validation["freq_response_error_db"]["value"] is not None


def test_default_true_peak_exact_target_is_system_validation_pass():
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    assert result.verification.true_peak_margin_warning is False
    assert result.system_validation.status == "PASS"
    assert "Reconstruction classified as MARGINAL" not in result.system_validation.warnings


def test_silent_target_is_rejected_before_normalization():
    with pytest.raises(ValueError, match="silent"):
        generate_fir_pipeline(
            magnitude=np.zeros(513, dtype=np.float64),
            fft_size=1024,
            headroom_db=6.0,
            true_peak=False,
            profile=RELAXED_TEST_PROFILE,
            return_details=True,
        )


def test_nonzero_low_level_target_is_normalized_like_same_shape():
    reference = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=False,
        window_preset=None,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    for target_peak in (1e-8, 1e-20):
        result = generate_fir_pipeline(
            magnitude=np.ones(513, dtype=np.float64) * target_peak,
            fft_size=1024,
            headroom_db=6.0,
            true_peak=False,
            window_preset=None,
            profile=RELAXED_TEST_PROFILE,
            return_details=True,
        )

        assert np.allclose(result.fir_final, reference.fir_final, atol=1e-12)
        assert result.error.max_abs_error_db <= 1e-9


def test_target_fit_reflects_input_without_window_distortion():
    freqs = np.array([0.0, 20.0, 100.0, 1000.0, 5000.0, 10000.0, 20000.0, 24000.0], dtype=np.float64)
    gains_db = np.array([0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0], dtype=np.float64)

    result = generate_fir_pipeline(
        magnitude=gains_db,
        freqs_hz=freqs,
        fft_size=1024,
        sample_rate=48000,
        headroom_db=0.0,
        input_scale="db",
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    assert result.error.max_abs_error_db <= 1e-9


def test_energy_front_ratio_failure_does_not_reject_otherwise_valid_target():
    freqs = np.array([0.0, 20.0, 80.0, 300.0, 1000.0, 4000.0, 12000.0, 24000.0], dtype=np.float64)
    gains_db = np.array(
        [
            8.945289546683746,
            0.2519874038115866,
            -20.39824245202498,
            -10.286192358422518,
            -15.459673034892118,
            -25.060608980237962,
            -20.053721966280015,
            -27.565307368726437,
        ],
        dtype=np.float64,
    )

    result = generate_fir_pipeline(
        magnitude=gains_db,
        freqs_hz=freqs,
        fft_size=1024,
        sample_rate=48000,
        headroom_db=6.0,
        input_scale="db",
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    core_validation = {
        item["metric"]: item
        for item in run_core_validations(
            result.fir_final,
            mode="eq",
            sample_rate=48_000,
            target_mag=result.target_magnitude,
            fft_size=1024,
        )
    }
    assert core_validation["freq_response_error_db"]["status"] == "PASS"
    assert core_validation["energy_front_ratio"]["status"] == "FAIL"
    assert result.system_validation.status == "WARN"
    assert any("energy_front_ratio warning" in warning for warning in result.system_validation.warnings)


def test_core_validation_fails_non_silent_response_for_silent_target():
    validation = {
        item["metric"]: item
        for item in run_core_validations(
            np.ones(1024, dtype=np.float64),
            sample_rate=48_000,
            target_mag=np.zeros(513, dtype=np.float64),
            fft_size=1024,
        )
    }

    assert validation["freq_response_error_db"]["status"] == "FAIL"
    assert validation["freq_response_error_db"]["reason"] == "silent target produced non-silent response"


def test_minimum_phase_pipeline_rejects_eq_targets_with_zero_bins():
    eq = {
        "type": "eq",
        "data": {
            "type": "parametric_eq",
            "parameters": {
                "gain_db": -8.6,
                "bands": [
                    {"type": "low_shelf", "frequency": 70.0, "gain_db": 7.0, "q": 0.7},
                    {"type": "peak_dip", "frequency": 230.0, "gain_db": -2.0, "q": 0.75},
                    {"type": "high_shelf", "frequency": 2100.0, "gain_db": 10.0, "q": 0.7},
                    {"type": "peak_dip", "frequency": 1550.0, "gain_db": -2.8, "q": 1.5},
                    {"type": "peak_dip", "frequency": 4500.0, "gain_db": -4.7, "q": 2.0},
                    {"type": "low_pass", "frequency": 11000.0, "slope": 12.0},
                ],
            },
        },
    }

    with pytest.raises(ValueError, match="minimum-phase design requires strictly positive magnitudes"):
        generate_fir_pipeline(
            magnitude={"type": "eq_json", "eq": eq},
            fft_size=131072,
            headroom_db=9.6,
            sample_rate=48000,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
        )


def test_release_verdict_requires_export_safety():
    from fir_dsp.cli import _build_release_verdict
    from fir_dsp.opra_cli import _build_release_verdict as _build_opra_release_verdict

    class Verification:
        true_peak_margin_warning = False

    class Error:
        max_abs_error_db = 999.0

    class Result:
        verification = Verification()
        error = Error()

    cross_rate_payload = {"strict_all_rates_pass": True}

    for build_verdict in (_build_release_verdict, _build_opra_release_verdict):
        verdict = build_verdict({48_000: Result()}, cross_rate_payload)
        assert verdict["export_safe"] is False
        assert verdict["playback_safe"] is True
        assert verdict["cross_rate_consistent"] is True
        assert verdict["recommended_use"] == "review_before_release"


def test_opra_release_verdict_treats_cross_rate_as_advisory():
    from fir_dsp.opra_cli import _build_release_verdict as _build_opra_release_verdict

    class Verification:
        true_peak_margin_warning = False

    class Error:
        max_abs_error_db = 0.0

    class Result:
        verification = Verification()
        error = Error()

    verdict = _build_opra_release_verdict({48_000: Result()}, {"strict_all_rates_pass": False})

    assert verdict["export_safe"] is True
    assert verdict["playback_safe"] is True
    assert verdict["cross_rate_consistent"] is False
    assert verdict["recommended_use"] == "production_release"


def test_opra_post_scale_updates_final_headroom_and_fingerprint():
    from fir_dsp.cli import _apply_opra_post_scale

    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    adjusted, attenuation_db = _apply_opra_post_scale(result, desired_margin_db=2.0)

    assert attenuation_db > 0.0
    assert adjusted.spec.requested_headroom_db == result.spec.requested_headroom_db
    assert adjusted.spec.post_scale_attenuation_db == pytest.approx(attenuation_db)
    assert adjusted.spec.normalization_headroom_db == pytest.approx(result.spec.normalization_headroom_db + attenuation_db)
    assert adjusted.verification.final_baked_headroom_db == pytest.approx(adjusted.spec.normalization_headroom_db)
    assert adjusted.gain_traceability.final_baked_headroom_db == pytest.approx(adjusted.spec.normalization_headroom_db)
    assert adjusted.verification.coeff_hash_sha256 != result.verification.coeff_hash_sha256
    assert adjusted.verification.request_fingerprint_sha256 != result.verification.request_fingerprint_sha256
    assert adjusted.system_validation.status == "PASS"
    assert adjusted.system_validation.warnings == []


def test_opra_cli_uses_shared_post_scale_helper(monkeypatch):
    import fir_dsp.opra_cli as opra_cli

    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        profile="default",
        return_details=True,
    )
    calls: list[float] = []

    def fake_post_scale(result_arg, desired_margin_db):
        calls.append(desired_margin_db)
        return result_arg, 0.0

    monkeypatch.setattr(opra_cli, "_apply_opra_post_scale", fake_post_scale)

    adjusted = opra_cli._apply_opra_post_scale_to_results({48_000: result})

    assert adjusted[48_000] is result
    assert calls[0] == pytest.approx(1.0)


def test_cli_exports_verification_block(tmp_path: Path):
    response_path = tmp_path / "response.txt"
    out_path = tmp_path / "fir.txt"
    np.savetxt(response_path, np.ones(513, dtype=np.float64))

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.cli",
        "--response",
        str(response_path),
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--true-peak",
        "--oversample-factor",
        "8",
        "--out",
        str(out_path),
        "--export-json",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, env=env)

    metadata = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    exported_fir = np.loadtxt(out_path, dtype=np.float64)
    assert "verification" in metadata
    assert "request_fingerprint_sha256" in metadata["verification"]
    assert coeff_hash_sha256(exported_fir) == metadata["verification"]["coeff_hash_sha256"]
    assert metadata["verification"]["fir_true_peak_linear"] > 0.0
    assert metadata["verification"]["gain_stage_preamp_source"] == "none"
    assert metadata["verification"]["preamp_applied_as_gain_stage"] is False
    assert metadata["verification"]["true_peak_policy"] == "normalize_to_oversampled_peak_estimate"
    assert metadata["verification"]["true_peak_target_is_baked_safety_ceiling"] is True
    assert metadata["true_peak_measurement"]["method"] == "polyphase_oversampled_peak_estimate"
    assert metadata["true_peak_measurement"]["exact"] is False
    assert metadata["interpolation_mode"] == "log"
    assert metadata["target_shape_normalization"] == "unit_peak_before_design"
    assert metadata["runtime"]["float_dtype"] == "float64"
    assert metadata["spec"]["gain_stage_preamp_source"] == "none"
    assert metadata["spec"]["preamp_applied_as_gain_stage"] is False
    assert metadata["artifact_contract"]["wav_subtype"] == "FLOAT"
    assert metadata["artifact_contract"]["wav_bits_per_sample"] == 32
    assert metadata["artifact_contract"]["channels"] == 1
    assert metadata["artifact_contract"]["frames"] == 1024
    assert metadata["export_parity"]["wav_txt_allclose"] is True
    assert metadata["effective_length"]["energy_99_ms"] >= 0.0
    assert "dc_gain_db" in metadata["gain_summary"]
    assert metadata["exported_wav_reconstruction_error"]["max_abs_error_db"] >= 0.0
    assert metadata["float_safety"]["has_nan"] is False
    assert "20_10000_hz" in metadata["listening_band_target_error"]
    assert "20_18000_hz" in metadata["listening_band_target_error"]
    assert metadata["perceptual_weighted_error"]["weighted_mean_abs_error_db"] >= 0.0
    assert "program_material_stress_summary" not in metadata
    assert metadata["audible_target_verdict"] in {"transparent", "very_close", "acceptable", "review"}
    assert metadata["minimum_phase"] is True


def test_pipeline_metadata_reuses_one_aligned_error_curve(monkeypatch):
    result = generate_fir_pipeline(
        magnitude=np.linspace(0.7, 1.3, 513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )
    call_count = 0
    real_aligned_error_db = report_module._aligned_error_db

    def counted_aligned_error_db(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_aligned_error_db(*args, **kwargs)

    monkeypatch.setattr(report_module, "_aligned_error_db", counted_aligned_error_db)

    metadata = pipeline_metadata(result)

    assert call_count == 1
    assert metadata["listening_band_target_error"]["20_10000_hz"] == dataclass_to_json_ready(
        target_error_band(
            result.target_magnitude,
            result.actual_magnitude,
            sample_rate=result.sample_rate,
            fft_size=result.fft_size,
            min_freq_hz=20.0,
            max_freq_hz=10_000.0,
            label="20_10000_hz",
        )
    )
    assert metadata["perceptual_weighted_error"] == dataclass_to_json_ready(
        perceptual_weighted_error(
            result.target_magnitude,
            result.actual_magnitude,
            sample_rate=result.sample_rate,
            fft_size=result.fft_size,
        )
    )


def test_cli_reproducible_json_is_byte_stable(tmp_path: Path):
    response_path = tmp_path / "response.txt"
    np.savetxt(response_path, np.ones(513, dtype=np.float64))
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    json_outputs: list[str] = []
    for index in (1, 2):
        out_dir = tmp_path / f"run_{index}"
        out_dir.mkdir()
        out_path = out_dir / "fir.txt"
        cmd = [
            sys.executable,
            "-m",
            "fir_dsp.cli",
            "--response",
            str(response_path),
            "--fft-size",
            "1024",
            "--headroom-db",
            "6",
            "--true-peak",
            "--oversample-factor",
            "8",
            "--out",
            str(out_path),
            "--export-json",
            "--reproducible",
        ]
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        json_outputs.append(out_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert json_outputs[0] == json_outputs[1]
    metadata = json.loads(json_outputs[0])
    assert "generation_time_ms" not in metadata
    assert metadata["verification"]["coeff_hash_sha256"] == "19240addadd6463b43eae6a2113099a1dc163bf5bd092978137c85ebbee6b5d5"


def test_pipeline_metadata_rejects_non_finite_elapsed_time():
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    with pytest.raises(ValueError, match="elapsed_ms must be finite"):
        pipeline_metadata(result, elapsed_ms=float("nan"))


def test_pipeline_report_helpers_reject_string_bool_contracts(tmp_path: Path):
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    with pytest.raises(TypeError, match="include_stress_probes must be a bool"):
        pipeline_metadata(result, include_stress_probes="false")

    with pytest.raises(TypeError, match="export_json must be a bool"):
        write_pipeline_report(tmp_path / "fir.txt", result, export_json="false", plot=False)

    with pytest.raises(TypeError, match="plot must be a bool"):
        write_pipeline_report(tmp_path / "fir.txt", result, export_json=False, plot="false")

    with pytest.raises(TypeError, match="reproducible must be a bool"):
        write_pipeline_report(tmp_path / "fir.txt", result, export_json=True, plot=False, reproducible="false")


def test_metadata_json_dumps_rejects_bool_and_invalid_precision_contracts():
    with pytest.raises(TypeError, match="reproducible must be a bool"):
        metadata_json_dumps({"x": 1.2345}, reproducible="false")

    with pytest.raises(TypeError, match="precision must be an integer, not bool"):
        metadata_json_dumps({"x": 1.2345}, reproducible=True, precision=True)

    with pytest.raises(ValueError, match="precision must be a positive integer"):
        metadata_json_dumps({"x": 1.2345}, reproducible=True, precision=0)


def test_cli_exports_program_material_stress_when_enabled(tmp_path: Path):
    response_path = tmp_path / "response.txt"
    out_path = tmp_path / "fir.txt"
    np.savetxt(response_path, np.ones(513, dtype=np.float64))

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.cli",
        "--response",
        str(response_path),
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--true-peak",
        "--oversample-factor",
        "8",
        "--out",
        str(out_path),
        "--export-json",
        "--audible-stress",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, env=env)

    metadata = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["program_material_stress_summary"]["worst_probe"] in {
        "step",
        "alternating",
        "multitone",
        "sweep",
        "noise",
    }


def test_cli_multi_rate_writes_cross_rate_consistency_report(tmp_path: Path):
    response_path = tmp_path / "response.txt"
    out_path = tmp_path / "fir.txt"
    np.savetxt(response_path, np.ones(513, dtype=np.float64))

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.cli",
        "--response",
        str(response_path),
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--true-peak",
        "--oversample-factor",
        "8",
        "--rates",
        "44100",
        "48000",
        "--out",
        str(out_path),
        "--export-json",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, env=env)

    shared_report = json.loads((tmp_path / "fir_cross_rate_consistency.json").read_text(encoding="utf-8"))
    assert shared_report["worst_pair"] == "44100_vs_48000"
    assert shared_report["max_response_delta_between_rates_db"] >= 0.0
    assert shared_report["p95_response_delta_between_rates_db"] >= 0.0
    assert shared_report["rms_response_delta_between_rates_db"] >= 0.0
    assert shared_report["worst_frequency_hz"] >= 20.0
    assert shared_report["comparison_mode"] == "gain_aligned_shape_audible_band"
    assert shared_report["min_freq_hz"] == 20.0
    assert shared_report["max_freq_hz"] == 18000.0
    assert shared_report["reference_rate"] == 44100
    assert shared_report["alignment_min_freq_hz"] == 100.0
    assert shared_report["alignment_max_freq_hz"] == 10000.0
    assert shared_report["strict_max_response_delta_between_rates_db"] >= 0.0
    assert shared_report["extended_max_response_delta_between_rates_db"] >= 0.0
    assert isinstance(shared_report["extended_warning"], bool)
    assert isinstance(shared_report["strict_all_rates_pass"], bool)
    assert shared_report["targets_derived_from_canonical_master"] is False
    assert shared_report["canonical_master_target_hash_sha256"] is None
    assert set(shared_report["cross_rate_gain_alignment_offsets_db"].keys()) == {"48000"}
    assert set(shared_report["rate_target_hashes_derived_from_master"].keys()) == {"44100", "48000"}
    assert isinstance(shared_report["band_summaries"], list)
    assert any(band["label"] == "16_20_khz" for band in shared_report["band_summaries"])
    assert all("p95_delta_db" in band for band in shared_report["band_summaries"])
    assert all("rms_delta_db" in band for band in shared_report["band_summaries"])
    assert all("worst_frequency_hz" in band for band in shared_report["band_summaries"])

    release_verdict = json.loads((tmp_path / "fir_release_verdict.json").read_text(encoding="utf-8"))
    assert release_verdict["export_safe"] is True
    assert release_verdict["playback_safe"] is True
    assert release_verdict["cross_rate_consistent"] is True
    assert release_verdict["recommended_use"] == "production_release"

    per_rate_metadata = json.loads((tmp_path / "fir_44100.json").read_text(encoding="utf-8"))
    assert per_rate_metadata["cross_rate_consistency"] == shared_report


def test_cli_reference_mode_applies_safe_defaults(tmp_path: Path):
    response_path = tmp_path / "response.txt"
    out_path = tmp_path / "fir.txt"
    np.savetxt(response_path, np.ones(513, dtype=np.float64))

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.cli",
        "--response",
        str(response_path),
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--mode",
        "reference",
        "--out",
        str(out_path),
        "--export-json",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, env=env)

    metadata = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "reference"
    assert metadata["true_peak"] is True
    assert metadata["oversample_factor"] == 8
    assert metadata["design_oversample"] == 1
    assert metadata["window_type"] is None
    assert metadata["window_preset"] is None


@pytest.mark.parametrize("disable_window_args", (["--no-window"], ["--window", "none"]))
def test_cli_reference_mode_no_window_keeps_exact_target_math(tmp_path: Path, disable_window_args: list[str]):
    response_path = tmp_path / "response.txt"
    out_path = tmp_path / "fir.txt"
    target = np.ones(513, dtype=np.float64)
    target[180:280] = 10 ** (4.0 / 20.0)
    target[380:] = 10 ** (-6.0 / 20.0)
    np.savetxt(response_path, target)

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.cli",
        "--response",
        str(response_path),
        "--input-scale",
        "linear",
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--mode",
        "reference",
        *disable_window_args,
        "--out",
        str(out_path),
        "--export-json",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, env=env)

    metadata = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "reference"
    assert metadata["window_type"] is None
    assert metadata["window_preset"] is None
    assert metadata["error"]["max_abs_error_db"] < 1e-8


def test_pipeline_spec_and_profile_are_exposed():
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        return_details=True,
        profile="default",
    )
    assert result.profile == "default"
    assert result.spec.profile == "default"
    assert result.spec.true_peak is True


def test_golden_reference_request_and_coeff_hashes():
    mag = np.ones(513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        return_details=True,
    )
    assert result.verification.coeff_hash_sha256 == "19240addadd6463b43eae6a2113099a1dc163bf5bd092978137c85ebbee6b5d5"
    assert result.verification.request_fingerprint_sha256 == "712d5de06544874af2655be45f4903ea77dd8708eafb671f3a5bf6dbb945d2b2"

def test_default_profile_applies_strict_closed_system_playback_margin():
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
        profile="default",
        design_oversample=1,
    )
    assert result.spec.requested_headroom_db == 6.0
    assert result.spec.normalization_headroom_db == 7.0
    assert result.verification.true_peak_target_dbfs == -6.0
    assert result.verification.true_peak_margin_db >= 1.0 - 1e-3
    assert result.verification.true_peak_margin_warning is False


def test_removed_registered_profile_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown profile"):
        generate_fir_pipeline(
            magnitude=np.ones(513, dtype=np.float64),
            fft_size=1024,
            headroom_db=6.0,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
            profile="reference_safe",
        )

    with pytest.raises(ValueError, match="Unknown profile"):
        generate_fir_pipeline(
            magnitude=np.ones(513, dtype=np.float64),
            fft_size=1024,
            headroom_db=6.0,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
            profile="mart_mode",
        )


def test_eq_json_source_preamp_is_reported_without_separate_gain_stage():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -3.6,
                    "bands": [],
                },
            },
        },
    }

    result = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=1024,
        headroom_db=9.6,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    assert result.gain_traceability.source_preamp_db == -3.6
    assert result.gain_traceability.source_preamp_present is True
    assert result.gain_traceability.source_preamp_used_for_target_shape is True
    assert result.gain_traceability.source_preamp_origin == "opra_eq_json"
    assert result.gain_traceability.gain_stage_preamp_source == "none"
    assert result.gain_traceability.preamp_applied_as_gain_stage is False
    assert result.gain_traceability.final_baked_headroom_db == 10.6
    assert result.gain_traceability.final_gain_policy == "normalized_to_oversampled_peak_estimate_target"
    assert result.gain_traceability.true_peak_policy == "normalize_to_oversampled_peak_estimate"
    assert result.gain_traceability.true_peak_target_is_baked_safety_ceiling is True
    assert result.spec.source_preamp_db == -3.6
    assert result.spec.source_preamp_present is True
    assert result.spec.source_preamp_used_for_target_shape is True
    assert result.spec.source_preamp_origin == "opra_eq_json"
    assert result.spec.gain_stage_preamp_source == "none"
    assert result.spec.preamp_applied_as_gain_stage is False
    assert result.verification.gain_stage_preamp_source == "none"
    assert result.verification.preamp_applied_as_gain_stage is False
    assert result.verification.source_preamp_db == -3.6
    assert result.verification.source_preamp_present is True
    assert result.verification.source_preamp_used_for_target_shape is True
    assert result.verification.source_preamp_origin == "opra_eq_json"
    assert result.verification.final_baked_headroom_db == 10.6
    assert result.verification.final_gain_policy == "normalized_to_oversampled_peak_estimate_target"
    assert result.verification.true_peak_policy == "normalize_to_oversampled_peak_estimate"
    assert result.verification.true_peak_target_is_baked_safety_ceiling is True


def test_closed_profile_keeps_eq_target_scale_invariant():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": 0.0,
                    "bands": [
                        {"type": "low_shelf", "frequency": 100.0, "gain_db": 20.0, "q": 0.7},
                    ],
                },
            },
        },
    }

    result = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=1024,
        headroom_db=6.0,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        profile="default",
        return_details=True,
    )

    assert np.max(result.target_magnitude) == pytest.approx(1.0)
    assert np.all(np.isfinite(result.fir_final))


def test_default_profile_allows_eq_json_source_preamp_overshoot_before_final_normalization():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": 0.0,
                    "bands": [
                        {"type": "low_shelf", "frequency": 100.0, "gain_db": 20.0, "q": 0.7},
                    ],
                },
            },
        },
    }

    result = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=1024,
        headroom_db=6.0,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    assert np.all(np.isfinite(result.fir_final))
    assert result.verification.source_preamp_present is True


def test_eq_json_low_pass_zero_nyquist_target_is_rejected():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -8.6,
                    "bands": [
                        {"type": "low_shelf", "frequency": 70.0, "gain_db": 7.0, "q": 0.7},
                        {"type": "peak_dip", "frequency": 230.0, "gain_db": -2.0, "q": 0.75},
                        {"type": "high_shelf", "frequency": 2100.0, "gain_db": 10.0, "q": 0.7},
                        {"type": "peak_dip", "frequency": 1550.0, "gain_db": -2.8, "q": 1.5},
                        {"type": "peak_dip", "frequency": 4500.0, "gain_db": -4.7, "q": 2.0},
                        {"type": "low_pass", "frequency": 11000.0, "slope": 12.0},
                    ],
                },
            },
        },
    }

    with pytest.raises(ValueError, match="minimum-phase design requires strictly positive magnitudes"):
        generate_fir_pipeline(
            magnitude=eq_json,
            fft_size=1024,
            headroom_db=9.6,
            sample_rate=48000,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
        )


def test_eq_json_source_preamp_metadata_rejects_out_of_safe_range():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": 1000.0,
                    "bands": [],
                },
            },
        },
    }

    with pytest.raises(ValueError, match="preamp_db must be within the safe range"):
        generate_fir_pipeline(
            magnitude=eq_json,
            fft_size=1024,
            headroom_db=6.0,
            sample_rate=48000,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
        )


def test_closed_system_profile_normalizes_raw_target_boost_shape():
    reference = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        profile="default",
        return_details=True,
    )
    boosted = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64) * 10.0,
        fft_size=1024,
        headroom_db=6.0,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        profile="default",
        return_details=True,
    )

    assert np.array_equal(boosted.fir_final, reference.fir_final)


def test_closed_system_profile_normalizes_db_target_boost_shape():
    freqs = np.array([0.0, 20.0, 1000.0, 24_000.0], dtype=np.float64)
    gains_db = np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float64)

    reference = generate_fir_pipeline(
        magnitude=np.zeros_like(gains_db),
        freqs_hz=freqs,
        input_scale="db",
        fft_size=1024,
        headroom_db=6.0,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        profile="default",
        return_details=True,
    )
    boosted = generate_fir_pipeline(
        magnitude=gains_db,
        freqs_hz=freqs,
        input_scale="db",
        fft_size=1024,
        headroom_db=6.0,
        sample_rate=48000,
        true_peak=True,
        oversample_factor=8,
        profile="default",
        return_details=True,
    )

    assert np.array_equal(boosted.fir_final, reference.fir_final)


def test_extreme_finite_target_is_scale_normalized_before_design():
    reference = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64) * 1e300,
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    assert np.array_equal(result.fir_final, reference.fir_final)
    assert result.error.max_abs_error_db <= 1e-9


def test_manual_preamp_is_reported_as_separate_gain_stage():
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        preamp_db=-6.0,
        preamp_source="peq",
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    assert result.gain_traceability.source_preamp_db == -6.0
    assert result.gain_traceability.source_preamp_present is True
    assert result.gain_traceability.source_preamp_used_for_target_shape is False
    assert result.gain_traceability.source_preamp_origin == "gain_stage"
    assert result.gain_traceability.gain_stage_preamp_source == "peq"
    assert result.gain_traceability.preamp_applied_as_gain_stage is True
    assert result.gain_traceability.final_baked_headroom_db == 7.0
    assert result.spec.source_preamp_db == -6.0
    assert result.spec.source_preamp_present is True
    assert result.spec.source_preamp_used_for_target_shape is False
    assert result.spec.source_preamp_origin == "gain_stage"
    assert result.spec.gain_stage_preamp_source == "peq"
    assert result.spec.preamp_applied_as_gain_stage is True
    assert result.verification.gain_stage_preamp_source == "peq"
    assert result.verification.preamp_applied_as_gain_stage is True
    assert result.verification.source_preamp_db == -6.0
    assert result.verification.source_preamp_present is True
    assert result.verification.source_preamp_used_for_target_shape is False
    assert result.verification.source_preamp_origin == "gain_stage"


def test_multi_rate_eq_json_uses_unified_master_target():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -6.2,
                    "bands": [
                        {"type": "low_shelf", "frequency": 105, "gain_db": 6.1, "q": 0.71},
                        {"type": "peak_dip", "frequency": 2800, "gain_db": -3.5, "q": 2.0},
                        {"type": "peak_dip", "frequency": 4200, "gain_db": 1.9, "q": 0.70},
                        {"type": "peak_dip", "frequency": 5900, "gain_db": -5.3, "q": 5.0},
                        {"type": "high_shelf", "frequency": 10000, "gain_db": -4.5, "q": 0.71},
                    ],
                },
            },
        },
    }

    results = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[44100, 48000, 96000, 192000],
        fft_size=131072,
        headroom_db=9.6,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        return_details=True,
    )

    direct_44100 = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=131072,
        headroom_db=9.6,
        sample_rate=44100,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        return_details=True,
    )
    assert not np.array_equal(results[44100].target_magnitude, direct_44100.target_magnitude)

    cross_rate_summary = cross_rate_consistency(results)
    assert cross_rate_summary is not None
    assert cross_rate_summary.strict_all_rates_pass is True
    assert cross_rate_summary.strict_max_response_delta_between_rates_db < 0.001


def test_multi_rate_eq_json_target_sample_rate_pins_source_curve():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -4.0,
                    "bands": [
                        {"type": "low_shelf", "frequency": 120, "gain_db": 5.5, "q": 0.71},
                        {"type": "peak_dip", "frequency": 3100, "gain_db": -4.2, "q": 2.4},
                        {"type": "high_shelf", "frequency": 9500, "gain_db": -3.0, "q": 0.71},
                    ],
                },
            },
        },
    }

    results = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[44100, 48000],
        fft_size=4096,
        headroom_db=9.6,
        true_peak=False,
        target_sample_rate=48000,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )
    direct_48000 = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=4096,
        headroom_db=9.6,
        sample_rate=48000,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )
    direct_44100 = generate_fir_pipeline(
        magnitude=eq_json,
        fft_size=4096,
        headroom_db=9.6,
        sample_rate=44100,
        true_peak=False,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    assert np.allclose(results[48000].target_magnitude, direct_48000.target_magnitude, rtol=0.0, atol=1e-12)
    assert not np.array_equal(results[44100].target_magnitude, direct_44100.target_magnitude)
    assert results[44100].spec.target_projection == {
        "measurement_domain": "continuous_source_eq",
        "target_sample_rate": 48000,
        "reference_target_rate": 48000,
        "projection_stage": "pre_design",
        "projection_grid": "canonical_union_fft_bin_centers",
        "interpolation_mode": "log",
        "design_sample_rate": 44100,
    }


def test_target_sample_rate_projection_is_consistent_between_design_rates():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -2.0,
                    "bands": [
                        {"type": "peak_dip", "frequency": 1000, "gain_db": 1.0, "q": 0.7},
                    ],
                },
            },
        },
    }

    results = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[48000, 96000],
        fft_size=4096,
        headroom_db=9.6,
        true_peak=False,
        target_sample_rate=48000,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    freqs_96000 = build_fft_freq_grid(4096, 96000)
    freqs_48000 = build_fft_freq_grid(4096, 48000)
    projected_96000_to_48000 = interpolate_log_frequency_response(
        freqs_96000,
        results[96000].target_magnitude,
        freqs_48000,
    )
    delta_db = np.max(
        np.abs(linear_to_db(projected_96000_to_48000) - linear_to_db(results[48000].target_magnitude))
    )

    assert delta_db < 0.001


def test_target_sample_rate_full_pack_coheres_on_reference_grid():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -2.0,
                    "bands": [
                        {"type": "peak_dip", "frequency": 1000, "gain_db": 1.0, "q": 0.7},
                    ],
                },
            },
        },
    }

    results = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[44100, 48000, 96000],
        fft_size=4096,
        headroom_db=9.6,
        true_peak=False,
        target_sample_rate=48000,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )

    cross_rate_summary = cross_rate_consistency(results)
    assert cross_rate_summary is not None
    assert cross_rate_summary.strict_all_rates_pass is True
    assert cross_rate_summary.strict_max_response_delta_between_rates_db < 0.001


def test_target_sample_rate_changes_request_fingerprint_and_fir():
    eq_json = {
        "type": "eq_json",
        "eq": {
            "type": "eq",
            "data": {
                "type": "parametric_eq",
                "parameters": {
                    "gain_db": -2.0,
                    "bands": [
                        {"type": "peak_dip", "frequency": 18000, "gain_db": 3.0, "q": 3.0},
                        {"type": "high_shelf", "frequency": 12000, "gain_db": -3.0, "q": 0.71},
                    ],
                },
            },
        },
    }

    target_44100 = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[48000],
        fft_size=4096,
        headroom_db=9.6,
        true_peak=False,
        target_sample_rate=44100,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )[48000]
    target_48000 = generate_fir_multi_rate(
        magnitude=eq_json,
        rates=[48000],
        fft_size=4096,
        headroom_db=9.6,
        true_peak=False,
        target_sample_rate=48000,
        profile=RELAXED_TEST_PROFILE,
        return_details=True,
    )[48000]

    assert target_44100.verification.request_fingerprint_sha256 != target_48000.verification.request_fingerprint_sha256
    assert not np.array_equal(target_44100.fir_final, target_48000.fir_final)
