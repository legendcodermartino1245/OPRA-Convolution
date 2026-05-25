import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fir_dsp.core as core_module
from fir_dsp.models import PipelineSpec
from fir_dsp.profiles import PipelineProfile
from fir_dsp.types import DbMagnitude, coerce_db_magnitude, coerce_linear_magnitude

from fir_dsp.api import generate_fir_from_spec, generate_fir_multi_rate, generate_fir_pipeline, is_power_of_two
from fir_dsp.core import (
    WINDOW_PRESETS,
    build_fft_freq_grid,
    db_to_linear,
    interpolate_db_response,
    interpolate_log_frequency_response,
    interpolate_magnitude_response,
    linear_to_db,
    resolve_window,
)


WINDOW_TEST_PROFILE = PipelineProfile(name="test_window")


def test_is_power_of_two():
    assert is_power_of_two(1024)
    assert not is_power_of_two(1000)
    assert not is_power_of_two(0)


def test_fft_grid_length():
    grid = build_fft_freq_grid(1024, 48000)
    assert len(grid) == 513
    assert np.isclose(grid[0], 0.0)
    assert np.isclose(grid[-1], 24000.0)


def test_fft_grid_is_cached_and_read_only():
    core_module._cached_fft_freq_grid.cache_clear()

    grid_a = build_fft_freq_grid(1024, 48000)
    grid_b = build_fft_freq_grid(1024, 48000)

    assert grid_a is grid_b
    assert not grid_a.flags.writeable
    with pytest.raises(ValueError):
        grid_a[1] = -1.0
    assert np.array_equal(grid_b, np.fft.rfftfreq(1024, d=1.0 / 48000.0))
    assert core_module._cached_fft_freq_grid.cache_info().hits == 1


def test_fft_grid_cache_distinguishes_fft_size_for_same_sample_rate():
    core_module._cached_fft_freq_grid.cache_clear()

    grid_1024 = build_fft_freq_grid(1024, 48000)
    grid_2048 = build_fft_freq_grid(2048, 48000)

    assert grid_1024 is not grid_2048
    assert len(grid_1024) == 513
    assert len(grid_2048) == 1025
    assert np.array_equal(grid_1024, np.fft.rfftfreq(1024, d=1.0 / 48000.0))
    assert np.array_equal(grid_2048, np.fft.rfftfreq(2048, d=1.0 / 48000.0))
    assert core_module._cached_fft_freq_grid.cache_info().misses == 2

    assert build_fft_freq_grid(1024, 48000) is grid_1024
    assert core_module._cached_fft_freq_grid.cache_info().hits == 1


def test_db_linear_roundtrip():
    db = np.array([-6.0, 0.0, 3.0], dtype=np.float64)
    lin = db_to_linear(db)
    db_back = linear_to_db(lin)
    assert np.allclose(db, db_back, atol=1e-6)


def test_pipeline_runs_with_fft_binned_magnitude():
    fir = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
    )
    assert fir is not None
    assert len(fir) == 1024


def test_pipeline_runs_with_linear_freq_response():
    freqs = np.array([0.0, 100.0, 1000.0, 10000.0, 24000.0], dtype=np.float64)
    mags = np.array([1.0, 1.0, 0.8, 0.9, 1.0], dtype=np.float64)
    fir = generate_fir_pipeline(
        magnitude=mags,
        freqs_hz=freqs,
        fft_size=1024,
        sample_rate=48000,
        headroom_db=6.0,
        input_scale="linear",
    )
    assert fir is not None
    assert len(fir) == 1024
    assert np.all(np.isfinite(fir))


def test_pipeline_runs_with_db_freq_response():
    freqs = np.array([0.0, 100.0, 1000.0, 10000.0, 24000.0], dtype=np.float64)
    gains_db = np.array([0.0, 0.0, -2.0, -1.0, 0.0], dtype=np.float64)
    fir = generate_fir_pipeline(
        magnitude=gains_db,
        freqs_hz=freqs,
        fft_size=1024,
        sample_rate=48000,
        headroom_db=6.0,
        input_scale="db",
    )
    assert fir is not None
    assert len(fir) == 1024
    assert np.all(np.isfinite(fir))


def test_pipeline_applies_preamp_once_in_linear_domain():
    base = np.ones(513, dtype=np.float64)
    result_no_preamp = generate_fir_pipeline(
        magnitude=base,
        fft_size=1024,
        headroom_db=6.0,
        return_details=True,
    )
    result_with_preamp = generate_fir_pipeline(
        magnitude=base,
        fft_size=1024,
        headroom_db=6.0,
        preamp_db=-6.0,
        preamp_source="peq",
        return_details=True,
    )
    assert np.max(result_with_preamp.target_magnitude) == pytest.approx(1.0)
    assert np.max(result_no_preamp.target_magnitude) == pytest.approx(1.0)
    assert np.array_equal(result_with_preamp.fir_final, result_no_preamp.fir_final)


def test_return_details_exposes_intermediate_stages():
    shaped_target = np.linspace(0.5, 1.5, 513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=shaped_target,
        fft_size=1024,
        headroom_db=6.0,
        window_preset="safe",
        profile=WINDOW_TEST_PROFILE,
        return_details=True,
    )
    assert len(result.fir_linear) == 1024
    assert len(result.fir_windowed) == 1024
    assert len(result.fir_final) == 1024
    assert result.window.preset == "safe"
    assert not np.array_equal(result.fir_linear, result.fir_windowed)
    assert np.all(np.isfinite(result.fir_final))
    assert result.latency.abs_centroid_ms >= 0.0
    assert result.latency.peak_latency_ms >= 0.0
    assert result.latency.energy_centroid_ms >= 0.0


def test_default_profile_rejects_windowing():
    with pytest.raises(ValueError, match="Profile 'default' forbids windowing"):
        generate_fir_pipeline(
            magnitude=np.ones(513, dtype=np.float64),
            fft_size=1024,
            headroom_db=6.0,
            window_preset="safe",
            return_details=True,
        )


def test_default_profile_rejects_spec_windowing():
    spec = PipelineSpec.from_profile(
        sample_rate=48000,
        fft_size=1024,
        input_scale="linear",
        requested_headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        window=resolve_window(preset="safe"),
        gain_stage_preamp_source="none",
        preamp_applied_as_gain_stage=False,
        profile="default",
    )
    with pytest.raises(ValueError, match="Profile 'default' forbids windowing"):
        generate_fir_from_spec(spec, np.ones(513, dtype=np.float64), return_details=True)


def test_pipeline_without_window_preserves_linear_stage():
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        return_details=True,
    )
    assert np.allclose(result.fir_linear, result.fir_windowed)


def test_empty_magnitude_rejected():
    with pytest.raises(ValueError):
        generate_fir_pipeline([], 1024, 6.0)


def test_invalid_fft_rejected():
    with pytest.raises(ValueError):
        generate_fir_pipeline(np.ones(513), 1000, 6.0)


def test_invalid_headroom_type_rejected():
    with pytest.raises(TypeError):
        generate_fir_pipeline(np.ones(513), 1024, "6")


def test_boolean_numeric_parameters_are_rejected():
    with pytest.raises(TypeError, match="headroom_db must be numeric, not bool"):
        generate_fir_pipeline(np.ones(513), 1024, True)

    with pytest.raises(TypeError, match="fft_size must be an integer, not bool"):
        generate_fir_pipeline(np.ones(2), True, 6.0)

    with pytest.raises(TypeError, match="beta must be numeric, not bool"):
        generate_fir_pipeline(
            np.ones(513),
            1024,
            6.0,
            true_peak=True,
            oversample_factor=8,
            window_type="kaiser",
            window_beta=True,
            return_details=True,
        )


def test_magnitude_wrappers_reject_boolean_arrays():
    with pytest.raises(TypeError, match="linear magnitude must be numeric, not bool"):
        coerce_linear_magnitude([True, False])

    with pytest.raises(TypeError, match="dB magnitude must be numeric, not bool"):
        coerce_db_magnitude([True, False])

    with pytest.raises(TypeError, match="oversample_factor must be an integer, not bool"):
        generate_fir_pipeline(
            np.ones(513),
            1024,
            6.0,
            true_peak=True,
            oversample_factor=True,
            return_details=True,
        )


def test_true_peak_must_be_bool():
    with pytest.raises(TypeError, match="true_peak must be a bool"):
        generate_fir_pipeline(
            np.ones(513),
            1024,
            6.0,
            true_peak="false",
            return_details=True,
        )

    with pytest.raises(TypeError, match="true_peak must be a bool"):
        generate_fir_multi_rate(
            np.ones(513),
            rates=[48_000],
            fft_size=1024,
            headroom_db=6.0,
            true_peak="false",
            return_details=True,
        )

    with pytest.raises(TypeError, match="true_peak must be a bool"):
        PipelineSpec.from_profile(
            sample_rate=48000,
            fft_size=1024,
            input_scale="linear",
            requested_headroom_db=6.0,
            true_peak="false",
            oversample_factor=8,
            design_oversample=1,
            window=resolve_window(preset="safe"),
            gain_stage_preamp_source="none",
            preamp_applied_as_gain_stage=False,
            profile="default",
        )


def test_public_bool_contracts_reject_string_flags():
    with pytest.raises(TypeError, match="return_details must be a bool"):
        generate_fir_pipeline(
            np.ones(513),
            1024,
            6.0,
            return_details="false",
        )

    with pytest.raises(TypeError, match="preamp_already_applied must be a bool"):
        generate_fir_pipeline(
            np.ones(513),
            1024,
            6.0,
            preamp_already_applied="false",
        )

    with pytest.raises(TypeError, match="return_details must be a bool"):
        generate_fir_multi_rate(
            np.ones(513),
            rates=[48_000],
            fft_size=1024,
            headroom_db=6.0,
            return_details="false",
        )


def test_pipeline_spec_rejects_bool_source_preamp():
    with pytest.raises(TypeError, match="preamp_db must be numeric, not bool"):
        PipelineSpec.from_profile(
            sample_rate=48000,
            fft_size=1024,
            input_scale="linear",
            requested_headroom_db=6.0,
            true_peak=True,
            oversample_factor=8,
            design_oversample=1,
            window=resolve_window(preset="safe"),
            gain_stage_preamp_source="none",
            preamp_applied_as_gain_stage=False,
            source_preamp_db=True,
            profile="default",
        )


def test_custom_profile_rejects_bool_policy_values():
    with pytest.raises(TypeError, match="true_peak_required must be a bool"):
        PipelineProfile(name="bad", true_peak_required="false")

    with pytest.raises(TypeError, match="minimum_true_peak_margin_db must be numeric, not bool"):
        PipelineProfile(name="bad", minimum_true_peak_margin_db=True)


def test_fft_size_one_rejected_with_clear_error():
    with pytest.raises(ValueError, match="fft_size must be an even integer >= 2"):
        generate_fir_pipeline(
            np.ones(1, dtype=np.float64),
            1,
            6.0,
            true_peak=True,
            oversample_factor=8,
            return_details=True,
        )

    with pytest.raises(ValueError, match="fft_size must be an even integer >= 2"):
        PipelineSpec.from_profile(
            sample_rate=48000,
            fft_size=1,
            input_scale="linear",
            requested_headroom_db=6.0,
            true_peak=True,
            oversample_factor=8,
            design_oversample=1,
            window=resolve_window(preset="safe"),
            gain_stage_preamp_source="none",
            preamp_applied_as_gain_stage=False,
            profile="default",
        )


def test_invalid_magnitude_length_rejected():
    with pytest.raises(ValueError):
        generate_fir_pipeline(np.ones(1024), 1024, 6.0)


def test_negative_magnitude_rejected():
    mag = np.ones(513, dtype=np.float64)
    mag[10] = -1.0
    with pytest.raises(ValueError):
        generate_fir_pipeline(mag, 1024, 6.0)


def test_interpolation_output_shape_linear():
    freqs = np.array([0.0, 100.0, 1000.0, 10000.0, 24000.0], dtype=np.float64)
    mags = np.array([1.0, 1.0, 0.8, 0.9, 1.0], dtype=np.float64)
    interp = interpolate_magnitude_response(freqs_hz=freqs, magnitude=mags, fft_size=1024, sample_rate=48000)
    assert len(interp) == 513
    assert np.all(np.isfinite(interp))
    assert np.all(interp >= 0)


def test_interpolation_output_shape_db():
    freqs = np.array([0.0, 100.0, 1000.0, 10000.0, 24000.0], dtype=np.float64)
    gains_db = np.array([0.0, 0.0, -2.0, -1.0, 0.0], dtype=np.float64)
    interp = interpolate_db_response(freqs_hz=freqs, db_values=gains_db, fft_size=1024, sample_rate=48000)
    assert len(interp) == 513
    assert np.all(np.isfinite(interp))
    assert np.all(interp >= 0)


def test_freqs_must_be_strictly_increasing():
    freqs = np.array([0.0, 1000.0, 1000.0, 24000.0], dtype=np.float64)
    mags = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        interpolate_magnitude_response(freqs, mags, 1024, 48000)


def test_response_must_reach_nyquist():
    freqs = np.array([0.0, 100.0, 1000.0, 10000.0], dtype=np.float64)
    mags = np.array([1.0, 1.0, 0.8, 0.9], dtype=np.float64)
    with pytest.raises(ValueError):
        interpolate_magnitude_response(freqs, mags, 1024, 48000)


def test_fft_binned_input_rejects_db_scale():
    with pytest.raises(ValueError):
        generate_fir_pipeline(
            magnitude=np.ones(513, dtype=np.float64),
            fft_size=1024,
            headroom_db=6.0,
            input_scale="db",
        )


def test_window_preset_resolution():
    spec = resolve_window(preset="sharp")
    assert spec.name == WINDOW_PRESETS["sharp"].name
    assert spec.beta == WINDOW_PRESETS["sharp"].beta


def test_log_frequency_interpolator_preserves_endpoints():
    source_freqs = np.array([0.0, 100.0, 1000.0, 10000.0, 24000.0], dtype=np.float64)
    source_values = np.array([1.2, 1.0, 0.8, 0.9, 1.1], dtype=np.float64)
    target_freqs = np.array([0.0, 50.0, 500.0, 5000.0, 24000.0], dtype=np.float64)
    interpolated = interpolate_log_frequency_response(source_freqs, source_values, target_freqs)
    assert np.isclose(interpolated[0], source_values[0])
    assert np.isclose(interpolated[-1], source_values[-1])


def test_cli_exports_json(tmp_path: Path):
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
        "--out",
        str(out_path),
        "--export-json",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, env=env)
    metadata = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["sample_rate"] == 48000
    assert "latency" in metadata
    assert "error" in metadata


def test_load_response_preserves_single_row_two_column_frequency_tables(tmp_path: Path):
    from fir_dsp.cli import load_response

    response_path = tmp_path / "response.txt"
    response_path.write_text("100.0 0.5\n", encoding="utf-8")

    freqs_hz, values = load_response(response_path)

    assert freqs_hz is not None
    assert np.array_equal(freqs_hz, np.array([100.0], dtype=np.float64))
    assert np.array_equal(values, np.array([0.5], dtype=np.float64))


def test_load_response_preserves_single_value_fft_binned_inputs(tmp_path: Path):
    from fir_dsp.cli import load_response

    response_path = tmp_path / "response.txt"
    response_path.write_text("0.5\n", encoding="utf-8")

    freqs_hz, values = load_response(response_path)

    assert freqs_hz is None
    assert np.array_equal(values, np.array([0.5], dtype=np.float64))


def test_load_response_rejects_non_finite_public_values(tmp_path: Path):
    from fir_dsp.cli import load_response

    response_path = tmp_path / "response.txt"

    response_path.write_text("nan\n1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response values must contain only finite values"):
        load_response(response_path)

    response_path.write_text("0.0 1.0\ninf 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response frequencies must contain only finite values"):
        load_response(response_path)


def test_cli_parser_accepts_common_aliases_and_all_pcm():
    from fir_dsp.cli import STANDARD_PCM_RATES, build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--input",
        "response.txt",
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--output",
        "out/fir.txt",
        "--all-pcm",
        "--output-dir",
        "out/pack",
    ])
    assert args.response == Path("response.txt")
    assert args.out == Path("out/fir.txt")
    assert args.all_pcm is True
    assert STANDARD_PCM_RATES[0] == 44100
    assert STANDARD_PCM_RATES[-1] == 384000


@pytest.mark.parametrize("window", ["none", "hann", "kaiser", "blackman"])
def test_cli_parser_accepts_supported_window_values(window):
    from fir_dsp.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--input",
        "response.txt",
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--output",
        "out/fir.txt",
        "--window",
        window,
    ])

    assert args.window == window


def test_cli_parser_accepts_target_scale():
    from fir_dsp.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--input",
        "response.txt",
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--output",
        "out/fir.txt",
        "--validate-target",
        "--target",
        "target.txt",
        "--target-scale",
        "linear",
    ])

    assert args.target == Path("target.txt")
    assert args.target_scale == "linear"


def test_separate_validation_target_requires_explicit_target_scale(monkeypatch):
    from argparse import Namespace

    import fir_dsp.cli as cli
    from fir_dsp.cli import _resolve_target_validation_source

    monkeypatch.setattr(
        cli,
        "load_response",
        lambda path: (
            np.array([0.0, 24_000.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        ),
    )
    args = Namespace(
        response=Path("response.txt"),
        target=Path("target.txt"),
        target_scale=None,
    )
    freqs = np.array([0.0, 24_000.0], dtype=np.float64)
    values = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="--target-scale"):
        _resolve_target_validation_source(args, freqs, values, "linear")


def test_validation_only_options_require_validate_target():
    from fir_dsp.cli import main

    base_args = [
        "--input",
        "response.txt",
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--output",
        "out/fir.txt",
    ]

    for validation_option in (
        ["--target-scale", "linear"],
        ["--target-min-freq", "30"],
        ["--target-max-freq", "10000"],
    ):
        with pytest.raises(SystemExit):
            main(base_args + validation_option)


def test_json_only_options_require_export_json():
    from fir_dsp.cli import main

    base_args = [
        "--input",
        "response.txt",
        "--fft-size",
        "1024",
        "--headroom-db",
        "6",
        "--output",
        "out/fir.txt",
    ]

    for json_only_option in (
        ["--reproducible"],
        ["--audible-stress"],
    ):
        with pytest.raises(SystemExit):
            main(base_args + json_only_option)


def test_opra_pack_all_skips_generation_failures(monkeypatch, capsys, tmp_path):
    from argparse import Namespace
    from types import SimpleNamespace

    import fir_dsp.opra_cli as opra_cli

    products = {
        "bad_product": {"id": "bad_product"},
        "good_product": {"id": "good_product"},
    }
    eqs_by_product = {
        "bad_product": [{"id": "bad_eq"}],
        "good_product": [{"id": "good_eq"}],
    }

    monkeypatch.setattr(opra_cli, "load_opra_jsonl", lambda db: (products, eqs_by_product, {}))
    monkeypatch.setattr(opra_cli, "select_eq", lambda product, *_args, **_kwargs: eqs_by_product[product["id"]][0])

    calls: list[str] = []

    def fake_generate_fir_multi_rate(*, magnitude, **_kwargs):
        eq_id = magnitude["eq"]["id"]
        calls.append(eq_id)
        if eq_id == "bad_eq":
            raise ValueError("invalid generated target")
        return {48_000: SimpleNamespace(fir_final=np.ones(1, dtype=np.float64))}

    monkeypatch.setattr(opra_cli, "generate_fir_multi_rate", fake_generate_fir_multi_rate)
    monkeypatch.setattr(opra_cli, "_apply_opra_post_scale_to_results", lambda results: results)
    monkeypatch.setattr(opra_cli, "_write_verification_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(opra_cli, "_write_shared_reports", lambda *args, **kwargs: None)
    monkeypatch.setattr(opra_cli, "_write_eq_pack", lambda output_path, *_args, **_kwargs: output_path.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(opra_cli, "_write_opra_notice", lambda *args, **kwargs: None)
    monkeypatch.setattr(opra_cli, "_write_opra_attribution", lambda *args, **kwargs: None)
    monkeypatch.setattr(opra_cli, "eq_to_txt", lambda eq: eq["id"])

    output_dir = tmp_path / "opra_pack_all_skip_test"
    exit_code = opra_cli.cmd_eq_pack_all(
        Namespace(
            db="unused.jsonl",
            rates=[48_000],
            output=output_dir,
            fft_size=1024,
            headroom=6.0,
            target_sample_rate=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == ["bad_eq", "good_eq"]
    assert "[SKIP] bad_product (FIR generation failed: invalid generated target)" in captured.out
    assert (output_dir / "good_product" / "selected_eq.txt").read_text(encoding="utf-8") == "good_eq"


def test_cli_rejects_removed_peq_source():
    from fir_dsp.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--peq",
            "profile.txt",
            "--fft-size",
            "1024",
            "--headroom-db",
            "6",
            "--output",
            "out/fir.txt",
        ])


from fir_dsp.preamp import apply_preamp_db


def test_pipeline_rejects_negative_linear_values_before_preamp() -> None:
    with pytest.raises(ValueError, match="linear-domain magnitude"):
        apply_preamp_db(np.array([1.0, -0.1], dtype=np.float64), -6.0)


def test_preamp_applied_only_once_regression() -> None:
    base = np.ones(513, dtype=np.float64)

    once = generate_fir_pipeline(
        magnitude=base,
        fft_size=1024,
        headroom_db=6.0,
        preamp_db=-6.0,
        preamp_source="peq",
        return_details=True,
    )

    assert np.max(once.target_magnitude) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Preamp would be applied twice"):
        generate_fir_pipeline(
            magnitude=apply_preamp_db(base, -6.0),
            fft_size=1024,
            headroom_db=6.0,
            preamp_db=-6.0,
            preamp_source="peq",
            preamp_already_applied=True,
            return_details=True,
        )


def test_preamp_already_applied_does_not_claim_gain_stage_application() -> None:
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        preamp_already_applied=True,
        return_details=True,
    )

    assert result.spec.preamp_applied_as_gain_stage is False
    assert result.verification.preamp_applied_as_gain_stage is False
    assert result.gain_traceability.preamp_applied_as_gain_stage is False


def test_pipeline_spec_executes_as_single_source_of_truth():
    spec = PipelineSpec.from_profile(
        sample_rate=48000,
        fft_size=1024,
        input_scale="linear",
        requested_headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        design_oversample=1,
        window=None,
        gain_stage_preamp_source="none",
        preamp_applied_as_gain_stage=False,
        profile="default",
        interpolation_mode="log",
    )
    result = generate_fir_from_spec(spec, np.ones(513, dtype=np.float64), return_details=True)
    assert result.spec == spec
    assert result.verification.true_peak_margin_db >= 1.0 - 1e-6


def test_pipeline_spec_rejects_profile_objects_before_fingerprinting():
    class ProfileLike:
        name = "custom_reference"

    with pytest.raises(TypeError, match="profile must be a registered profile name string"):
        PipelineSpec.from_profile(
            sample_rate=48000,
            fft_size=1024,
            input_scale="linear",
            requested_headroom_db=6.0,
            true_peak=True,
            oversample_factor=8,
            design_oversample=1,
            window=resolve_window(preset="safe"),
            gain_stage_preamp_source="none",
            preamp_applied_as_gain_stage=False,
            profile=ProfileLike(),
        )


def test_spec_rejects_profile_contract_drift():
    spec = PipelineSpec.from_profile(
        sample_rate=48000,
        fft_size=1024,
        input_scale="linear",
        requested_headroom_db=6.0,
        true_peak=False,
        oversample_factor=8,
        design_oversample=1,
        window=resolve_window(preset="safe"),
        gain_stage_preamp_source="none",
        preamp_applied_as_gain_stage=False,
        profile="default",
        interpolation_mode="log",
    )
    with pytest.raises(ValueError, match="requires true_peak=True"):
        generate_fir_from_spec(spec, np.ones(513, dtype=np.float64), return_details=True)


def test_linear_pipeline_rejects_db_magnitude_wrapper():
    with pytest.raises(TypeError, match="Expected linear-domain magnitude"):
        generate_fir_pipeline(
            magnitude=DbMagnitude(np.zeros(513, dtype=np.float64)),
            fft_size=1024,
            headroom_db=6.0,
            input_scale="linear",
        )


def test_cli_doctor_reports_program_probe_headroom(tmp_path: Path):
    cmd = [sys.executable, "-m", "fir_dsp.cli", "--doctor"]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    assert "program_probe_headroom: PASS" in completed.stdout
    assert "Worst simulated program-material TP:" in completed.stdout
    assert "Probe noise:" in completed.stdout


def test_design_oversample_above_one_is_rejected():
    freqs = np.array([20.0, 100.0, 1000.0, 5000.0, 12000.0, 24000.0], dtype=np.float64)
    gains_db = np.array([-3.0, -1.0, 0.0, 1.5, -0.5, -2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="design_oversample"):
        generate_fir_pipeline(
            magnitude=gains_db,
            freqs_hz=freqs,
            fft_size=1024,
            sample_rate=48000,
            headroom_db=6.0,
            input_scale="db",
            design_oversample=4,
            return_details=True,
        )
