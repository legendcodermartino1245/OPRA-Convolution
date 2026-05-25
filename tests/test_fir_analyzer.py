import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fir_dsp.analyze import analyze_fir_quality, load_fir
from fir_dsp.analysis_ext import db, energy_decay, energy_window, evaluate, sweep_stress_peak
from fir_dsp.api import generate_fir_pipeline


def test_analyzer_flags_minimum_phase_as_low_preringing():
    mag = np.linspace(0.85, 1.15, 513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=6.0,
        return_details=True,
    )
    summary = analyze_fir_quality(result.fir_final, sample_rate=48000, oversample_factor=8)
    assert summary.phase_character == "minimum_like"
    assert summary.preringing_risk in {"very_low", "low", "minimal"}
    assert summary.quality_verdict in {"excellent_for_minimum_phase", "strong"}


def test_analyzer_cli_writes_json(tmp_path: Path):
    fir_path = tmp_path / "fir.txt"
    json_path = tmp_path / "analysis.json"
    np.savetxt(fir_path, np.array([0.0, 0.2, 0.8, 0.1], dtype=np.float64))

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.analyze",
        "--fir",
        str(fir_path),
        "--sample-rate",
        "48000",
        "--json-out",
        str(json_path),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    assert "Quality verdict:" in completed.stdout

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sample_rate"] == 48000
    assert "quality_verdict" in payload
    assert "effective_length" in payload
    assert "gain_summary" in payload
    assert "float_safety" in payload


def test_analyzer_rejects_bool_policy_inputs():
    fir = np.ones(8, dtype=np.float64)

    with pytest.raises(TypeError, match="fir must be numeric, not bool"):
        analyze_fir_quality(np.array([True, False]), 48_000, oversample_factor=1)

    with pytest.raises(TypeError, match="true_peak_enforced must be a bool"):
        analyze_fir_quality(fir, 48_000, oversample_factor=1, true_peak_enforced="false")

    with pytest.raises(TypeError, match="true_peak_target_dbfs must be numeric, not bool"):
        analyze_fir_quality(fir, 48_000, oversample_factor=1, true_peak_target_dbfs=True)

    with pytest.raises(TypeError, match="minimum_true_peak_margin_db must be numeric, not bool"):
        analyze_fir_quality(fir, 48_000, oversample_factor=1, minimum_true_peak_margin_db=True)


def test_load_fir_rejects_non_finite_coefficients(tmp_path: Path):
    fir_path = tmp_path / "bad_fir.txt"
    fir_path.write_text("nan\n1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="FIR file coefficients must contain only finite values"):
        load_fir(fir_path)


def test_load_fir_preserves_single_column_single_row_input(tmp_path: Path):
    fir_path = tmp_path / "single_value_fir.txt"
    fir_path.write_text("0.25\n", encoding="utf-8")

    loaded = load_fir(fir_path)

    assert np.array_equal(loaded, np.array([0.25], dtype=np.float64))


def test_load_fir_treats_single_row_two_column_input_as_one_coefficient_column(tmp_path: Path):
    fir_path = tmp_path / "single_row_two_col_fir.txt"
    fir_path.write_text("0.25 999.0\n", encoding="utf-8")

    loaded = load_fir(fir_path)

    assert np.array_equal(loaded, np.array([0.25], dtype=np.float64))


def test_analyzer_rejects_values_that_would_emit_non_finite_metrics():
    with pytest.raises(ValueError, match="too large for finite analyzer metrics"):
        analyze_fir_quality(np.ones(64, dtype=np.float64) * 1e300, 48_000, oversample_factor=1)


def test_extended_analysis_helpers_reject_invalid_public_scalars():
    fir = np.ones(8, dtype=np.float64)

    with pytest.raises(ValueError, match="x must be finite"):
        db(float("nan"))

    with pytest.raises(ValueError, match="value must be finite"):
        evaluate("bad", float("nan"), 1.0, 2.0)

    with pytest.raises(TypeError, match="lower_is_better must be a bool"):
        evaluate("bad", 0.0, 1.0, 2.0, lower_is_better="false")

    with pytest.raises(TypeError, match="window_ms must be numeric, not bool"):
        energy_window(fir, 48_000, True)

    with pytest.raises(TypeError, match="duration_s must be numeric, not bool"):
        sweep_stress_peak(fir, 48_000, duration_s=True)

    with pytest.raises(ValueError, match="duration_s must be finite"):
        sweep_stress_peak(fir, 48_000, duration_s=float("nan"))

    with pytest.raises(ValueError, match="h must contain only finite values"):
        energy_decay(np.array([np.nan, 1.0], dtype=np.float64), 48_000)


def test_sweep_stress_peak_handles_fir_longer_than_probe():
    fir = np.zeros(1024, dtype=np.float64)
    fir[0] = 1.0

    result = sweep_stress_peak(fir, 48_000, duration_s=0.001, oversample_factor=1)

    assert result["value"] > -100.0


def test_analyzer_cli_writes_extended_json_block(tmp_path: Path):
    fir_path = tmp_path / "fir.txt"
    json_path = tmp_path / "analysis_extended.json"
    mag = np.linspace(0.95, 1.05, 513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=mag,
        fft_size=1024,
        headroom_db=6.0,
        return_details=True,
    )
    np.savetxt(fir_path, result.fir_final)

    cmd = [
        sys.executable,
        "-m",
        "fir_dsp.analyze",
        "--fir",
        str(fir_path),
        "--sample-rate",
        "48000",
        "--extended",
        "--json-out",
        str(json_path),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["extended_validation"]["mode"] == "fast"
    assert payload["extended_validation"]["results"]
    assert any(
        item["metric"] == "sweep_stress_peak_dbfs"
        for item in payload["extended_validation"]["results"]
    )
