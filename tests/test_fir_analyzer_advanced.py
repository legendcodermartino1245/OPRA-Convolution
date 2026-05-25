import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fir_dsp.analyze import analyze_fir_quality
from fir_dsp.analysis_ext import analyze_ext


def test_analyzer_exposes_advanced_metrics() -> None:
    fir = np.array([0.8, 0.15, 0.04, 0.01, 0.0], dtype=np.float64)
    summary = analyze_fir_quality(fir, sample_rate=48000)
    assert summary.pre_ringing_ms >= 0.0
    assert 0.0 <= summary.low_band_energy_pct <= 100.0
    assert 0.0 <= summary.mid_band_energy_pct <= 100.0
    assert 0.0 <= summary.high_band_energy_pct <= 100.0
    assert summary.impulse_kurtosis > 0.0
    assert summary.effective_length["energy_99_ms"] >= 0.0
    assert "dc_gain_db" in summary.gain_summary
    assert summary.float_safety["has_nan"] is False


def test_extended_analyzer_rejects_invalid_sample_rates() -> None:
    fir = np.ones(64, dtype=np.float64)

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        analyze_ext(fir, True)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        analyze_ext(fir, -48_000)


def test_benchmark_cli_writes_json(tmp_path: Path) -> None:
    fir_path = tmp_path / 'fir.txt'
    json_path = tmp_path / 'benchmark.json'
    np.savetxt(fir_path, np.array([0.8, 0.15, 0.04, 0.01, 0.0], dtype=np.float64))

    cmd = [
        sys.executable,
        '-m',
        'fir_dsp.benchmark_analyze',
        '--fir',
        str(fir_path),
        '--sample-rate',
        '48000',
        '--warmup',
        '1',
        '--repeat',
        '2',
        '--json-out',
        str(json_path),
    ]
    env = dict(**__import__('os').environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    payload = json.loads(completed.stdout)
    assert payload['measured_runs'] == 2
    assert payload['mean_ms'] >= 0.0
    disk_payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert disk_payload['sample_rate'] == 48000
