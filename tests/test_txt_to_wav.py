from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from fir_dsp.txt_to_wav import txt_to_wav, write_wav_float32


def _write_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_txt_to_wav_raw_export_has_no_preamp_path(tmp_path: Path) -> None:
    input_path = tmp_path / "ir.txt"
    output_path = tmp_path / "ir.wav"
    _write_txt(input_path, "# Preamp: -6.0 dB\n1.0\n0.5\n")

    result = txt_to_wav(input_path, output_path, sample_rate=48000)

    assert result["normalized"] is False
    assert result["peak_before"] == pytest.approx(1.0)
    assert result["peak_written"] == pytest.approx(1.0)
    assert result["sample_format"] == "float32"
    assert result["artifact_contract"]["wav_subtype"] == "FLOAT"
    assert result["artifact_contract"]["wav_bits_per_sample"] == 32
    assert result["artifact_contract"]["channels"] == 1
    assert result["artifact_contract"]["frames"] == 2
    assert result["export_parity"]["wav_txt_allclose"] is True
    assert result["exported_wav_reconstruction_error"]["max_abs_error_db"] < 1e-4
    assert result["float_safety"]["has_nan"] is False
    assert result["float_safety"]["has_inf"] is False
    sample_rate, samples = wavfile.read(str(output_path))
    assert sample_rate == 48000
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) == pytest.approx(1.0)


def test_txt_to_wav_normalize_requires_opt_in(tmp_path: Path) -> None:
    input_path = tmp_path / "ir.txt"
    output_path = tmp_path / "ir.wav"
    _write_txt(input_path, "1.0\n0.5\n")

    result = txt_to_wav(input_path, output_path, normalize=True)
    assert result["normalized"] is True
    assert result["peak_written"] == pytest.approx(1.0)
    _, samples = wavfile.read(str(output_path))
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) == pytest.approx(1.0)


def test_txt_to_wav_preserves_values_above_unity_without_normalization(tmp_path: Path) -> None:
    input_path = tmp_path / "ir.txt"
    output_path = tmp_path / "ir.wav"
    _write_txt(input_path, "1.1\n0.5\n")

    result = txt_to_wav(input_path, output_path)
    assert result["normalized"] is False
    assert result["peak_before"] == pytest.approx(1.1)
    assert result["peak_written"] == pytest.approx(1.1)
    assert result["export_parity"]["wav_txt_max_abs_diff"] < 1e-6
    _, samples = wavfile.read(str(output_path))
    assert samples.dtype == np.float32
    assert np.max(np.abs(samples)) == pytest.approx(1.1, rel=1e-6)


def test_txt_to_wav_rejects_invalid_public_contract_values(tmp_path: Path) -> None:
    input_path = tmp_path / "ir.txt"
    output_path = tmp_path / "ir.wav"
    _write_txt(input_path, "0.1\n0.2\n")

    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        txt_to_wav(input_path, output_path, sample_rate=True)

    with pytest.raises(TypeError, match="normalize must be a bool"):
        txt_to_wav(input_path, output_path, sample_rate=48_000, normalize="false")


def test_txt_to_wav_rejects_non_finite_samples_at_load_time(tmp_path: Path) -> None:
    input_path = tmp_path / "ir.txt"
    output_path = tmp_path / "ir.wav"

    for text in ("nan\n1.0\n", "inf\n1.0\n"):
        _write_txt(input_path, text)
        with pytest.raises(ValueError, match="must be finite"):
            txt_to_wav(input_path, output_path, sample_rate=48_000)


def test_direct_wav_writer_rejects_non_finite_samples(tmp_path: Path) -> None:
    output_path = tmp_path / "bad.wav"

    with pytest.raises(ValueError, match="signal must contain only finite values"):
        write_wav_float32(output_path, np.array([np.nan, 1.0]), sample_rate=48_000)


def test_direct_wav_writer_rejects_float32_overflow(tmp_path: Path) -> None:
    output_path = tmp_path / "huge.wav"

    with pytest.raises(ValueError, match="finite float32 WAV range"):
        write_wav_float32(output_path, np.array([1e300, 1.0]), sample_rate=48_000)
