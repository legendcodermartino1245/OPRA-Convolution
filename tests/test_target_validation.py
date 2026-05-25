import numpy as np
import pytest

from fir_dsp.api import generate_fir_pipeline
from fir_dsp.target_validation import validate_fir_against_target


def test_validate_fir_against_target_from_two_column_db_input():
    fft_size = 1024
    sample_rate = 48000
    freqs_hz = np.array([20.0, 100.0, 1000.0, 5000.0, 12000.0, 24000.0], dtype=np.float64)
    values_db = np.array([-3.0, -1.0, 0.0, 1.5, -0.5, -2.0], dtype=np.float64)

    result = generate_fir_pipeline(
        magnitude=values_db,
        freqs_hz=freqs_hz,
        fft_size=fft_size,
        headroom_db=6.0,
        sample_rate=sample_rate,
        input_scale="db",
        return_details=True,
    )

    summary = validate_fir_against_target(
        result.fir_final,
        sample_rate=sample_rate,
        target_freqs_hz=freqs_hz,
        target_values=values_db,
        target_scale="db",
        n_fft=fft_size,
        min_freq_hz=20.0,
    )

    assert summary.bins_compared > 0
    assert summary.max_abs_error_db < 0.25
    assert summary.mean_abs_error_db < 0.10
    assert 20.0 <= summary.max_error_freq_hz <= sample_rate / 2.0
    assert summary.listening_band_summary is not None
    assert summary.listening_band_summary.label == "listening_band"
    assert summary.listening_band_summary.min_freq_hz >= 20.0
    assert summary.listening_band_summary.max_freq_hz <= 10_000.0
    assert len(summary.band_summaries) == 4
    assert summary.band_summaries[0].label == "sub_bass_to_low"
    assert summary.band_summaries[-1].label == "air"
    assert all(band.bins_compared > 0 for band in summary.band_summaries)


def test_validate_fir_against_target_rejects_invalid_frequency_bounds():
    fir = np.ones(1024, dtype=np.float64)
    freqs_hz = np.array([0.0, 24_000.0], dtype=np.float64)
    values_db = np.array([0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="min_freq_hz must be finite"):
        validate_fir_against_target(
            fir,
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=values_db,
            n_fft=1024,
            min_freq_hz=float("nan"),
        )

    with pytest.raises(ValueError, match="max_freq_hz must be finite"):
        validate_fir_against_target(
            fir,
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=values_db,
            n_fft=1024,
            max_freq_hz=float("nan"),
        )

    with pytest.raises(ValueError, match="min_freq_hz must be <= max_freq_hz"):
        validate_fir_against_target(
            fir,
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=values_db,
            n_fft=1024,
            min_freq_hz=1000.0,
            max_freq_hz=20.0,
        )

    with pytest.raises(ValueError, match="fir frequency response contains non-finite values"):
        validate_fir_against_target(
            np.ones(1024, dtype=np.float64) * 1e308,
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=values_db,
            n_fft=1024,
            min_freq_hz=20.0,
            max_freq_hz=1000.0,
        )

    with pytest.raises(ValueError, match="n_fft must be >= FIR length"):
        validate_fir_against_target(
            fir,
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=values_db,
            n_fft=512,
        )


def test_validate_fir_against_target_rejects_boolean_public_arrays():
    fir = np.ones(1024, dtype=np.float64)
    freqs_hz = np.array([0.0, 24_000.0], dtype=np.float64)
    values_db = np.array([0.0, 0.0], dtype=np.float64)

    with pytest.raises(TypeError, match="fir must be numeric, not bool"):
        validate_fir_against_target(
            np.array([True, False]),
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=values_db,
            n_fft=1024,
        )

    with pytest.raises(TypeError, match="target_freqs_hz must be numeric, not bool"):
        validate_fir_against_target(
            fir,
            sample_rate=48_000,
            target_freqs_hz=np.array([False, True]),
            target_values=values_db,
            n_fft=1024,
        )

    with pytest.raises(TypeError, match="target_values must be numeric, not bool"):
        validate_fir_against_target(
            fir,
            sample_rate=48_000,
            target_freqs_hz=freqs_hz,
            target_values=np.array([False, True]),
            n_fft=1024,
        )
