import numpy as np
import pytest

from fir_dsp.api import generate_fir_pipeline
from fir_dsp.core import compute_frequency_response, design_fir_from_mag_fft



def test_fir_not_silent():
    fir = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
    )
    assert np.max(np.abs(fir)) > 0



def test_output_is_finite():
    fir = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
    )
    assert np.all(np.isfinite(fir)), "FIR contains NaN or Inf"



def test_peak_matches_headroom():
    fir = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
    )
    expected_peak = 10 ** (-7.0 / 20.0)
    assert np.isclose(np.max(np.abs(fir)), expected_peak, atol=1e-3)



def test_frequency_response_matches_flat_target_within_tolerance():
    target = np.ones(513, dtype=np.float64)
    result = generate_fir_pipeline(
        magnitude=target,
        fft_size=1024,
        headroom_db=0.0,
        return_details=True,
    )
    actual = compute_frequency_response(result.fir_final)
    expected = target * (10 ** (-1.0 / 20.0))
    assert np.max(np.abs(actual - expected)) < 1e-3
    assert result.error.max_abs_error_db < 1e-2


def test_minimum_phase_pipeline_rejects_zero_magnitude_bins():
    target = np.ones(513, dtype=np.float64)
    target[256] = 0.0

    with pytest.raises(ValueError, match="minimum-phase design requires strictly positive magnitudes"):
        generate_fir_pipeline(
            magnitude=target,
            fft_size=1024,
            headroom_db=0.0,
        )


def test_frequency_response_rejects_fft_shorter_than_fir():
    with pytest.raises(ValueError, match="fft_size must be >= FIR length"):
        compute_frequency_response(np.ones(8, dtype=np.float64), fft_size=4)


def test_core_design_rejects_fft_size_one_with_clear_error():
    with pytest.raises(ValueError, match="fft_size must be an even integer >= 2"):
        design_fir_from_mag_fft(np.ones(1, dtype=np.float64), 1)
