import numpy as np
import pytest

from fir_dsp.api import generate_fir_pipeline
from fir_dsp.core import design_fir_from_mag_fft


def test_minimum_phase_default_is_finite_and_correct_length():
    mag = np.linspace(0.8, 1.2, 513, dtype=np.float64)
    result = generate_fir_pipeline(magnitude=mag, fft_size=1024, headroom_db=6.0, return_details=True)
    assert result.fir_final.shape == (1024,)
    assert np.all(np.isfinite(result.fir_final))


def test_minimum_phase_default_matches_basic_call():
    mag = np.linspace(0.8, 1.2, 513, dtype=np.float64)
    implicit = generate_fir_pipeline(magnitude=mag, fft_size=1024, headroom_db=6.0)
    detailed = generate_fir_pipeline(magnitude=mag, fft_size=1024, headroom_db=6.0, return_details=True)
    assert np.allclose(implicit, detailed.fir_final, atol=1e-6)


def test_minimum_phase_even_fft_preserves_nyquist_quefrency_term():
    fft_size = 1024
    mag = np.linspace(0.25, 1.75, fft_size // 2 + 1, dtype=np.float64)

    log_mag = np.log(mag)
    cepstrum = np.fft.irfft(log_mag, n=fft_size)
    expected_cepstrum = np.zeros_like(cepstrum)
    expected_cepstrum[0] = cepstrum[0]
    expected_cepstrum[1:fft_size // 2] = 2 * cepstrum[1:fft_size // 2]
    expected_cepstrum[fft_size // 2] = cepstrum[fft_size // 2]
    expected = np.fft.irfft(np.exp(np.fft.rfft(expected_cepstrum, n=fft_size)), n=fft_size)

    actual = design_fir_from_mag_fft(mag, fft_size=fft_size, minimum_phase=True)

    assert np.allclose(actual, expected)


def test_minimum_phase_rejects_interior_zero_magnitude_bins():
    mag = np.ones(513, dtype=np.float64)
    mag[128] = 0.0

    with pytest.raises(ValueError, match="minimum-phase design requires strictly positive magnitudes"):
        design_fir_from_mag_fft(mag, fft_size=1024, minimum_phase=True)


def test_minimum_phase_rejects_boundary_zero_magnitude_bins():
    mag = np.ones(513, dtype=np.float64)
    mag[-1] = 0.0

    with pytest.raises(ValueError, match="minimum-phase design requires strictly positive magnitudes"):
        design_fir_from_mag_fft(mag, fft_size=1024, minimum_phase=True)
