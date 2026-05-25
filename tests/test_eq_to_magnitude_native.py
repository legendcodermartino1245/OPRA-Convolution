import numpy as np
import pytest

from fir_dsp.eq_to_magnitude_native import eq_json_to_native_magnitude


def test_eq_json_to_native_magnitude_handles_empty_band_list() -> None:
    eq = {
        "data": {
            "parameters": {
                "gain_db": -3.0,
                "bands": [],
            }
        }
    }
    freqs = np.array([0.0, 20.0, 1000.0, 10_000.0], dtype=np.float64)
    magnitude = eq_json_to_native_magnitude(eq, freqs, 48_000)
    expected = np.full(freqs.shape, 10.0 ** (-3.0 / 20.0), dtype=np.float64)
    assert np.allclose(magnitude, expected)


def test_eq_json_to_native_magnitude_rejects_invalid_band_q() -> None:
    eq = {
        "data": {
            "parameters": {
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.0},
                ]
            }
        }
    }
    freqs = np.array([20.0, 1000.0, 10_000.0], dtype=np.float64)
    with pytest.raises(ValueError, match="band q must be a positive finite value"):
        eq_json_to_native_magnitude(eq, freqs, 48_000)


def test_eq_json_to_native_magnitude_rejects_frequency_at_or_above_nyquist() -> None:
    eq = {
        "data": {
            "parameters": {
                "bands": [
                    {"type": "high_shelf", "frequency": 24_000.0, "gain_db": -4.0, "q": 0.71},
                ]
            }
        }
    }
    freqs = np.array([20.0, 1000.0, 10_000.0], dtype=np.float64)
    with pytest.raises(ValueError, match="band frequency must be below Nyquist"):
        eq_json_to_native_magnitude(eq, freqs, 48_000)


def test_eq_json_to_native_magnitude_rejects_source_preamp_out_of_safe_range() -> None:
    eq = {
        "data": {
            "parameters": {
                "gain_db": 1000.0,
                "bands": [],
            }
        }
    }
    freqs = np.array([20.0, 1000.0, 10_000.0], dtype=np.float64)
    with pytest.raises(ValueError, match="preamp_db must be within the safe range"):
        eq_json_to_native_magnitude(eq, freqs, 48_000)


def test_eq_json_to_native_magnitude_rejects_band_gain_out_of_safe_range() -> None:
    eq = {
        "data": {
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 10_000.0, "q": 0.7},
                ],
            }
        }
    }
    freqs = np.array([20.0, 1000.0, 10_000.0], dtype=np.float64)
    with pytest.raises(ValueError, match="band gain_db must be within the safe range"):
        eq_json_to_native_magnitude(eq, freqs, 48_000)


def test_eq_json_to_native_magnitude_rejects_non_finite_combined_response() -> None:
    eq = {
        "data": {
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 60.0, "q": 1.0}
                    for _ in range(200)
                ],
            }
        }
    }
    freqs = np.array([1000.0], dtype=np.float64)
    with pytest.raises(ValueError, match="eq response produces non-finite magnitudes"):
        eq_json_to_native_magnitude(eq, freqs, 48_000)


def test_eq_json_to_native_magnitude_rejects_boolean_filter_parameters() -> None:
    freqs = np.array([20.0, 1000.0, 10_000.0], dtype=np.float64)

    for key in ("frequency", "gain_db", "q"):
        band = {"type": "peak_dip", "frequency": 1000.0, "gain_db": 1.0, "q": 0.7}
        band[key] = True
        eq = {
            "data": {
                "parameters": {
                    "gain_db": 0.0,
                    "bands": [band],
                }
            }
        }
        with pytest.raises(TypeError, match="must be numeric, not bool"):
            eq_json_to_native_magnitude(eq, freqs, 48_000)

    eq = {
        "data": {
            "parameters": {
                "gain_db": True,
                "bands": [],
            }
        }
    }
    with pytest.raises(TypeError, match="preamp_db must be numeric, not bool"):
        eq_json_to_native_magnitude(eq, freqs, 48_000)


def test_eq_json_to_native_magnitude_aligns_public_sample_rate_and_grid_contracts() -> None:
    eq = {
        "data": {
            "parameters": {
                "gain_db": 0.0,
                "bands": [],
            }
        }
    }

    with pytest.raises(TypeError, match="freqs_hz must be numeric, not bool"):
        eq_json_to_native_magnitude(eq, [True, False], 48_000)

    with pytest.raises(ValueError, match="sample_rate must be an integer value"):
        eq_json_to_native_magnitude(eq, [20.0, 1000.0], 48_000.5)

    with pytest.raises(ValueError, match="freqs_hz must not exceed Nyquist"):
        eq_json_to_native_magnitude(eq, [20.0, 24_000.0, 24_001.0], 48_000)


def test_eq_json_to_native_magnitude_supports_opra_style_bands() -> None:
    eq = {
        "data": {
            "parameters": {
                "gain_db": -3.6,
                "bands": [
                    {"type": "low_shelf", "frequency": 105.0, "gain_db": 6.1, "q": 0.71},
                    {"type": "peak_dip", "frequency": 2800.0, "gain_db": -3.5, "q": 2.0},
                    {"type": "high_shelf", "frequency": 10_000.0, "gain_db": -4.5, "q": 0.71},
                ],
            }
        }
    }
    freqs = np.geomspace(20.0, 20_000.0, num=512)
    magnitude = eq_json_to_native_magnitude(eq, freqs, 48_000)
    assert magnitude.shape == freqs.shape
    assert magnitude.dtype == np.float64
    assert np.all(np.isfinite(magnitude))
    assert np.all(magnitude > 0.0)
