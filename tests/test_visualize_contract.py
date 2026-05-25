from __future__ import annotations

import numpy as np
import pytest

from fir_dsp.visualize import plot_frequency_response, plot_impulse


def test_visualize_rejects_non_finite_fir_values(tmp_path):
    with pytest.raises(ValueError, match="fir must contain only finite values"):
        plot_impulse(np.array([1.0, np.nan]), save_path=tmp_path / "impulse.png")

    with pytest.raises(ValueError, match="fir must contain only finite values"):
        plot_frequency_response(np.array([1.0, np.inf]), save_path=tmp_path / "response.png")

    with pytest.raises(ValueError, match="frequency response contains non-finite values"):
        plot_frequency_response(np.ones(1024, dtype=np.float64) * 1e308, save_path=tmp_path / "response.png")


def test_visualize_rejects_invalid_target_and_sample_rate(tmp_path):
    with pytest.raises(TypeError, match="sample_rate must be numeric, not bool"):
        plot_frequency_response(np.ones(4), fs=True, save_path=tmp_path / "response.png")

    with pytest.raises(ValueError, match="target_magnitude must contain only finite values"):
        plot_frequency_response(
            np.ones(4),
            target_magnitude=np.array([1.0, np.nan, 1.0]),
            save_path=tmp_path / "response.png",
        )
