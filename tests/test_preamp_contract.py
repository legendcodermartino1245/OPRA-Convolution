import numpy as np
import pytest

from fir_dsp.api import generate_fir_multi_rate, generate_fir_pipeline
from fir_dsp.preamp import apply_preamp_db, validate_preamp_db


def test_manual_preamp_is_rejected_by_default_profile() -> None:
    magnitude = np.ones(513, dtype=np.float64)
    with pytest.raises(ValueError, match="trusted pipeline data"):
        generate_fir_pipeline(
            magnitude=magnitude,
            fft_size=1024,
            headroom_db=6.0,
            preamp_db=-6.0,
        )


def test_double_preamp_detection_is_explicit() -> None:
    magnitude = np.ones(513, dtype=np.float64)
    with pytest.raises(ValueError, match="Preamp would be applied twice"):
        generate_fir_pipeline(
            magnitude=magnitude,
            fft_size=1024,
            headroom_db=6.0,
            preamp_db=-6.0,
            preamp_source="peq",
            preamp_already_applied=True,
        )


def test_multi_rate_manual_preamp_is_rejected_by_default_profile() -> None:
    magnitude = np.ones(513, dtype=np.float64)
    with pytest.raises(ValueError, match="trusted pipeline data"):
        generate_fir_multi_rate(
            magnitude=magnitude,
            rates=[44100, 48000],
            fft_size=1024,
            headroom_db=6.0,
            preamp_db=-6.0,
        )


def test_preamp_out_of_safe_range_is_rejected() -> None:
    values = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"safe range \[-120, 60\] dB"):
        apply_preamp_db(values, 200.0)
    with pytest.raises(ValueError, match=r"safe range \[-120, 60\] dB"):
        apply_preamp_db(values, -300.0)


def test_apply_preamp_rejects_non_finite_output() -> None:
    with pytest.raises(ValueError, match="preamp produces non-finite linear magnitude"):
        apply_preamp_db(np.array([1e306], dtype=np.float64), 60.0)


def test_boolean_preamp_is_rejected() -> None:
    with pytest.raises(TypeError, match="preamp_db must be numeric, not bool"):
        validate_preamp_db(True)

    with pytest.raises(TypeError, match="preamp_db must be numeric, not bool"):
        generate_fir_pipeline(
            np.ones(513, dtype=np.float64) * 0.5,
            1024,
            6.0,
            true_peak=True,
            preamp_db=True,
            preamp_source="peq",
            return_details=True,
        )


def test_final_true_peak_contract_passes_when_true_peak_normalized() -> None:
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=0.0,
        true_peak=True,
        return_details=True,
    )

    assert result.verification.fir_true_peak_dbfs <= result.verification.true_peak_target_dbfs + 0.001
    assert result.verification.true_peak_margin_db >= -1e-3


def test_verification_metadata_exposes_true_peak_margin_threshold() -> None:
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=0.0,
        true_peak=True,
        return_details=True,
    )

    assert result.verification.true_peak_margin_db >= -1e-3
    assert result.verification.true_peak_min_safe_margin_db == pytest.approx(1.0)
    assert result.verification.true_peak_margin_warning is False
