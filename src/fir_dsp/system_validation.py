from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .analyze import analyze_fir_quality
from .target_validation import validate_fir_against_target
from .core_validation import run_core_validations


@dataclass(frozen=True)
class SystemValidationResult:
    status: str  # PASS / WARN / FAIL
    violations: List[str]
    warnings: List[str]


def validate_system(
    fir: np.ndarray,
    sample_rate: int,
    *,
    target_magnitude: Optional[np.ndarray] = None,
    fft_size: Optional[int] = None,
    target_freqs_hz: Optional[np.ndarray] = None,
    target_values: Optional[np.ndarray] = None,
    target_scale: str = "db",
    true_peak_enforced: bool = True,
    true_peak_target_dbfs: float = 0.0,
    profile=None,
) -> SystemValidationResult:

    violations: List[str] = []
    warnings: List[str] = []

    # === FIR structural analysis ===
    analysis = analyze_fir_quality(
        fir,
        sample_rate,
        true_peak_enforced=true_peak_enforced,
        true_peak_target_dbfs=true_peak_target_dbfs,
        minimum_true_peak_margin_db=0.0 if profile is None else float(getattr(profile, "minimum_true_peak_margin_db", 0.0)),
    )

    if analysis.reconstruction_class == "FRAGILE":
        violations.append("Reconstruction classified as FRAGILE")
    elif analysis.reconstruction_class == "MARGINAL":
        warnings.append("Reconstruction classified as MARGINAL")

    if analysis.preringing_risk in {"high"}:
        warnings.append(f"High pre-ringing risk: {analysis.preringing_risk}")

    # === Core validation (playback simulation) ===
    core_results = run_core_validations(
        fir,
        mode="eq",
        sample_rate=sample_rate,
        target_mag=target_magnitude,
        fft_size=fft_size,
    )

    for r in core_results:
        if r.get("status") == "FAIL" and r.get("metric") == "energy_front_ratio":
            warnings.append(f"{r.get('metric')} warning ({r.get('value')})")
        elif r.get("status") == "FAIL":
            violations.append(f"{r.get('metric')} failed ({r.get('value')})")
        elif r.get("status") == "WARN":
            warnings.append(f"{r.get('metric')} warning ({r.get('value')})")

    # === Target validation (optional) ===
    if target_freqs_hz is not None and target_values is not None:
        target_summary = validate_fir_against_target(
            fir,
            sample_rate,
            target_freqs_hz,
            target_values,
            target_scale=target_scale,
        )

        if target_summary.max_abs_error_db > 1.0:
            violations.append(
                f"Target deviation too high: {target_summary.max_abs_error_db:.3f} dB"
            )
        elif target_summary.max_abs_error_db > 0.5:
            warnings.append(
                f"Target deviation elevated: {target_summary.max_abs_error_db:.3f} dB"
            )

    # === Final decision ===
    if violations:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return SystemValidationResult(
        status=status,
        violations=violations,
        warnings=warnings,
    )
