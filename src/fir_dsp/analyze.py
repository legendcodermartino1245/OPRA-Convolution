from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_ext import analyze_ext
from .artifact_metrics import effective_length, float_safety, gain_summary
from .core import compute_true_peak
from .validation import (
    ensure_1d_finite_array,
    ensure_bool,
    ensure_non_negative_float,
    ensure_numeric_scalar,
    ensure_positive_int,
    ensure_positive_sample_rate,
)

EPS = 1e-12
TRUE_PEAK_ANALYSIS_TOLERANCE_DB = 0.01
ANALYSIS_SIGNAL_ABS_LIMIT = 1e150


@dataclass(frozen=True)
class FirQualitySummary:
    reconstruction_class: str
    sample_rate: int
    taps: int
    peak_index: int
    peak_latency_ms: float
    energy_centroid_ms: float
    sample_peak_linear: float
    sample_peak_dbfs: float
    true_peak_linear: float
    true_peak_dbfs: float
    crest_factor_db: float
    energy_before_peak_pct: float
    energy_after_peak_pct: float
    abs_sum_before_peak_pct: float
    abs_sum_after_peak_pct: float
    preringing_risk: str
    phase_character: str
    transient_character: str
    quality_verdict: str
    pre_ringing_ms: float
    step_response_overshoot: float
    low_band_energy_pct: float
    mid_band_energy_pct: float
    high_band_energy_pct: float
    impulse_kurtosis: float
    effective_length: dict[str, float | int]
    gain_summary: dict[str, float]
    float_safety: dict[str, Any]


@dataclass(frozen=True)
class _SpectralContext:
    spectrum: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    freqs_hz: np.ndarray
    power: np.ndarray
    log_magnitude: np.ndarray



def _db(value: float, floor: float = EPS) -> float:
    return float(20.0 * np.log10(max(float(value), floor)))



def _safe_ratio_db(a: float, b: float) -> float:
    return float(20.0 * np.log10(max(float(a), EPS) / max(float(b), EPS)))



def _build_spectral_context(fir: np.ndarray, sample_rate: int) -> _SpectralContext:
    spectrum = np.fft.rfft(fir)
    magnitude = np.abs(spectrum)
    phase = np.unwrap(np.angle(spectrum))
    freqs_hz = np.fft.rfftfreq(fir.size, d=1.0 / float(sample_rate))
    power = np.square(magnitude)
    log_magnitude = np.log(np.maximum(magnitude, EPS))
    return _SpectralContext(
        spectrum=spectrum,
        magnitude=magnitude,
        phase=phase,
        freqs_hz=freqs_hz,
        power=power,
        log_magnitude=log_magnitude,
    )


def _band_energy_pct(power: np.ndarray, freqs_hz: np.ndarray, f_lo: float, f_hi: float) -> float:
    total_power = float(np.sum(power))
    if total_power <= EPS:
        return 0.0
    mask = (freqs_hz >= f_lo) & (freqs_hz < f_hi)
    return float((np.sum(power[mask]) / total_power) * 100.0)



def analyze_fir_quality(
    fir: np.ndarray,
    sample_rate: int,
    oversample_factor: int = 8,
    true_peak_enforced: bool = True,
    true_peak_target_dbfs: float = 0.0,
    minimum_true_peak_margin_db: float = 0.0,
) -> FirQualitySummary:
    fir = ensure_1d_finite_array("fir", fir)
    if fir.size and float(np.max(np.abs(fir))) > ANALYSIS_SIGNAL_ABS_LIMIT:
        raise ValueError("fir magnitude is too large for finite analyzer metrics")
    sample_rate = ensure_positive_sample_rate(sample_rate)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    true_peak_enforced = ensure_bool("true_peak_enforced", true_peak_enforced)
    true_peak_target_dbfs = ensure_numeric_scalar("true_peak_target_dbfs", true_peak_target_dbfs)
    minimum_true_peak_margin_db = ensure_non_negative_float(
        "minimum_true_peak_margin_db", minimum_true_peak_margin_db
    )

    taps = int(fir.size)
    abs_fir = np.abs(fir)
    sq_fir = np.square(fir)
    idx = np.arange(taps, dtype=np.float64)
    peak_index = int(np.argmax(abs_fir))

    total_energy = float(np.sum(sq_fir))
    total_abs = float(np.sum(abs_fir))
    if total_energy > 0:
        energy_centroid_index = float(np.sum(idx * sq_fir) / total_energy)
    else:
        energy_centroid_index = 0.0

    before = fir[:peak_index]
    after = fir[peak_index + 1 :]
    before_energy = float(np.sum(np.square(before)))
    after_energy = float(np.sum(np.square(after)))
    before_abs = float(np.sum(np.abs(before)))
    after_abs = float(np.sum(np.abs(after)))
    asymmetry_ratio_db = _safe_ratio_db(after_abs + EPS, before_abs + EPS)

    sample_peak_linear = float(np.max(abs_fir))
    rms = float(np.sqrt(np.mean(sq_fir)))
    true_peak_linear = float(compute_true_peak(fir, oversample_factor=oversample_factor))
    true_peak_dbfs = _db(true_peak_linear)
    true_peak_margin = true_peak_target_dbfs - true_peak_dbfs
    required_true_peak_margin_db = minimum_true_peak_margin_db

    if not true_peak_enforced:
        reconstruction_class = "MEASURE_ONLY"
    elif true_peak_margin + TRUE_PEAK_ANALYSIS_TOLERANCE_DB >= required_true_peak_margin_db:
        reconstruction_class = "SAFE"
    elif true_peak_margin >= required_true_peak_margin_db - 0.1:
        reconstruction_class = "MARGINAL"
    else:
        reconstruction_class = "FRAGILE"

    energy_before_peak_pct = float((before_energy / max(total_energy, EPS)) * 100.0)
    energy_after_peak_pct = float((after_energy / max(total_energy, EPS)) * 100.0)
    abs_sum_before_peak_pct = float((before_abs / max(total_abs, EPS)) * 100.0)
    abs_sum_after_peak_pct = float((after_abs / max(total_abs, EPS)) * 100.0)

    if energy_before_peak_pct <= 0.5 and abs_sum_before_peak_pct <= 3.0:
        preringing_risk = "very_low"
    elif energy_before_peak_pct <= 2.0 and abs_sum_before_peak_pct <= 8.0:
        preringing_risk = "low"
    elif energy_before_peak_pct <= 8.0:
        preringing_risk = "moderate"
    else:
        preringing_risk = "high"

    peak_latency_ms = float((peak_index / float(sample_rate)) * 1000.0)
    energy_centroid_ms = float((energy_centroid_index / float(sample_rate)) * 1000.0)

    if float(np.max(np.abs(fir - fir[::-1]))) < 1e-9:
        phase_character = "linear_like"
    elif peak_index <= max(2, int(0.02 * taps)):
        phase_character = "minimum_like"
    else:
        phase_character = "shifted_or_mixed"
    if asymmetry_ratio_db >= 18.0 and preringing_risk in {"very_low", "low"}:
        transient_character = "front_loaded_clean"
    elif asymmetry_ratio_db >= 8.0:
        transient_character = "mostly_front_loaded"
    elif asymmetry_ratio_db > -3.0:
        transient_character = "balanced_or_broad"
    else:
        transient_character = "rear_weighted"

    if preringing_risk == "very_low" and transient_character == "front_loaded_clean":
        quality_verdict = "excellent_for_minimum_phase"
    elif preringing_risk in {"very_low", "low"} and transient_character in {"front_loaded_clean", "mostly_front_loaded"}:
        quality_verdict = "strong"
    elif preringing_risk == "moderate":
        quality_verdict = "usable_but_check_by_ear"
    else:
        quality_verdict = "risky_for_transient_purity"

    spectral = _build_spectral_context(fir, sample_rate)

    peak_value = sample_peak_linear
    pre_threshold = peak_value * 0.01
    significant_pre = np.flatnonzero(abs_fir[:peak_index] >= pre_threshold)
    if significant_pre.size:
        pre_ringing_ms = float(((peak_index - int(significant_pre[0])) / float(sample_rate)) * 1000.0)
    else:
        pre_ringing_ms = 0.0

    step_response = np.cumsum(fir)
    final_value = step_response[-1]
    step_response_overshoot = float(np.max(step_response - final_value))

    low_band_energy_pct = _band_energy_pct(spectral.power, spectral.freqs_hz, 20.0, 200.0)
    mid_band_energy_pct = _band_energy_pct(spectral.power, spectral.freqs_hz, 200.0, 2000.0)
    high_band_energy_pct = _band_energy_pct(spectral.power, spectral.freqs_hz, 2000.0, 20000.0)

    centered = fir - float(np.mean(fir))
    sigma = float(np.std(centered))
    impulse_kurtosis = float(np.mean(np.power(centered / max(sigma, EPS), 4.0)))
    effective_length_summary = effective_length(fir, sample_rate)
    gain_summary_payload = gain_summary(fir, sample_rate, fft_size=max(taps, 4096))
    float_safety_payload = float_safety(fir)

    return FirQualitySummary(
        reconstruction_class=reconstruction_class,
        sample_rate=int(sample_rate),
        taps=taps,
        peak_index=peak_index,
        peak_latency_ms=peak_latency_ms,
        energy_centroid_ms=energy_centroid_ms,
        sample_peak_linear=sample_peak_linear,
        sample_peak_dbfs=_db(sample_peak_linear),
        true_peak_linear=true_peak_linear,
        true_peak_dbfs=true_peak_dbfs,
        crest_factor_db=_safe_ratio_db(sample_peak_linear, rms + EPS),
        energy_before_peak_pct=energy_before_peak_pct,
        energy_after_peak_pct=energy_after_peak_pct,
        abs_sum_before_peak_pct=abs_sum_before_peak_pct,
        abs_sum_after_peak_pct=abs_sum_after_peak_pct,
        preringing_risk=preringing_risk,
        phase_character=phase_character,
        transient_character=transient_character,
        quality_verdict=quality_verdict,
        pre_ringing_ms=pre_ringing_ms,
        step_response_overshoot=step_response_overshoot,
        low_band_energy_pct=low_band_energy_pct,
        mid_band_energy_pct=mid_band_energy_pct,
        high_band_energy_pct=high_band_energy_pct,
        impulse_kurtosis=impulse_kurtosis,
        effective_length=asdict(effective_length_summary),
        gain_summary=asdict(gain_summary_payload),
        float_safety=asdict(float_safety_payload),
    )



def load_fir(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"FIR file not found: {path}")
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.ndim != 2:
        raise ValueError("FIR file must resolve to one column of coefficients")
    if data.shape[1] == 1:
        return ensure_1d_finite_array("FIR file coefficients", data[:, 0])
    if data.shape[1] == 2:
        return ensure_1d_finite_array("FIR file coefficients", data[:, 0])
    raise ValueError("FIR file must contain one column of coefficients")



def format_summary(summary: FirQualitySummary) -> str:
    lines = [
        f"Taps: {summary.taps}",
        f"Sample rate: {summary.sample_rate} Hz",
        f"Peak index: {summary.peak_index}",
        f"Peak latency: {summary.peak_latency_ms:.3f} ms",
        f"Energy centroid (sq-weighted): {summary.energy_centroid_ms:.3f} ms",
        f"Sample peak: {summary.sample_peak_dbfs:.3f} dBFS",
        f"True peak: {summary.true_peak_dbfs:.3f} dBFS",
        f"Reconstruction: {summary.reconstruction_class}",
        f"Crest factor: {summary.crest_factor_db:.3f} dB",
        f"Energy before peak: {summary.energy_before_peak_pct:.4f}%",
        f"Energy after peak: {summary.energy_after_peak_pct:.4f}%",
        f"Abs sum before peak: {summary.abs_sum_before_peak_pct:.4f}%",
        f"Abs sum after peak: {summary.abs_sum_after_peak_pct:.4f}%",
        f"Pre-ringing risk: {summary.preringing_risk}",
        f"Phase character: {summary.phase_character}",
        f"Transient character: {summary.transient_character}",
        f"Quality verdict: {summary.quality_verdict}",
        f"Pre-ringing duration: {summary.pre_ringing_ms:.6f} ms",
        f"Step-response overshoot: {summary.step_response_overshoot:.6f}",
        f"Low-band energy: {summary.low_band_energy_pct:.4f}%",
        f"Mid-band energy: {summary.mid_band_energy_pct:.4f}%",
        f"High-band energy: {summary.high_band_energy_pct:.4f}%",
        f"Impulse kurtosis: {summary.impulse_kurtosis:.6f}",
        f"Effective length (99% energy): {summary.effective_length['energy_99_ms']:.6f} ms",
        f"Gain max/min: {summary.gain_summary['max_gain_db']:.3f} / {summary.gain_summary['min_gain_db']:.3f} dB",
        f"Float safety: nan={summary.float_safety['has_nan']} inf={summary.float_safety['has_inf']} denormals={summary.float_safety['has_denormals']}",
    ]
    return "\n".join(lines)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze FIR quality and transient behavior"
    )

    parser.add_argument("--fir", "--input", dest="fir", type=Path, required=True, help="Path to FIR coefficient text file")
    parser.add_argument("--sample-rate", type=int, required=True, help="FIR sample rate in Hz")
    parser.add_argument("--oversample-factor", type=int, default=8, help="Oversampling factor for true-peak analysis")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--strict", action="store_true")

    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fir = load_fir(args.fir)
    sample_rate = args.sample_rate

    summary = analyze_fir_quality(
        fir,
        sample_rate=sample_rate,
        oversample_factor=args.oversample_factor,
    )

    print(format_summary(summary))

    extended_results: list[dict[str, float | str]] | None = None
    extended_mode: str | None = None

    if args.extended:
        extended_mode = "strict" if args.strict else "fast"
        extended_results = analyze_ext(fir, sample_rate, mode=extended_mode)

        print("\n--- Extended Validation ---")

        IGNORED_METRICS = set()

        real_failures = []
        has_warn = False

        for r in extended_results:
            value_str = f"{r['value']:.6f}" if r["value"] is not None else "N/A"
            print(f"{r['metric']}: {value_str} [{r['status']}]")

            if r["status"] == "FAIL" and r["metric"] not in IGNORED_METRICS:
                real_failures.append(r)

            if r["status"] == "WARN":
                has_warn = True

        if real_failures:
            print("\n[ERROR] VALIDATION FAILED (real issues only)")
            for f in real_failures:
                value_str = f"{f['value']:.6f}" if f["value"] is not None else "N/A"
                print(f" - {f['metric']}: {value_str}")
            return 2

        if has_warn and args.strict:
            print("\n[WARN] treated as FAIL (--strict)")
            return 1

        print("\n[SUCCESS] VALIDATION PASSED")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(summary)
        if extended_results is not None:
            payload["extended_validation"] = {
                "mode": extended_mode,
                "results": extended_results,
            }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
