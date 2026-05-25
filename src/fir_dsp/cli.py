from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from .api import _target_projection_metadata, generate_fir_multi_rate, generate_fir_pipeline
from .artifact_metrics import cross_rate_consistency
from .core import (
    TRUE_PEAK_MARGIN_TOLERANCE_DB,
    coeff_hash_sha256,
    compute_frequency_response,
    compute_true_peak,
    linear_to_db,
    request_fingerprint,
    summarize_response_error,
)
from .logging_utils import configure_logging, logger
from .opra_selector import (
    DEFAULT_OPRA_DB_URL,
    build_ui_attribution,
    eq_to_txt,
    find_product,
    load_opra_jsonl,
    select_eq,
)
from .profiles import PROFILE_REGISTRY
from .report import metadata_json_dumps, write_pipeline_report
from .system_validation import validate_system
from .target_validation import TargetValidationSummary, validate_fir_against_target
from .validation import ensure_1d_finite_array

STANDARD_PCM_RATES: tuple[int, ...] = (44_100, 48_000, 88_200, 96_000, 176_400, 192_000, 352_800, 384_000)

MODE_DEFAULTS: dict[str, dict[str, object]] = {
    "reference": {"true_peak": True, "oversample_factor": 8, "window_preset": None, "design_os": 1},
    "high_precision": {"true_peak": True, "oversample_factor": 8, "window_preset": None, "design_os": 1},
    "mart_reference": {"true_peak": True, "oversample_factor": 8, "window_preset": None, "design_os": 1},
}
OPRA_REPO_URL = "https://github.com/opra-project/OPRA"


def load_response(path: Path) -> tuple[np.ndarray | None, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Response file not found: {path}")
    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except Exception as e:
        raise RuntimeError(f"Failed to load response file: {e}") from e
    if data.ndim != 2:
        raise ValueError("Response file must resolve to a 1-column FFT magnitude or 2-column freq_hz/value table")
    if data.shape[1] == 1:
        return None, ensure_1d_finite_array("response values", data[:, 0])
    if data.shape[1] == 2:
        return (
            ensure_1d_finite_array("response frequencies", data[:, 0]),
            ensure_1d_finite_array("response values", data[:, 1]),
        )
    raise ValueError("Response file must be either 1 column (FFT magnitude) or 2 columns (freq_hz value)")


def save_fir(path: Path, fir: np.ndarray) -> None:
    fir = ensure_1d_finite_array("fir", fir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savetxt(path, fir, fmt="%.17e")
    except Exception as e:
        raise RuntimeError(f"Failed to save FIR: {e}") from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic minimum-phase FIR DSP toolchain")
    parser.add_argument("--doctor", action="store_true", help="Print environment and deterministic self-check information")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--response", "--input", dest="response", type=Path, help="Path to response file")
    source_group.add_argument("--opra-query", type=str, help="Resolve an OPRA product by name/ID and use its EQ directly from the database")
    parser.add_argument("--opra-db", default=DEFAULT_OPRA_DB_URL, help="OPRA database URL or local JSONL path")
    parser.add_argument("--opra-target", type=str, help="Restrict OPRA selection to a target id suffix such as autoeq_kazi or oratory1990_harman_target")
    parser.add_argument("--opra-measurement", type=str, help="Restrict OPRA selection to a measurement source such as kazi or crinacle")
    parser.add_argument("--input-scale", choices=["linear", "db"], default="db")
    parser.add_argument("--fft-size", type=int)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="For OPRA/eq_json generation, evaluate the shared target curve at this playback sample rate before per-rate discretization",
    )
    parser.add_argument("--headroom-db", type=float)
    parser.add_argument("--out", "--output", dest="out", type=Path)
    parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS), help="Preset mode for safe defaults")
    parser.add_argument("--profile", choices=sorted(PROFILE_REGISTRY), default="default", help="Pipeline policy profile")
    parser.add_argument("--true-peak", action="store_true")
    parser.add_argument("--no-true-peak", dest="true_peak", action="store_false")
    parser.set_defaults(true_peak=None)
    parser.add_argument("--oversample-factor", type=int, default=None, help="Oversampling factor for true-peak checks")
    parser.add_argument("--rates", type=int, nargs="+", help="Generate FIR for multiple sample rates")
    parser.add_argument("--all-pcm", action="store_true", help="Generate FIR for standard PCM rates 44.1k through 384k")
    parser.add_argument("--window", choices=["none", "hann", "kaiser", "blackman"], help="Explicit window type; use 'none' to disable windowing")
    parser.add_argument("--window-beta", type=float, default=8.6)
    parser.add_argument("--window-preset", choices=["safe", "sharp", "minimal_ringing"], help="Named window preset")
    parser.add_argument("--no-window", action="store_true", help="Disable mode/default windowing for exact target matching")
    parser.add_argument(
        "--design-os",
        type=int,
        default=None,
        help="Reserved design-grid compatibility knob. Values >1 currently keep the stable direct-grid design instead of running a separate oversampled-decimation path.",
    )
    parser.add_argument("--export-json", action="store_true", help="Write per-rate JSON metadata alongside FIR text files")
    parser.add_argument(
        "--reproducible",
        action="store_true",
        help="Write stable JSON metadata for reproducible artifacts: sorted keys, rounded diagnostic floats, and no runtime timing.",
    )
    parser.add_argument(
        "--audible-stress",
        action="store_true",
        help="Include expensive program-material stress probes in exported JSON metadata.",
    )
    parser.add_argument("--plot", action="store_true", help="Write impulse and response plots alongside outputs")
    parser.add_argument("--output-dir", type=Path, help="Directory for multi-rate exports; filenames are derived from --out")
    parser.add_argument("--validate-target", action="store_true", help="Validate the generated FIR against a target response in the frequency domain")
    parser.add_argument("--target", type=Path, help="Optional target response file for validation; defaults to --response")
    parser.add_argument("--target-scale", choices=["linear", "db"], help="Scale for --target values; required when --target points to a separate file")
    parser.add_argument("--target-min-freq", type=float, help="Lower frequency bound for target validation in Hz; defaults to 20")
    parser.add_argument("--target-max-freq", type=float, help="Upper frequency bound for target validation in Hz; defaults to Nyquist")
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging")
    return parser


def _write_outputs(
    base_out: Path,
    result,
    export_json: bool,
    plot: bool,
    elapsed_ms: float | None = None,
    mode: str | None = None,
    target_validation: TargetValidationSummary | None = None,
    cross_rate_consistency_payload: dict[str, object] | None = None,
    include_stress_probes: bool = False,
    reproducible: bool = False,
) -> None:
    save_fir(base_out, result.fir_final)
    write_pipeline_report(
        base_out=base_out,
        result=result,
        export_json=export_json,
        plot=plot,
        elapsed_ms=elapsed_ms,
        mode=mode,
        target_validation=target_validation,
        cross_rate_consistency=cross_rate_consistency_payload,
        include_stress_probes=include_stress_probes,
        reproducible=reproducible,
    )


def _build_release_verdict(
    results: dict[int, object],
    cross_rate_payload: dict[str, object] | None,
) -> dict[str, object]:
    playback_safe = all(
        not bool(result.verification.true_peak_margin_warning)
        for result in results.values()
    )
    export_safe = all(
        result.error.max_abs_error_db <= 0.1
        for result in results.values()
    )
    cross_rate_consistent = bool(cross_rate_payload is None or cross_rate_payload.get("strict_all_rates_pass", False))
    if export_safe and playback_safe and cross_rate_consistent:
        recommended_use = "production_release"
    elif export_safe and playback_safe:
        recommended_use = "safe_to_use_but_review_cross_rate"
    else:
        recommended_use = "review_before_release"
    return {
        "export_safe": bool(export_safe),
        "playback_safe": bool(playback_safe),
        "cross_rate_consistent": bool(cross_rate_consistent),
        "recommended_use": recommended_use,
    }


def _apply_opra_post_scale(result, desired_margin_db: float):
    desired_margin_db = float(max(0.0, desired_margin_db))
    current_margin_db = float(result.verification.true_peak_margin_db)
    attenuation_db = desired_margin_db - current_margin_db

    if attenuation_db <= 1e-9:
        return result, 0.0

    scale = float(10.0 ** (-attenuation_db / 20.0))
    fir_final = np.asarray(result.fir_final * scale, dtype=np.float64)
    actual_magnitude = compute_frequency_response(fir_final, fft_size=result.fft_size)
    actual_magnitude_db = linear_to_db(actual_magnitude)
    error = summarize_response_error(result.target_magnitude, actual_magnitude)

    fir_peak_linear = float(np.max(np.abs(fir_final)))
    fir_true_peak_linear = compute_true_peak(fir_final, oversample_factor=result.oversample_factor)
    fir_true_peak_dbfs = float(20.0 * np.log10(max(fir_true_peak_linear, 1e-12)))
    true_peak_margin_linear = float(result.verification.true_peak_target_linear - fir_true_peak_linear)
    true_peak_margin_db = float(result.verification.true_peak_target_dbfs - fir_true_peak_dbfs)
    total_post_scale_attenuation_db = float(result.spec.post_scale_attenuation_db) + attenuation_db
    spec = replace(result.spec, post_scale_attenuation_db=total_post_scale_attenuation_db)
    final_baked_headroom_db = float(spec.normalization_headroom_db)
    request_fingerprint_sha256 = request_fingerprint(
        {
            "base_request_fingerprint_sha256": result.verification.request_fingerprint_sha256,
            "opra_post_scale_attenuation_db": attenuation_db,
            "total_post_scale_attenuation_db": total_post_scale_attenuation_db,
            "post_scale_policy": "opra_true_peak_margin",
        }
    )

    verification = replace(
        result.verification,
        fir_peak_linear=fir_peak_linear,
        fir_peak_dbfs=float(20.0 * np.log10(max(fir_peak_linear, 1e-12))),
        fir_true_peak_linear=fir_true_peak_linear,
        fir_true_peak_dbfs=fir_true_peak_dbfs,
        true_peak_margin_linear=true_peak_margin_linear,
        true_peak_margin_db=true_peak_margin_db,
        true_peak_margin_warning=bool(
            true_peak_margin_db + TRUE_PEAK_MARGIN_TOLERANCE_DB
            < float(result.verification.true_peak_min_safe_margin_db)
        ),
        final_baked_headroom_db=final_baked_headroom_db,
        coeff_hash_sha256=coeff_hash_sha256(fir_final),
        request_fingerprint_sha256=request_fingerprint_sha256,
    )
    gain_traceability = replace(
        result.gain_traceability,
        final_baked_headroom_db=final_baked_headroom_db,
    )
    system_validation = validate_system(
        fir_final,
        sample_rate=result.sample_rate,
        target_magnitude=result.target_magnitude,
        fft_size=result.fft_size,
        true_peak_enforced=bool(result.true_peak),
        true_peak_target_dbfs=float(verification.true_peak_target_dbfs),
        profile=spec.profile_config,
    )

    return replace(
        result,
        spec=spec,
        headroom_db=float(spec.requested_headroom_db),
        fir_final=fir_final,
        actual_magnitude=actual_magnitude,
        actual_magnitude_db=actual_magnitude_db,
        error=error,
        verification=verification,
        gain_traceability=gain_traceability,
        system_validation=system_validation,
    ), attenuation_db


def _write_opra_sidecars(
    root_dir: Path,
    stem: str,
    *,
    product: dict,
    eq: dict,
    vendors: dict,
    db_source: str,
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{stem}_" if stem else ""
    (root_dir / f"{prefix}selected_eq.txt").write_text(eq_to_txt(eq), encoding="utf-8")
    (root_dir / f"{prefix}NOTICE_OPRA.txt").write_text(
        "\n".join(
            [
                "This output was generated from OPRA database content.",
                "",
                "Required attribution:",
                "- Product and preset data source: OPRA Project",
                f"- Repository: {OPRA_REPO_URL}",
                f"- Database source used: {db_source}",
                "- Hosted mirror guidance for app/database consumers: https://opra.roonlabs.net/database_v1.jsonl",
                "",
                "License notes:",
                "- OPRA repository source code is published under the MIT License.",
                "- OPRA dataset content is published under CC BY-SA 4.0.",
                "- If you redistribute this output, preserve attribution and review whether your distribution needs to comply with CC BY-SA 4.0 share-alike terms.",
                "",
                f"Selected product: {product['id']}",
            ]
        ),
        encoding="utf-8",
    )
    attribution_payload = {
        "source": "OPRA",
        "repository": OPRA_REPO_URL,
        "database_source": db_source,
        **build_ui_attribution(product, eq, vendors),
    }
    (root_dir / f"{prefix}ATTRIBUTION_OPRA.json").write_text(
        metadata_json_dumps(attribution_payload, reproducible=True),
        encoding="utf-8",
    )


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.mode:
        return args
    defaults = MODE_DEFAULTS[args.mode]
    if args.true_peak is None:
        args.true_peak = bool(defaults["true_peak"])
    if args.oversample_factor is None:
        args.oversample_factor = int(defaults["oversample_factor"])
    default_window_preset = defaults.get("window_preset")
    if default_window_preset is not None and not args.no_window and args.window is None and args.window_preset is None:
        args.window_preset = str(default_window_preset)
    if args.design_os is None:
        args.design_os = int(defaults["design_os"])
    return args


def _resolve_target_validation_source(
    args: argparse.Namespace,
    response_freqs_hz: np.ndarray | None,
    response_values: np.ndarray,
    inferred_input_scale: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if args.response is None:
        raise ValueError("Target validation is only supported when --response is used")
    target_path = args.target if args.target is not None else args.response
    target_freqs_hz, target_values = load_response(target_path)
    if target_freqs_hz is None:
        if target_path == args.response and response_freqs_hz is not None:
            return response_freqs_hz, response_values, args.target_scale or inferred_input_scale
        raise ValueError(
            "Target validation requires a 2-column frequency response file (freq_hz value). Use --target with a frequency-domain target."
        )
    if target_path != args.response and args.target_scale is None:
        raise ValueError("--target-scale must be provided when --target points to a separate file")
    target_scale = args.target_scale or inferred_input_scale
    return (target_freqs_hz, target_values, "linear") if target_scale == "linear" else (target_freqs_hz, target_values, "db")


def _normalize_probe_peak(signal: np.ndarray, peak_linear: float) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        raise ValueError("Probe signal must not be silent")
    return np.asarray((signal / peak) * peak_linear, dtype=np.float64)


def _build_doctor_program_probes(length: int, peak_linear: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260401)
    t = np.arange(length, dtype=np.float64)
    sweep_phase = 2.0 * np.pi * (20.0 * t / length + ((20_000.0 - 20.0) / (2.0 * length * length)) * t * t)
    multitone = (
        np.sin(2.0 * np.pi * 53.0 * t / length)
        + np.sin(2.0 * np.pi * 997.0 * t / length)
        + np.sin(2.0 * np.pi * 5021.0 * t / length)
    )
    probes = {
        "step": np.ones(length, dtype=np.float64),
        "alternating": np.where((t.astype(np.int64) % 2) == 0, 1.0, -1.0),
        "multitone": multitone,
        "sweep": np.sin(sweep_phase),
        "noise": rng.standard_normal(length),
    }
    return {name: _normalize_probe_peak(signal, peak_linear) for name, signal in probes.items()}


def _simulate_program_material_true_peak(
    fir: np.ndarray,
    *,
    requested_headroom_db: float,
    oversample_factor: int,
    probe_length: int = 32768,
) -> tuple[dict[str, float], float]:
    playback_peak_linear = 10.0 ** (-float(requested_headroom_db) / 20.0)
    probes = _build_doctor_program_probes(probe_length, playback_peak_linear)
    peaks: dict[str, float] = {}
    for name, signal in probes.items():
        convolved = np.convolve(signal, fir, mode="same")
        peaks[name] = 20.0 * np.log10(max(compute_true_peak(convolved, oversample_factor=oversample_factor), 1e-12))
    worst_case = max(peaks.values())
    return peaks, worst_case


def _run_doctor() -> int:
    probe = np.ones(513, dtype=np.float64)
    result_a = generate_fir_pipeline(
        probe,
        1024,
        6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
        profile="default",
        design_oversample=1,
    )
    result_b = generate_fir_pipeline(
        probe,
        1024,
        6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
        profile="default",
        design_oversample=1,
    )
    deterministic = (
        np.array_equal(result_a.fir_final, result_b.fir_final)
        and result_a.verification.coeff_hash_sha256 == result_b.verification.coeff_hash_sha256
        and result_a.verification.request_fingerprint_sha256 == result_b.verification.request_fingerprint_sha256
    )
    profile_name = getattr(result_a.profile, "name", result_a.profile)
    spec_profile_name = getattr(result_a.spec.profile, "name", result_a.spec.profile)
    simulated_probe_peaks_dbfs, worst_case_probe_true_peak_dbfs = _simulate_program_material_true_peak(
        result_a.fir_final,
        requested_headroom_db=result_a.spec.requested_headroom_db,
        oversample_factor=result_a.spec.oversample_factor,
    )
    strict_checks = {
        "deterministic": deterministic,
        "true_peak_within_target": result_a.verification.fir_true_peak_dbfs <= result_a.verification.true_peak_target_dbfs + 0.01,
        "safe_true_peak_margin": not result_a.verification.true_peak_margin_warning,
        "closed_profile": profile_name == "default" and spec_profile_name == "default",
        "spec_contract": abs(
            result_a.spec.normalization_headroom_db
            - (result_a.spec.requested_headroom_db + result_a.spec.profile_config.playback_true_peak_margin_db)
        ) < 1e-12,
        "program_probe_headroom": worst_case_probe_true_peak_dbfs <= 0.0 + 0.01,
    }
    passed = all(strict_checks.values())
    print("fir-dsp doctor")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    for label, ok in strict_checks.items():
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
    print(f"Reference coeff hash: {result_a.verification.coeff_hash_sha256}")
    print(f"Reference request fingerprint: {result_a.verification.request_fingerprint_sha256}")
    print(f"Requested headroom: {result_a.spec.requested_headroom_db:.2f} dB")
    print(f"Normalization headroom: {result_a.spec.normalization_headroom_db:.2f} dB")
    print(f"True peak: {result_a.verification.fir_true_peak_dbfs:.3f} dBFS")
    print(f"True-peak margin: {result_a.verification.true_peak_margin_db:.3f} dB")
    print(f"Worst simulated program-material TP: {worst_case_probe_true_peak_dbfs:.3f} dBFS")
    for label, peak_dbfs in simulated_probe_peaks_dbfs.items():
        print(f"Probe {label}: {peak_dbfs:.3f} dBFS")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.doctor:
        return _run_doctor()

    if args.response is None and args.opra_query is None:
        parser.error("one of --response/--input or --opra-query is required unless --doctor is used")
    if args.fft_size is None:
        parser.error("--fft-size is required unless --doctor is used")
    if args.headroom_db is None:
        parser.error("--headroom-db is required unless --doctor is used")
    if args.out is None:
        parser.error("--out/--output is required unless --doctor is used")
    if args.window == "none":
        if args.no_window:
            parser.error("Use either --window none or --no-window, not both")
        if args.window_preset:
            parser.error("Use --window none without --window-preset")
        args.window = None
        args.no_window = True
    if args.no_window and (args.window or args.window_preset):
        parser.error("Use --no-window without --window or --window-preset")
    if args.window and args.window_preset:
        parser.error("Use either --window or --window-preset, not both")
    if args.all_pcm and args.rates:
        parser.error("Use either --rates or --all-pcm, not both")
    if args.output_dir and not (args.rates or args.all_pcm):
        parser.error("--output-dir is only valid with multi-rate export")
    if args.target and not args.validate_target:
        parser.error("--target requires --validate-target")
    if not args.validate_target and args.target_scale is not None:
        parser.error("--target-scale requires --validate-target")
    if not args.validate_target and args.target_min_freq is not None:
        parser.error("--target-min-freq requires --validate-target")
    if not args.validate_target and args.target_max_freq is not None:
        parser.error("--target-max-freq requires --validate-target")
    if args.reproducible and not args.export_json:
        parser.error("--reproducible requires --export-json")
    if args.audible_stress and not args.export_json:
        parser.error("--audible-stress requires --export-json")
    if args.validate_target and args.response is None:
        parser.error("--validate-target currently requires --response")
    if args.target_min_freq is not None and args.target_min_freq < 0:
        parser.error("--target-min-freq must be >= 0")

    args = _apply_mode_defaults(args)

    if args.true_peak is None:
        args.true_peak = True

    if args.oversample_factor is None:
        args.oversample_factor = 8
    if args.design_os is None:
        args.design_os = 1

    if args.true_peak and args.oversample_factor < 8:
        logger.warning(
            "Oversample factor %s too low for true peak, upgrading to 8",
            args.oversample_factor,
        )
        args.oversample_factor = 8

    try:
        freqs_hz = None
        values = None
        inferred_input_scale = args.input_scale
        rates = list(STANDARD_PCM_RATES) if args.all_pcm else (args.rates or [args.sample_rate])
        multi_rate = bool(args.rates or args.all_pcm)

        target_freqs_hz = None
        target_values = None
        target_scale = None
        selected_opra_product = None
        selected_opra_eq = None
        selected_opra_vendors = None

        if args.response is not None:
            freqs_hz, values = load_response(args.response)
            if freqs_hz is None and inferred_input_scale == "db":
                inferred_input_scale = "linear"
            if args.validate_target:
                target_freqs_hz, target_values, target_scale = _resolve_target_validation_source(
                    args=args,
                    response_freqs_hz=freqs_hz,
                    response_values=values,
                    inferred_input_scale=inferred_input_scale,
                )
        elif args.opra_query is not None:
            products, eqs_by_product, vendors = load_opra_jsonl(args.opra_db)
            selected_opra_product = find_product(products, args.opra_query)
            selected_opra_eq = select_eq(
                selected_opra_product,
                eqs_by_product,
                vendors,
                target=args.opra_target,
                measurement=args.opra_measurement,
            )
            selected_opra_vendors = vendors

        run_start = perf_counter()
        if args.opra_query is not None:
            assert selected_opra_eq is not None
            common_kwargs = dict(
                magnitude={"type": "eq_json", "eq": selected_opra_eq},
                fft_size=args.fft_size,
                headroom_db=args.headroom_db,
                input_scale="linear",
                true_peak=args.true_peak,
                oversample_factor=args.oversample_factor,
                window_type=args.window,
                window_beta=args.window_beta,
                window_preset=args.window_preset,
                design_oversample=args.design_os,
                return_details=True,
                profile=args.profile,
            )
            results = (
                generate_fir_multi_rate(rates=rates, target_sample_rate=args.target_sample_rate, **common_kwargs)
                if multi_rate or args.target_sample_rate is not None
                else {args.sample_rate: generate_fir_pipeline(sample_rate=args.sample_rate, **common_kwargs)}
            )
        else:
            if args.target_sample_rate is not None:
                parser.error("--target-sample-rate is only valid with --opra-query")
            assert values is not None
            common_kwargs = dict(
                magnitude=values,
                freqs_hz=freqs_hz,
                fft_size=args.fft_size,
                headroom_db=args.headroom_db,
                input_scale=inferred_input_scale,
                true_peak=args.true_peak,
                oversample_factor=args.oversample_factor,
                window_type=args.window,
                window_beta=args.window_beta,
                window_preset=args.window_preset,
                design_oversample=args.design_os,
                return_details=True,
                profile=args.profile,
            )
            results = generate_fir_multi_rate(rates=rates, **common_kwargs) if multi_rate else {args.sample_rate: generate_fir_pipeline(sample_rate=args.sample_rate, **common_kwargs)}

        total_elapsed_ms = (perf_counter() - run_start) * 1000.0
        per_rate_elapsed_ms = total_elapsed_ms / max(len(results), 1)
        cross_rate_payload = None

        if selected_opra_eq is not None:
            adjusted_results = {}
            for sr, result in results.items():
                desired_margin_db = max(
                    float(result.spec.profile_config.playback_true_peak_margin_db),
                    float(result.verification.true_peak_min_safe_margin_db),
                )
                adjusted_result, attenuation_db = _apply_opra_post_scale(result, desired_margin_db)
                if attenuation_db > 0.0:
                    logger.info(
                        "Applied OPRA-only post true-peak safety scaling: %.3f dB attenuation at %s Hz",
                        attenuation_db,
                        sr,
                    )
                adjusted_results[sr] = adjusted_result
            results = adjusted_results

        cross_rate_summary = cross_rate_consistency(results)
        if cross_rate_summary is not None:
            cross_rate_payload = asdict(cross_rate_summary)
            rate_target_hashes = {
                str(sr): result.verification.target_hash_sha256
                for sr, result in results.items()
            }
            targets_derived_from_master = bool(selected_opra_eq is not None and multi_rate)
            cross_rate_payload["rate_target_hashes_derived_from_master"] = rate_target_hashes
            cross_rate_payload["targets_derived_from_canonical_master"] = targets_derived_from_master
            if targets_derived_from_master:
                cross_rate_payload["target_projection"] = _target_projection_metadata(
                    rates=tuple(sorted(int(sr) for sr in results)),
                    target_sample_rate=args.target_sample_rate,
                    interpolation_mode="log",
                )
            cross_rate_payload["canonical_master_target_hash_sha256"] = (
                request_fingerprint(
                    {
                        "target_policy": "canonical_union_fft_bin_centers",
                        "target_sample_rate": args.target_sample_rate,
                        "target_projection": _target_projection_metadata(
                            rates=tuple(sorted(int(sr) for sr in results)),
                            target_sample_rate=args.target_sample_rate,
                            interpolation_mode="log",
                        ),
                        "eq_id": selected_opra_eq.get("id") if selected_opra_eq is not None else None,
                        "fft_size": int(args.fft_size),
                        "rates": sorted(int(sr) for sr in results),
                        "rate_target_hashes": rate_target_hashes,
                    }
                )
                if targets_derived_from_master
                else None
            )
            if args.export_json:
                cross_rate_root = (
                    args.output_dir
                    if multi_rate and args.output_dir is not None
                    else args.out.parent
                )
                cross_rate_root.mkdir(parents=True, exist_ok=True)
                (cross_rate_root / f"{args.out.stem}_cross_rate_consistency.json").write_text(
                    metadata_json_dumps(cross_rate_payload, reproducible=bool(args.reproducible)),
                    encoding="utf-8",
                )
                (cross_rate_root / f"{args.out.stem}_release_verdict.json").write_text(
                    metadata_json_dumps(
                        _build_release_verdict(results, cross_rate_payload),
                        reproducible=bool(args.reproducible),
                    ),
                    encoding="utf-8",
                )

        for sr, result in results.items():
            out_path = args.out if not multi_rate else (args.output_dir if args.output_dir is not None else args.out.parent) / f"{args.out.stem}_{sr}{args.out.suffix or '.txt'}"
            target_validation = None
            if args.validate_target:
                target_validation = validate_fir_against_target(
                    result.fir_final,
                    sample_rate=sr,
                    target_freqs_hz=target_freqs_hz,
                    target_values=target_values,
                    target_scale=target_scale,
                    n_fft=result.fft_size,
                    min_freq_hz=20.0 if args.target_min_freq is None else args.target_min_freq,
                    max_freq_hz=args.target_max_freq,
                )

            _write_outputs(
                out_path,
                result,
                export_json=args.export_json,
                plot=args.plot,
                elapsed_ms=per_rate_elapsed_ms,
                mode=args.mode,
                target_validation=target_validation,
                cross_rate_consistency_payload=cross_rate_payload,
                include_stress_probes=args.audible_stress,
                reproducible=bool(args.reproducible),
            )

            print(f"[OK] {sr} Hz -> {out_path}")
            print(f"Profile: {getattr(result.profile, 'name', result.profile)}")
            print(f"Length: {len(result.fir_final)} samples")
            print("Phase mode: minimum")
            print(f"Peak latency: {result.latency.peak_latency_ms:.3f} ms")
            print(f"Abs centroid: {result.latency.abs_centroid_ms:.3f} ms")
            print(f"Energy centroid: {result.latency.energy_centroid_ms:.3f} ms")
            print(f"Nominal linear-phase latency: {result.latency.nominal_linear_phase_latency_ms:.3f} ms")
            print(f"Max response error: {result.error.max_abs_error_db:.3f} dB")
            print(f"95th percentile error: {result.error.p95_abs_error_db:.3f} dB")
            if target_validation is not None:
                print("--- Target validation ---")
                print(f"Band: {target_validation.min_freq_hz:.1f} Hz - {target_validation.max_freq_hz:.1f} Hz")
                print(f"Bins compared: {target_validation.bins_compared}")
                print(f"Target max error: {target_validation.max_abs_error_db:.3f} dB @ {target_validation.max_error_freq_hz:.1f} Hz")
                print(f"Target mean abs error: {target_validation.mean_abs_error_db:.3f} dB")
                print(f"Target RMS error: {target_validation.rms_error_db:.3f} dB")
                print(f"Target p95 error: {target_validation.p95_abs_error_db:.3f} dB")
                if target_validation.listening_band_summary is not None:
                    listening = target_validation.listening_band_summary
                    print(
                        f"Target listening band ({listening.min_freq_hz:.1f}-{listening.max_freq_hz:.1f} Hz): "
                        f"max {listening.max_abs_error_db:.3f} dB, "
                        f"p95 {listening.p95_abs_error_db:.3f} dB"
                    )
                for band in target_validation.band_summaries:
                    print(
                        f"Target band {band.label}: "
                        f"{band.min_freq_hz:.1f}-{band.max_freq_hz:.1f} Hz, "
                        f"max {band.max_abs_error_db:.3f} dB, "
                        f"p95 {band.p95_abs_error_db:.3f} dB"
                    )
            print(f"Requested headroom: {result.spec.requested_headroom_db:.3f} dB")
            print(f"Normalization headroom: {result.spec.normalization_headroom_db:.3f} dB")
            print(f"Sample peak: {result.verification.fir_peak_dbfs:.3f} dBFS")
            print(f"True peak: {result.verification.fir_true_peak_dbfs:.3f} dBFS")
            print(f"True-peak target: {result.verification.true_peak_target_dbfs:.3f} dBFS")
            print(f"True-peak margin: {result.verification.true_peak_margin_db:.3f} dB")
            print(f"Playback safety margin (profile): {result.spec.profile_config.playback_true_peak_margin_db:.3f} dB")
            print(f"Target SHA256: {result.verification.target_hash_sha256}")
            print(f"Coeff SHA256: {result.verification.coeff_hash_sha256}")
            print(f"FIR ID (short): {result.verification.coeff_hash_sha256[:16]}")
            print(f"Request fingerprint: {result.verification.request_fingerprint_sha256}")
            print(f"Deterministic artifact in current runtime: YES (hash={result.verification.coeff_hash_sha256[:8]})")
            if result.verification.true_peak_margin_warning:
                print(
                    "WARNING: Low true-peak margin "
                    f"({result.verification.true_peak_margin_db:.3f} dB < "
                    f"{result.verification.true_peak_min_safe_margin_db:.3f} dB safe threshold)"
                )
            logger.info("Completed export for %s Hz in %.2f ms", sr, per_rate_elapsed_ms)

        if selected_opra_eq is not None and selected_opra_product is not None and selected_opra_vendors is not None:
            sidecar_root = (
                args.output_dir
                if multi_rate and args.output_dir is not None
                else args.out.parent
            )
            _write_opra_sidecars(
                sidecar_root,
                args.out.stem,
                product=selected_opra_product,
                eq=selected_opra_eq,
                vendors=selected_opra_vendors,
                db_source=str(args.opra_db),
            )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
