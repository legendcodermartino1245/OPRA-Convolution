from __future__ import annotations

import argparse
import hashlib
import json
import re
import platform
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy.io.wavfile import write

from .api import _target_projection_metadata, generate_fir_multi_rate, generate_fir_pipeline
from .artifact_metrics import (
    artifact_contract,
    canonical_wav_array,
    cross_rate_consistency,
    export_parity,
    float_safety,
    reconstruction_error,
)
from .analysis_ext import analyze_ext
from .analyze import analyze_fir_quality
from .benchmark_analyze import AnalyzeBenchmarkSummary
from .cli import _apply_opra_post_scale, save_fir
from .core import (
    build_fft_freq_grid,
    coeff_hash_sha256,
    db_to_linear,
    interpolate_log_frequency_response,
    request_fingerprint,
    summarize_response_error,
)
from .eq_to_magnitude_native import eq_json_to_native_magnitude
from .models import PipelineResult
from .opra_selector import (
    DEFAULT_OPRA_DB_URL,
    build_ui_attribution,
    eq_to_txt,
    extract_measurement,
    find_product,
    load_opra_jsonl,
    parse_eq_id,
    select_eq,
    validate_eq,
)
from .profiles import DEFAULT_PROFILE, PipelineProfile, resolve_profile
from .report import metadata_json_dumps, pipeline_metadata, write_pipeline_report
from .validation import (
    ensure_choice,
    ensure_non_negative_array,
    ensure_non_negative_float,
    ensure_positive_int,
    ensure_positive_sample_rate,
    ensure_strictly_increasing_freqs,
)

DEFAULT_RATES: tuple[int, ...] = (44_100, 48_000, 88_200, 96_000, 176_400, 192_000)
OPRA_NOTICE_FILENAME = "NOTICE_OPRA.txt"
OPRA_ATTRIBUTION_FILENAME = "ATTRIBUTION_OPRA.json"
OPRA_REPO_URL = "https://github.com/opra-project/OPRA"
DEFAULT_PROFILE_ROOT = "fir_profiles"
DEFAULT_TIER3_PROFILE_ROOT = "fir_profiles_tier3"
DEFAULT_FIR_DIRNAME = "fir_pack"
DEFAULT_WAV_DIRNAME = "wav_pack"
DEFAULT_PROFILE_NAME = "default"
DEFAULT_WINDOW_TYPE: str | None = None
DEFAULT_WINDOW_PRESET: str | None = None
STRICT_OPRA_FFT_SIZE = 131072
STRICT_OPRA_HEADROOM_DB = 9.6
STRICT_OPRA_OVERSAMPLE_FACTOR = 8
STRICT_OPRA_DESIGN_OVERSAMPLE = 1
STRICT_OPRA_BENCHMARK_WARMUP = 0
STRICT_OPRA_BENCHMARK_REPEAT = 1
TRUE_PEAK_MARGIN_TOLERANCE_DB = 1e-6
TIER3_PROJECTION_STAGE = "post_chain_compensated"
TIER3_CHAIN_MODEL_TYPE = "resampler_compensated"
TIER3_MIN_CHAIN_MAGNITUDE = 1e-6
TIER3_MIN_CHAIN_MAGNITUDE_DB = -120.0
TIER3_INVERSE_GAIN_WARNING_DB = 20.0
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _build_opra_notice(product_id: str, db_source: str) -> str:
    return "\n".join(
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
            f"Selected product: {product_id}",
        ]
    )


def _write_opra_attribution(
    output_dir: Path,
    *,
    product: dict,
    eq: dict,
    vendors: dict,
    db_source: str,
) -> None:
    attribution = build_ui_attribution(product, eq, vendors)
    payload = {
        "source": "OPRA",
        "repository": OPRA_REPO_URL,
        "database_source": db_source,
        **attribution,
    }
    (output_dir / OPRA_ATTRIBUTION_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("_") or "unnamed"


@lru_cache(maxsize=16_384)
def _profile_directory_parts(eq_id: str) -> tuple[str, str, str]:
    vendor_id, product_key, target = parse_eq_id(eq_id)
    _, product_slug = product_key.split("::", 1)
    return _safe_name(vendor_id), _safe_name(product_slug), _safe_name(target)


def _build_profile_directory_map(
    products: dict[str, dict[str, Any]],
    eqs_by_product: dict[str, list[dict[str, Any]]],
    vendors: dict[str, dict[str, Any]],
    *,
    root_dir: Path,
    ) -> dict[str, Any]:
    root_dir = Path(root_dir)
    profiles: dict[str, dict[str, Any]] = {}
    auto_selected_count = 0
    manual_only_count = 0

    for product_id, product in sorted(products.items()):
        eqs = eqs_by_product.get(product_id, [])
        if not eqs:
            continue

        try:
            selected_eq = select_eq(product, eqs_by_product, vendors)
            default_selected_eq_id = selected_eq["id"]
            selection_mode = "auto"
        except ValueError:
            default_selected_eq_id = None
            selection_mode = "manual_required"

        for eq in sorted(eqs, key=lambda item: item["id"]):
            try:
                validate_eq(eq, product, vendors)
            except ValueError:
                continue

            vendor_id, _, target = parse_eq_id(eq["id"])
            vendor_dir, product_dir, target_dir = _profile_directory_parts(eq["id"])
            rel_dir = Path(vendor_dir) / product_dir / target_dir
            if eq["id"] in profiles:
                raise ValueError(f"Duplicate EQ id in profile directory map: {eq['id']}")

            is_default = eq["id"] == default_selected_eq_id
            if is_default:
                auto_selected_count += 1
            else:
                manual_only_count += 1

            profiles[eq["id"]] = {
                "eq_id": eq["id"],
                "product_id": product_id,
                "vendor_id": vendor_id,
                "target": target,
                "measurement": extract_measurement(eq),
                "path_parts": {
                    "vendor": vendor_dir,
                    "product": product_dir,
                    "target": target_dir,
                },
                "directory_relpath": rel_dir.as_posix(),
                "default_selected": is_default,
                "selection_mode": selection_mode,
            }

    return {
        "root_directory": root_dir.as_posix(),
        "profile_count": len(profiles),
        "auto_selected_count": auto_selected_count,
        "manual_only_count": manual_only_count,
        "profiles": profiles,
    }


def _iter_profile_map_pack_jobs(
    products: dict[str, dict[str, Any]],
    eqs_by_product: dict[str, list[dict[str, Any]]],
    vendors: dict[str, dict[str, Any]],
    *,
    root_dir: Path,
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any], Path]]:
    profile_map = _build_profile_directory_map(
        products,
        eqs_by_product,
        vendors,
        root_dir=root_dir,
    )
    eq_by_id = {
        eq["id"]: eq
        for eqs in eqs_by_product.values()
        for eq in eqs
    }

    profiles = profile_map["profiles"].values()
    for profile in sorted(profiles, key=lambda item: item["directory_relpath"]):
        product_id = str(profile["product_id"])
        eq_id = str(profile["eq_id"])
        yield (
            product_id,
            products[product_id],
            eq_by_id[eq_id],
            Path(root_dir) / Path(str(profile["directory_relpath"])),
        )


def _build_default_pack_root(root_dir: Path, eq_id: str, *, product_id: str | None = None) -> Path:
    try:
        vendor_dir, product_dir, target_dir = _profile_directory_parts(eq_id)
        return Path(root_dir) / vendor_dir / product_dir / target_dir
    except ValueError:
        fallback = product_id if product_id is not None else eq_id
        return Path(root_dir) / _safe_name(fallback)


def _resolve_single_pack_root(args: argparse.Namespace, eq_id: str, *, product_id: str | None = None) -> Path:
    if args.output:
        return Path(args.output)
    return _build_default_pack_root(Path(args.root), eq_id, product_id=product_id)


def _resolve_multi_pack_root(base_root: Path, eq_id: str, *, product_id: str | None = None) -> Path:
    return _build_default_pack_root(base_root, eq_id, product_id=product_id)


def _arg_or_default(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _resolve_pack_shard_args(args: argparse.Namespace) -> tuple[int, int]:
    shard_count = ensure_positive_int("shard_count", _arg_or_default(args, "shard_count", 1))
    shard_index = _ensure_non_negative_int("shard_index", _arg_or_default(args, "shard_index", 0))
    if shard_index >= shard_count:
        raise ValueError("shard_index must be less than shard_count")
    return shard_count, shard_index


def _resolve_pack_window_args(args: argparse.Namespace) -> tuple[str | None, str | None]:
    window_type = _arg_or_default(args, "window", DEFAULT_WINDOW_TYPE)
    window_preset = _arg_or_default(args, "window_preset", DEFAULT_WINDOW_PRESET)
    if window_type == "none":
        window_type = None
    if window_type is not None and window_preset is not None:
        raise ValueError("Use either --window or --window-preset, not both")
    return window_type, window_preset


def _enforce_default_profile(profile: str | PipelineProfile) -> str:
    if isinstance(profile, PipelineProfile):
        if profile is not DEFAULT_PROFILE:
            raise RuntimeError("OPRA release export is locked to the registered default profile")
        return DEFAULT_PROFILE_NAME
    resolved = resolve_profile(profile)
    if resolved is not DEFAULT_PROFILE:
        raise RuntimeError("OPRA release export is locked to the registered default profile")
    return DEFAULT_PROFILE_NAME


def _validate_strict_opra_release_config(
    *,
    rates: tuple[int, ...],
    fft_size: int,
    headroom_db: float,
    profile: str | PipelineProfile,
    oversample_factor: int,
    design_oversample: int,
    window_type: str | None,
    window_preset: str | None,
    target_sample_rate: int | None,
    benchmark_warmup: int,
    benchmark_repeat: int,
    fir_dirname: str,
    wav_dirname: str,
    keep_existing_artifacts: bool,
) -> str:
    locked_profile = _enforce_default_profile(profile)
    if tuple(int(rate) for rate in rates) != tuple(DEFAULT_RATES):
        raise RuntimeError("OPRA release export uses locked default sample rates")
    if int(fft_size) != int(STRICT_OPRA_FFT_SIZE):
        raise RuntimeError("OPRA release export uses locked fft_size=131072")
    if abs(float(headroom_db) - float(STRICT_OPRA_HEADROOM_DB)) > 1e-12:
        raise RuntimeError("OPRA release export uses locked headroom=9.6 dB")
    if int(oversample_factor) != int(STRICT_OPRA_OVERSAMPLE_FACTOR):
        raise RuntimeError("OPRA release export uses locked oversample_factor=8")
    if int(design_oversample) != int(STRICT_OPRA_DESIGN_OVERSAMPLE):
        raise RuntimeError("OPRA release export uses locked design_oversample=1")
    if window_type is not None or window_preset is not None:
        raise RuntimeError("OPRA release export forbids windowing")
    if target_sample_rate is not None:
        ensure_positive_sample_rate(target_sample_rate)
    if int(benchmark_warmup) != int(STRICT_OPRA_BENCHMARK_WARMUP):
        raise RuntimeError("OPRA release export uses locked benchmark warmup")
    if int(benchmark_repeat) != int(STRICT_OPRA_BENCHMARK_REPEAT):
        raise RuntimeError("OPRA release export uses locked benchmark repeat")
    if str(fir_dirname) != DEFAULT_FIR_DIRNAME:
        raise RuntimeError("OPRA release export uses locked FIR directory name")
    if str(wav_dirname) != DEFAULT_WAV_DIRNAME:
        raise RuntimeError("OPRA release export uses locked WAV directory name")
    if bool(keep_existing_artifacts):
        raise RuntimeError("OPRA release export always rebuilds artifacts from a clean pack directory")
    return locked_profile


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _write_directory_zip(
    source_dir: Path,
    zip_path: Path,
    *,
    file_paths: Iterable[Path] | None = None,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(zip_path)
    source_dir = Path(source_dir)
    if file_paths is None:
        zip_file_paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    else:
        unique_file_paths = dict.fromkeys(Path(path) for path in file_paths)
        zip_file_paths = sorted(
            unique_file_paths,
            key=lambda path: Path(path).relative_to(source_dir).as_posix(),
        )
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for file_path in zip_file_paths:
            relative_path = file_path.relative_to(source_dir).as_posix()
            info = ZipInfo(relative_path)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            with file_path.open("rb") as handle:
                archive.writestr(info, handle.read())


def _validate_pipeline_payload(payload: dict[str, Any], *, sample_rate: int, profile: str | PipelineProfile) -> None:
    if int(payload["sample_rate"]) != int(sample_rate):
        raise RuntimeError(f"Pipeline report sample rate mismatch for {sample_rate} Hz")

    expected_profile = _enforce_default_profile(profile)
    if str(payload["profile"]) != str(expected_profile):
        raise RuntimeError(f"Pipeline profile mismatch for {sample_rate} Hz")
    if int(payload["fft_size"]) != int(STRICT_OPRA_FFT_SIZE):
        raise RuntimeError(f"Pipeline FFT size mismatch for {sample_rate} Hz")
    if abs(float(payload["headroom_db"]) - float(STRICT_OPRA_HEADROOM_DB)) > 1e-12:
        raise RuntimeError(f"Pipeline headroom mismatch for {sample_rate} Hz")
    if int(payload["oversample_factor"]) != int(STRICT_OPRA_OVERSAMPLE_FACTOR):
        raise RuntimeError(f"Pipeline oversample factor mismatch for {sample_rate} Hz")
    if int(payload["design_oversample"]) != int(STRICT_OPRA_DESIGN_OVERSAMPLE):
        raise RuntimeError(f"Pipeline design oversample mismatch for {sample_rate} Hz")
    if payload["window_type"] is not None or payload["window_preset"] is not None:
        raise RuntimeError(f"Pipeline windowing must remain disabled for {sample_rate} Hz")

    verification = payload["verification"]

    if not bool(payload["true_peak"]):
        raise RuntimeError(f"true_peak must remain enabled for {sample_rate} Hz")

    true_peak_margin_db = float(verification["true_peak_margin_db"])
    true_peak_min_safe_margin_db = float(verification["true_peak_min_safe_margin_db"])

    true_peak_margin_below_safe = (
        true_peak_margin_db + TRUE_PEAK_MARGIN_TOLERANCE_DB
        < true_peak_min_safe_margin_db
    )

    if true_peak_margin_below_safe:
        raise RuntimeError(f"True-peak margin below safe threshold for {sample_rate} Hz")

    if bool(verification["true_peak_margin_warning"]) != true_peak_margin_below_safe:
        raise RuntimeError(f"True-peak warning mismatch for {sample_rate} Hz")

    if float(payload["error"]["max_abs_error_db"]) > 0.1:
        raise RuntimeError(f"Max response error exceeds strict release threshold for {sample_rate} Hz")

    if str(payload["audible_target_verdict"]) == "review":
        raise RuntimeError(f"Audible target verdict requires review for {sample_rate} Hz")

    stress_summary = payload.get("program_material_stress_summary")

    if stress_summary is None:
        raise RuntimeError(f"Missing program material stress summary for {sample_rate} Hz")

    if not bool(stress_summary["passes_target"]):
        raise RuntimeError(f"Stress summary failed playback target for {sample_rate} Hz")


def _ensure_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be >= 0")
    return checked


def _write_pipeline_metadata_payload(
    pipeline_json_path: Path,
    *,
    result: PipelineResult,
    include_stress_probes: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    payload = pipeline_metadata(
        result,
        elapsed_ms=None,
        include_stress_probes=include_stress_probes,
    )
    payload.update(extra)
    pipeline_json_path.write_text(
        metadata_json_dumps(payload, reproducible=True),
        encoding="utf-8",
    )
    return payload


def _benchmark_and_write_analysis_payloads(
    *,
    benchmark_path: Path,
    analysis_path: Path,
    fir_path: Path,
    fir: np.ndarray,
    sample_rate: int,
    oversample_factor: int,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    warmup = _ensure_non_negative_int("warmup", warmup)
    repeat = ensure_positive_int("repeat", repeat)
    fir = np.asarray(fir, dtype=np.float64)

    for _ in range(warmup):
        analyze_fir_quality(fir, sample_rate=sample_rate, oversample_factor=oversample_factor)

    timings_ms: list[float] = []
    summary = None
    for _ in range(repeat):
        start = perf_counter()
        summary = analyze_fir_quality(fir, sample_rate=sample_rate, oversample_factor=oversample_factor)
        timings_ms.append((perf_counter() - start) * 1000.0)

    if summary is None:
        raise RuntimeError("Benchmark did not produce an analysis summary")

    benchmark_payload = asdict(
        AnalyzeBenchmarkSummary(
            fir_path=str(fir_path),
            sample_rate=sample_rate,
            taps=int(fir.size),
            oversample_factor=oversample_factor,
            warmup_runs=warmup,
            measured_runs=len(timings_ms),
            min_ms=float(min(timings_ms)),
            median_ms=float(statistics.median(timings_ms)),
            mean_ms=float(statistics.fmean(timings_ms)),
            max_ms=float(max(timings_ms)),
            stdev_ms=float(statistics.pstdev(timings_ms)) if len(timings_ms) > 1 else 0.0,
            python_version=platform.python_version(),
            platform=platform.platform(),
        )
    )
    benchmark_path.write_text(
        metadata_json_dumps(benchmark_payload, reproducible=True),
        encoding="utf-8",
    )

    extended_results = analyze_ext(fir, sample_rate, mode="strict")
    for result in extended_results:
        if str(result["status"]) in {"FAIL", "WARN"}:
            raise RuntimeError(
                f"Strict extended validation reported {result['status']} for metric "
                f"'{result['metric']}' at {sample_rate} Hz"
            )
    analysis_payload: dict[str, Any] = asdict(summary)
    analysis_payload["extended_validation"] = {
        "mode": "strict",
        "results": extended_results,
    }
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(metadata_json_dumps(analysis_payload, reproducible=True), encoding="utf-8")
    return benchmark_payload, analysis_payload


def _write_float32_wav_and_hash(output_file: Path, sample_rate: int, wav_samples: np.ndarray) -> str:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    wav_buffer = BytesIO()
    write(wav_buffer, sample_rate, wav_samples)
    wav_bytes = wav_buffer.getvalue()
    output_file.write_bytes(wav_bytes)
    return hashlib.sha256(wav_bytes).hexdigest()


def _write_wav_payload_and_hash_from_fir(
    *,
    fir: np.ndarray,
    output_file: Path,
    sample_rate: int,
    precomputed_export_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, object], str]:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    processed = np.asarray(fir, dtype=np.float64)
    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    wav_samples = canonical_wav_array(processed)
    wav_sha256 = _write_float32_wav_and_hash(output_file, sample_rate, wav_samples)

    if precomputed_export_metadata is not None:
        contract_payload = precomputed_export_metadata["artifact_contract"]
        parity_payload = precomputed_export_metadata["export_parity"]
        reconstruction_payload = precomputed_export_metadata["exported_wav_reconstruction_error"]
        float_safety_payload = precomputed_export_metadata["float_safety"]
    else:
        wav_roundtrip = np.asarray(wav_samples, dtype=np.float64)
        contract_payload = artifact_contract(wav_samples, sample_rate).__dict__
        parity_payload = export_parity(processed, wav_roundtrip).__dict__
        reconstruction_payload = reconstruction_error(
            processed,
            wav_roundtrip,
            sample_rate=sample_rate,
            fft_size=max(processed.size, 4096),
        ).__dict__
        float_safety_payload = float_safety(wav_samples).__dict__

    return (
        {
            "sample_rate": sample_rate,
            "normalized": False,
            "peak_before": peak,
            "peak_written": peak,
            "sample_format": "float32",
            "artifact_contract": contract_payload,
            "export_parity": parity_payload,
            "exported_wav_reconstruction_error": reconstruction_payload,
            "float_safety": float_safety_payload,
        },
        wav_sha256,
    )


def _write_wav_payload_from_fir(
    *,
    fir: np.ndarray,
    output_file: Path,
    sample_rate: int,
    precomputed_export_metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    payload, _wav_sha256 = _write_wav_payload_and_hash_from_fir(
        fir=fir,
        output_file=output_file,
        sample_rate=sample_rate,
        precomputed_export_metadata=precomputed_export_metadata,
    )
    return payload


def _write_pack_manifest(
    manifest_path: Path,
    *,
    pack_root: Path,
    fir_output_dir: Path,
    wav_output_dir: Path,
    product_id: str,
    eq_id: str,
    target: str,
    db_source: str,
    rates: tuple[int, ...],
    fft_size: int,
    headroom_db: float,
    profile: str | PipelineProfile,
    oversample_factor: int,
    design_oversample: int,
    window_type: str | None,
    window_preset: str | None,
    target_sample_rate: int | None,
    benchmark_warmup: int,
    benchmark_repeat: int,
    keep_existing_artifacts: bool,
    release_verdict: dict[str, Any],
    cross_rate_payload: dict[str, Any] | None,
    artifacts: list[dict[str, Any]],
) -> None:
    payload = {
        "pack_root": str(pack_root),
        "fir_output_dir": str(fir_output_dir),
        "wav_output_dir": str(wav_output_dir),
        "product_id": product_id,
        "eq_id": eq_id,
        "target": target,
        "opra_db": db_source,
        "fft_size": int(fft_size),
        "headroom_db": float(headroom_db),
        "profile": profile,
        "oversample_factor": int(oversample_factor),
        "design_oversample": int(design_oversample),
        "window_type": window_type,
        "window_preset": window_preset,
        "target_sample_rate": None if target_sample_rate is None else int(target_sample_rate),
        "target_projection": _target_projection_metadata(
            rates=rates,
            target_sample_rate=target_sample_rate,
        ),
        "benchmark_warmup": int(benchmark_warmup),
        "benchmark_repeat": int(benchmark_repeat),
        "keep_existing_artifacts": bool(keep_existing_artifacts),
        "rates": list(rates),
        "release_verdict": release_verdict,
        "cross_rate_report_sha256": _sha256_file(fir_output_dir / "fir_cross_rate_consistency.json")
        if cross_rate_payload is not None
        else None,
        "artifacts": artifacts,
    }
    manifest_path.write_text(metadata_json_dumps(payload, reproducible=True), encoding="utf-8")


def _prepare_pack_directories(
    pack_root: Path,
    *,
    fir_dirname: str,
    wav_dirname: str,
) -> tuple[Path, Path]:
    fir_output_dir = pack_root / fir_dirname
    wav_output_dir = pack_root / wav_dirname
    fir_output_dir.mkdir(parents=True, exist_ok=True)
    wav_output_dir.mkdir(parents=True, exist_ok=True)
    return fir_output_dir, wav_output_dir


def _clean_existing_pack_artifacts(
    fir_output_dir: Path,
    wav_output_dir: Path,
    pack_root: Path,
    rates: tuple[int, ...],
) -> None:
    for sample_rate in rates:
        for path in (
            fir_output_dir / f"fir_{sample_rate}.txt",
            fir_output_dir / f"fir_{sample_rate}.json",
            fir_output_dir / f"fir_{sample_rate}_benchmark.json",
            fir_output_dir / f"fir_{sample_rate}_analysis.json",
            wav_output_dir / f"fir_{sample_rate}.wav",
        ):
            _remove_if_exists(path)
    for path in (
        fir_output_dir / "selected_eq.txt",
        fir_output_dir / OPRA_NOTICE_FILENAME,
        fir_output_dir / OPRA_ATTRIBUTION_FILENAME,
        fir_output_dir / "fir_cross_rate_consistency.json",
        fir_output_dir / "fir_release_verdict.json",
        fir_output_dir / "build_manifest.json",
        pack_root / "wav_pack.zip",
        pack_root / "proof_files.zip",
    ):
        _remove_if_exists(path)


def _load_tier3_chain_response(path: Path, *, scale: str) -> tuple[np.ndarray, np.ndarray]:
    scale = ensure_choice("chain_response_scale", scale, {"linear", "db"})
    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Tier 3 chain response file: {path}") from exc
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Tier 3 chain response must be a two-column file: frequency_hz magnitude")
    freqs_hz = ensure_strictly_increasing_freqs(data[:, 0])
    if float(freqs_hz[0]) != 0.0:
        raise ValueError("Tier 3 chain response must start at 0 Hz to avoid low-frequency extrapolation")
    values = data[:, 1]
    magnitudes = db_to_linear(values) if scale == "db" else ensure_non_negative_array("chain_response", values)
    if np.any(magnitudes <= 0.0):
        raise ValueError("Tier 3 chain response magnitudes must be strictly positive")
    if np.any(magnitudes < TIER3_MIN_CHAIN_MAGNITUDE):
        raise ValueError(
            "Tier 3 chain response contains magnitudes below "
            f"{TIER3_MIN_CHAIN_MAGNITUDE_DB:.0f} dB; refusing unstable inverse compensation"
        )
    return freqs_hz, np.asarray(magnitudes, dtype=np.float64)


def _project_tier3_chain_response(
    *,
    response_freqs_hz: np.ndarray,
    response_magnitude: np.ndarray,
    source_rate: int,
    design_rate: int,
    fft_size: int,
    transition_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    source_rate = ensure_positive_sample_rate(source_rate)
    design_rate = ensure_positive_sample_rate(design_rate)
    fft_size = ensure_positive_int("fft_size", fft_size)
    transition_hz = ensure_non_negative_float("transition_hz", transition_hz)
    if fft_size < 2 or fft_size % 2 != 0:
        raise ValueError("fft_size must be an even integer >= 2")
    active_max_hz = min(float(source_rate), float(design_rate)) / 2.0
    if float(response_freqs_hz[-1]) < active_max_hz:
        raise ValueError(
            "Tier 3 chain response must extend to the active source/content Nyquist "
            f"({active_max_hz} Hz), got {response_freqs_hz[-1]} Hz"
        )
    fft_freqs_hz = build_fft_freq_grid(fft_size, design_rate)
    chain_magnitude = np.ones_like(fft_freqs_hz, dtype=np.float64)
    active_mask = fft_freqs_hz <= active_max_hz
    chain_magnitude[active_mask] = interpolate_log_frequency_response(
        response_freqs_hz,
        response_magnitude,
        fft_freqs_hz[active_mask],
    )
    if transition_hz > 0.0:
        transition_start_hz = max(0.0, active_max_hz - transition_hz)
        transition_mask = (fft_freqs_hz >= transition_start_hz) & (fft_freqs_hz <= active_max_hz)
        if np.any(transition_mask) and active_max_hz > transition_start_hz:
            x = (fft_freqs_hz[transition_mask] - transition_start_hz) / (active_max_hz - transition_start_hz)
            blend_to_unity = 0.5 - (0.5 * np.cos(np.pi * x))
            chain_magnitude[transition_mask] = (
                chain_magnitude[transition_mask] * (1.0 - blend_to_unity)
                + blend_to_unity
            )
    if not np.all(np.isfinite(chain_magnitude)) or np.any(chain_magnitude <= 0.0):
        raise ValueError("Tier 3 projected chain response must be finite and strictly positive")
    return fft_freqs_hz, chain_magnitude, active_max_hz


def _tier3_active_band_mask(freqs_hz: np.ndarray, active_max_hz: float) -> np.ndarray:
    freqs = ensure_non_negative_array("freqs_hz", freqs_hz)
    active_max_hz = ensure_non_negative_float("active_max_hz", active_max_hz)
    mask = freqs <= active_max_hz
    if not np.any(mask):
        raise ValueError("Tier 3 active compensation band contains no FFT bins")
    return mask


def _tier3_chain_conditioning(
    *,
    freqs_hz: np.ndarray,
    chain_magnitude: np.ndarray,
    active_max_hz: float,
) -> dict[str, Any]:
    freqs = ensure_non_negative_array("freqs_hz", freqs_hz)
    chain = ensure_non_negative_array("chain_magnitude", chain_magnitude)
    active_mask = _tier3_active_band_mask(freqs, active_max_hz)
    active_chain = chain[active_mask]
    active_freqs = freqs[active_mask]
    min_idx = int(np.argmin(active_chain))
    min_chain_magnitude = float(active_chain[min_idx])
    max_inverse_gain_db = float(-20.0 * np.log10(max(min_chain_magnitude, TIER3_MIN_CHAIN_MAGNITUDE)))

    positive_mask = active_freqs > 0.0
    max_local_slope_db_per_octave = 0.0
    if np.count_nonzero(positive_mask) >= 2:
        positive_freqs = active_freqs[positive_mask]
        chain_db = 20.0 * np.log10(np.maximum(active_chain[positive_mask], TIER3_MIN_CHAIN_MAGNITUDE))
        octave_steps = np.diff(np.log2(positive_freqs))
        db_steps = np.diff(chain_db)
        valid_steps = octave_steps > 0.0
        if np.any(valid_steps):
            slopes = np.abs(db_steps[valid_steps] / octave_steps[valid_steps])
            max_local_slope_db_per_octave = float(np.max(slopes)) if slopes.size else 0.0

    return {
        "min_chain_magnitude": min_chain_magnitude,
        "min_chain_magnitude_freq_hz": float(active_freqs[min_idx]),
        "max_inverse_gain_db": max_inverse_gain_db,
        "inverse_gain_warning_threshold_db": float(TIER3_INVERSE_GAIN_WARNING_DB),
        "inverse_gain_warning": bool(max_inverse_gain_db > TIER3_INVERSE_GAIN_WARNING_DB),
        "max_local_slope_db_per_octave": max_local_slope_db_per_octave,
    }


def _tier3_chain_model(
    *,
    source_rate: int,
    design_rate: int,
    chain_response_path: Path,
    chain_response_scale: str,
    chain_response_magnitude: np.ndarray,
    active_max_hz: float,
    resampler_name: str,
    kernel_id: str | None,
    transition_hz: float,
    conditioning: dict[str, Any],
) -> dict[str, Any]:
    kernel_identity = kernel_id if kernel_id is not None else chain_response_path.name
    return {
        "type": TIER3_CHAIN_MODEL_TYPE,
        "resampler": {
            "source_rate": ensure_positive_sample_rate(source_rate),
            "target_rate": ensure_positive_sample_rate(design_rate),
            "method": "measured_response",
            "name": str(resampler_name),
            "kernel": "measured",
            "kernel_identity": str(kernel_identity),
            "response_file": str(chain_response_path),
            "response_file_sha256": _sha256_file(chain_response_path),
            "response_scale": ensure_choice("chain_response_scale", chain_response_scale, {"linear", "db"}),
            "projected_response_hash_sha256": coeff_hash_sha256(chain_response_magnitude),
            "active_compensation_min_hz": 0.0,
            "active_compensation_max_hz": float(active_max_hz),
            "design_nyquist_hz": float(ensure_positive_sample_rate(design_rate)) / 2.0,
            "above_active_band_policy": "unity_no_source_content",
            "minimum_allowed_chain_magnitude": float(TIER3_MIN_CHAIN_MAGNITUDE),
            "minimum_allowed_chain_magnitude_db": float(TIER3_MIN_CHAIN_MAGNITUDE_DB),
            "edge_transition_width_hz": float(ensure_non_negative_float("transition_hz", transition_hz)),
            "edge_transition_policy": "cosine_blend_to_unity" if float(transition_hz) > 0.0 else "disabled_exact_inverse",
            "conditioning": conditioning,
        },
    }


def _tier3_target_projection_metadata(
    *,
    source_rate: int,
    design_rate: int,
    target_sample_rate: int,
    chain_model: dict[str, Any],
) -> dict[str, Any]:
    metadata = _target_projection_metadata(
        rates=(ensure_positive_sample_rate(design_rate),),
        target_sample_rate=target_sample_rate,
        design_sample_rate=design_rate,
    )
    metadata["projection_stage"] = TIER3_PROJECTION_STAGE
    metadata["chain_model"] = chain_model
    metadata["compensation"] = {
        "equation": "(resampler_chain * fir) ~= requested_target",
        "fir_design_target": "requested_target_magnitude / measured_chain_magnitude",
        "source_rate": ensure_positive_sample_rate(source_rate),
        "design_sample_rate": ensure_positive_sample_rate(design_rate),
        "compensation_band_hz": {
            "min": 0.0,
            "max": float(chain_model["resampler"]["active_compensation_max_hz"]),
        },
        "outside_compensation_band_policy": str(chain_model["resampler"]["above_active_band_policy"]),
    }
    return metadata


def _tier3_pack_root(eq_id: str, *, product_id: str, source_rate: int, design_rate: int) -> Path:
    root = _build_default_pack_root(Path(DEFAULT_TIER3_PROFILE_ROOT), eq_id, product_id=product_id)
    return root / f"{ensure_positive_sample_rate(source_rate)}_to_{ensure_positive_sample_rate(design_rate)}"


def _write_tier3_pack(
    pack_root: Path,
    *,
    product: dict[str, Any],
    eq: dict[str, Any],
    vendors: dict[str, dict[str, Any]],
    db_source: str,
    source_rate: int,
    design_rate: int,
    target_sample_rate: int,
    chain_response_path: Path,
    chain_response_scale: str,
    resampler_name: str,
    kernel_id: str | None,
    transition_hz: float = 0.0,
) -> None:
    profile = _validate_strict_opra_release_config(
        rates=DEFAULT_RATES,
        fft_size=STRICT_OPRA_FFT_SIZE,
        headroom_db=STRICT_OPRA_HEADROOM_DB,
        profile=DEFAULT_PROFILE_NAME,
        oversample_factor=STRICT_OPRA_OVERSAMPLE_FACTOR,
        design_oversample=STRICT_OPRA_DESIGN_OVERSAMPLE,
        window_type=DEFAULT_WINDOW_TYPE,
        window_preset=DEFAULT_WINDOW_PRESET,
        target_sample_rate=target_sample_rate,
        benchmark_warmup=STRICT_OPRA_BENCHMARK_WARMUP,
        benchmark_repeat=STRICT_OPRA_BENCHMARK_REPEAT,
        fir_dirname=DEFAULT_FIR_DIRNAME,
        wav_dirname=DEFAULT_WAV_DIRNAME,
        keep_existing_artifacts=False,
    )
    source_rate = ensure_positive_sample_rate(source_rate)
    design_rate = ensure_positive_sample_rate(design_rate)
    target_sample_rate = ensure_positive_sample_rate(target_sample_rate)
    transition_hz = ensure_non_negative_float("transition_hz", transition_hz)
    chain_response_path = Path(chain_response_path)

    fir_output_dir, wav_output_dir = _prepare_pack_directories(
        pack_root,
        fir_dirname=DEFAULT_FIR_DIRNAME,
        wav_dirname=DEFAULT_WAV_DIRNAME,
    )
    _clean_existing_pack_artifacts(fir_output_dir, wav_output_dir, pack_root, (design_rate,))

    response_freqs_hz, response_magnitude = _load_tier3_chain_response(
        chain_response_path,
        scale=chain_response_scale,
    )
    fft_freqs_hz, chain_magnitude, active_max_hz = _project_tier3_chain_response(
        response_freqs_hz=response_freqs_hz,
        response_magnitude=response_magnitude,
        source_rate=source_rate,
        design_rate=design_rate,
        fft_size=STRICT_OPRA_FFT_SIZE,
        transition_hz=transition_hz,
    )
    active_mask = _tier3_active_band_mask(fft_freqs_hz, active_max_hz)
    requested_target = eq_json_to_native_magnitude(eq, fft_freqs_hz, target_sample_rate)
    compensated_target = np.asarray(requested_target / chain_magnitude, dtype=np.float64)
    if not np.all(np.isfinite(compensated_target)) or np.any(compensated_target <= 0.0):
        raise ValueError("Tier 3 compensated target must be finite and strictly positive")
    conditioning = _tier3_chain_conditioning(
        freqs_hz=fft_freqs_hz,
        chain_magnitude=chain_magnitude,
        active_max_hz=active_max_hz,
    )

    chain_model = _tier3_chain_model(
        source_rate=source_rate,
        design_rate=design_rate,
        chain_response_path=chain_response_path,
        chain_response_scale=chain_response_scale,
        chain_response_magnitude=chain_magnitude,
        active_max_hz=active_max_hz,
        resampler_name=resampler_name,
        kernel_id=kernel_id,
        transition_hz=transition_hz,
        conditioning=conditioning,
    )
    target_projection = _tier3_target_projection_metadata(
        source_rate=source_rate,
        design_rate=design_rate,
        target_sample_rate=target_sample_rate,
        chain_model=chain_model,
    )

    started = perf_counter()
    result = generate_fir_pipeline(
        magnitude=compensated_target,
        fft_size=STRICT_OPRA_FFT_SIZE,
        sample_rate=design_rate,
        headroom_db=STRICT_OPRA_HEADROOM_DB,
        input_scale="linear",
        true_peak=True,
        oversample_factor=STRICT_OPRA_OVERSAMPLE_FACTOR,
        window_type=DEFAULT_WINDOW_TYPE,
        window_preset=DEFAULT_WINDOW_PRESET,
        design_oversample=STRICT_OPRA_DESIGN_OVERSAMPLE,
        return_details=True,
        profile=profile,
        _raw_input_magnitude={"type": "eq_json", "eq": eq},
        _target_projection=target_projection,
    )
    result = _apply_opra_post_scale_to_results({design_rate: result})[design_rate]
    if not isinstance(result, PipelineResult):
        raise TypeError("Tier 3 release pack generation expects a detailed PipelineResult")
    elapsed_ms = (perf_counter() - started) * 1000.0

    fir_txt_path = fir_output_dir / f"fir_{design_rate}.txt"
    pipeline_json_path = fir_output_dir / f"fir_{design_rate}.json"
    benchmark_json_path = fir_output_dir / f"fir_{design_rate}_benchmark.json"
    analysis_json_path = fir_output_dir / f"fir_{design_rate}_analysis.json"
    wav_path = wav_output_dir / f"fir_{design_rate}.wav"

    save_fir(fir_txt_path, result.fir_final)
    chain_actual_magnitude = np.asarray(result.actual_magnitude * chain_magnitude, dtype=np.float64)
    chain_closure_scope = {
        "frequency_min_hz": 0.0,
        "frequency_max_hz": float(active_max_hz),
        "bin_count": int(np.count_nonzero(active_mask)),
        "policy": "active_source_content_band_only",
    }
    chain_error = summarize_response_error(
        requested_target[active_mask],
        chain_actual_magnitude[active_mask],
    )
    tier3_request_fingerprint = request_fingerprint(
        {
            "tier": 3,
            "eq_id": eq["id"],
            "source_rate": source_rate,
            "design_rate": design_rate,
            "target_sample_rate": target_sample_rate,
            "projection_stage": TIER3_PROJECTION_STAGE,
            "chain_model": chain_model,
            "chain_closure_scope": chain_closure_scope,
            "requested_target_hash_sha256": coeff_hash_sha256(requested_target),
            "compensated_target_hash_sha256": coeff_hash_sha256(compensated_target),
            "pipeline_request_fingerprint_sha256": result.verification.request_fingerprint_sha256,
        }
    )

    pipeline_payload = _write_pipeline_metadata_payload(
        pipeline_json_path,
        result=result,
        tier=3,
        projection_stage=TIER3_PROJECTION_STAGE,
        chain_model=chain_model,
        requested_target_hash_sha256=coeff_hash_sha256(requested_target),
        compensated_target_hash_sha256=coeff_hash_sha256(compensated_target),
        chain_closure_scope=chain_closure_scope,
        chain_closure_error=asdict(chain_error),
        tier3_request_fingerprint_sha256=tier3_request_fingerprint,
    )
    _validate_pipeline_payload(pipeline_payload, sample_rate=design_rate, profile=profile)
    if float(chain_error.max_abs_error_db) > 0.1:
        raise RuntimeError("Tier 3 chain closure error exceeds strict release threshold")

    wav_payload, wav_sha256 = _write_wav_payload_and_hash_from_fir(
        fir=result.fir_final,
        output_file=wav_path,
        sample_rate=design_rate,
        precomputed_export_metadata=pipeline_payload,
    )
    benchmark_payload, _analysis_payload = _benchmark_and_write_analysis_payloads(
        benchmark_path=benchmark_json_path,
        analysis_path=analysis_json_path,
        fir=result.fir_final,
        fir_path=fir_txt_path,
        sample_rate=design_rate,
        oversample_factor=STRICT_OPRA_OVERSAMPLE_FACTOR,
        warmup=STRICT_OPRA_BENCHMARK_WARMUP,
        repeat=STRICT_OPRA_BENCHMARK_REPEAT,
    )

    selected_eq_path = fir_output_dir / "selected_eq.txt"
    notice_path = fir_output_dir / OPRA_NOTICE_FILENAME
    attribution_path = fir_output_dir / OPRA_ATTRIBUTION_FILENAME
    release_verdict_path = fir_output_dir / "fir_release_verdict.json"
    manifest_path = fir_output_dir / "build_manifest.json"

    selected_eq_path.write_text(eq_to_txt(eq), encoding="utf-8")
    _write_opra_notice(fir_output_dir, product["id"], db_source)
    _write_opra_attribution(
        fir_output_dir,
        product=product,
        eq=eq,
        vendors=vendors,
        db_source=db_source,
    )

    release_verdict = {
        "export_safe": bool(result.error.max_abs_error_db <= 0.1 and chain_error.max_abs_error_db <= 0.1),
        "playback_safe": bool(not result.verification.true_peak_margin_warning),
        "cross_rate_consistent": None,
        "recommended_use": "tier3_chain_specific_release",
    }
    release_verdict_path.write_text(
        metadata_json_dumps(release_verdict, reproducible=True),
        encoding="utf-8",
    )

    _, _, target = parse_eq_id(eq["id"])
    manifest = {
        "tier": 3,
        "pack_root": str(pack_root),
        "fir_output_dir": str(fir_output_dir),
        "wav_output_dir": str(wav_output_dir),
        "product_id": product["id"],
        "eq_id": eq["id"],
        "target": target,
        "opra_db": db_source,
        "source_rate": source_rate,
        "design_sample_rate": design_rate,
        "target_sample_rate": target_sample_rate,
        "fft_size": int(STRICT_OPRA_FFT_SIZE),
        "headroom_db": float(STRICT_OPRA_HEADROOM_DB),
        "profile": profile,
        "oversample_factor": int(STRICT_OPRA_OVERSAMPLE_FACTOR),
        "design_oversample": int(STRICT_OPRA_DESIGN_OVERSAMPLE),
        "window_type": DEFAULT_WINDOW_TYPE,
        "window_preset": DEFAULT_WINDOW_PRESET,
        "projection_stage": TIER3_PROJECTION_STAGE,
        "target_projection": target_projection,
        "chain_model": chain_model,
        "chain_closure_scope": chain_closure_scope,
        "chain_closure_error": asdict(chain_error),
        "tier3_request_fingerprint_sha256": tier3_request_fingerprint,
        "release_verdict": release_verdict,
        "artifact": {
            "sample_rate": int(design_rate),
            "fir_path": str(fir_txt_path),
            "wav_path": str(wav_path),
            "pipeline_json": str(pipeline_json_path),
            "benchmark_json": str(benchmark_json_path),
            "analysis_json": str(analysis_json_path),
            "fir_sha256": _sha256_file(fir_txt_path),
            "wav_sha256": wav_sha256,
            "coeff_hash_sha256": str(result.verification.coeff_hash_sha256),
            "request_fingerprint_sha256": str(result.verification.request_fingerprint_sha256),
            "wav_export_summary": wav_payload,
        },
    }
    manifest_path.write_text(
        metadata_json_dumps(manifest, reproducible=True),
        encoding="utf-8",
    )

    _write_directory_zip(
        wav_output_dir,
        pack_root / "wav_pack.zip",
        file_paths=[wav_path],
    )
    _write_directory_zip(
        fir_output_dir,
        pack_root / "proof_files.zip",
        file_paths=[
            fir_txt_path,
            pipeline_json_path,
            benchmark_json_path,
            analysis_json_path,
            selected_eq_path,
            notice_path,
            attribution_path,
            release_verdict_path,
            manifest_path,
        ],
    )


def _write_release_rate_artifact(
    *,
    sample_rate: int,
    result: PipelineResult,
    rates: tuple[int, ...],
    target_sample_rate: int | None,
    fir_output_dir: Path,
    wav_output_dir: Path,
    profile: str | PipelineProfile,
    oversample_factor: int,
    benchmark_warmup: int,
    benchmark_repeat: int,
) -> dict[str, Any]:
    if not isinstance(result, PipelineResult):
        raise TypeError("Release pack generation expects detailed PipelineResult values")

    fir_txt_path = fir_output_dir / f"fir_{sample_rate}.txt"
    pipeline_json_path = fir_output_dir / f"fir_{sample_rate}.json"
    benchmark_json_path = fir_output_dir / f"fir_{sample_rate}_benchmark.json"
    analysis_json_path = fir_output_dir / f"fir_{sample_rate}_analysis.json"
    wav_path = wav_output_dir / f"fir_{sample_rate}.wav"

    save_fir(fir_txt_path, result.fir_final)
    pipeline_payload = _write_pipeline_metadata_payload(
        pipeline_json_path,
        result=result,
        include_stress_probes=True,
    )
    _validate_pipeline_payload(pipeline_payload, sample_rate=sample_rate, profile=profile)

    wav_payload, wav_sha256 = _write_wav_payload_and_hash_from_fir(
        fir=result.fir_final,
        output_file=wav_path,
        sample_rate=sample_rate,
        precomputed_export_metadata=pipeline_payload,
    )
    benchmark_payload, analysis_payload = _benchmark_and_write_analysis_payloads(
        benchmark_path=benchmark_json_path,
        analysis_path=analysis_json_path,
        fir_path=fir_txt_path,
        fir=result.fir_final,
        sample_rate=sample_rate,
        oversample_factor=oversample_factor,
        warmup=benchmark_warmup,
        repeat=benchmark_repeat,
    )
    if int(benchmark_payload["sample_rate"]) != sample_rate:
        raise RuntimeError(f"Benchmark sample rate mismatch for {sample_rate} Hz")
    if int(benchmark_payload["measured_runs"]) != benchmark_repeat:
        raise RuntimeError(f"Benchmark repeat count mismatch for {sample_rate} Hz")
    if float(benchmark_payload["min_ms"]) < 0.0:
        raise RuntimeError(f"Benchmark min_ms invalid for {sample_rate} Hz")
    if int(analysis_payload["sample_rate"]) != sample_rate:
        raise RuntimeError(f"Analysis sample rate mismatch for {sample_rate} Hz")

    return {
        "sample_rate": int(sample_rate),
        "design_sample_rate": int(sample_rate),
        "target_projection": _target_projection_metadata(
            rates=rates,
            target_sample_rate=target_sample_rate,
            design_sample_rate=sample_rate,
        ),
        "fir_path": str(fir_txt_path),
        "wav_path": str(wav_path),
        "pipeline_json": str(pipeline_json_path),
        "benchmark_json": str(benchmark_json_path),
        "analysis_json": str(analysis_json_path),
        "fir_sha256": _sha256_file(fir_txt_path),
        "wav_sha256": wav_sha256,
        "coeff_hash_sha256": str(result.verification.coeff_hash_sha256),
        "request_fingerprint_sha256": str(result.verification.request_fingerprint_sha256),
        "wav_export_summary": wav_payload,
    }


def _write_release_pack(
    pack_root: Path,
    *,
    product: dict[str, Any],
    eq: dict[str, Any],
    vendors: dict[str, dict[str, Any]],
    db_source: str,
    rates: tuple[int, ...],
    fft_size: int,
    headroom_db: float,
    profile: str | PipelineProfile,
    oversample_factor: int,
    design_oversample: int,
    window_type: str | None,
    window_preset: str | None,
    target_sample_rate: int | None,
    benchmark_warmup: int,
    benchmark_repeat: int,
    fir_dirname: str,
    wav_dirname: str,
    keep_existing_artifacts: bool,
) -> None:
    profile = _validate_strict_opra_release_config(
        rates=rates,
        fft_size=fft_size,
        headroom_db=headroom_db,
        profile=profile,
        oversample_factor=oversample_factor,
        design_oversample=design_oversample,
        window_type=window_type,
        window_preset=window_preset,
        target_sample_rate=target_sample_rate,
        benchmark_warmup=benchmark_warmup,
        benchmark_repeat=benchmark_repeat,
        fir_dirname=fir_dirname,
        wav_dirname=wav_dirname,
        keep_existing_artifacts=keep_existing_artifacts,
    )
    fir_output_dir, wav_output_dir = _prepare_pack_directories(
        pack_root,
        fir_dirname=fir_dirname,
        wav_dirname=wav_dirname,
    )
    if not keep_existing_artifacts:
        _clean_existing_pack_artifacts(fir_output_dir, wav_output_dir, pack_root, rates)

    started = perf_counter()
    results = generate_fir_multi_rate(
        magnitude={"type": "eq_json", "eq": eq},
        rates=rates,
        fft_size=fft_size,
        headroom_db=headroom_db,
        true_peak=True,
        oversample_factor=oversample_factor,
        window_type=window_type,
        window_preset=window_preset,
        design_oversample=design_oversample,
        target_sample_rate=target_sample_rate,
        return_details=True,
        profile=profile,
    )
    results = _apply_opra_post_scale_to_results(results)
    elapsed_ms_per_rate = ((perf_counter() - started) * 1000.0) / max(len(results), 1)

    if not all(isinstance(result, PipelineResult) for result in results.values()):
        _write_verification_artifacts(fir_output_dir, product["id"], results, elapsed_ms_per_rate)
        _write_shared_reports(fir_output_dir, product["id"], results)
        _write_eq_pack(
            pack_root,
            product["id"],
            {
                sample_rate: np.asarray(result.fir_final, dtype=np.float64)
                for sample_rate, result in results.items()
            },
        )
        (pack_root / "selected_eq.txt").write_text(eq_to_txt(eq), encoding="utf-8")
        _write_opra_notice(pack_root, product["id"], db_source)
        _write_opra_attribution(
            pack_root,
            product=product,
            eq=eq,
            vendors=vendors,
            db_source=db_source,
        )
        return

    detailed_results: list[tuple[int, PipelineResult]] = []
    for sample_rate, result in sorted(results.items()):
        if not isinstance(result, PipelineResult):
            raise TypeError("Release pack generation expects detailed PipelineResult values")
        detailed_results.append((int(sample_rate), result))

    if len(detailed_results) <= 1:
        artifacts = [
            _write_release_rate_artifact(
                sample_rate=sample_rate,
                result=result,
                rates=rates,
                target_sample_rate=target_sample_rate,
                fir_output_dir=fir_output_dir,
                wav_output_dir=wav_output_dir,
                profile=profile,
                oversample_factor=oversample_factor,
                benchmark_warmup=benchmark_warmup,
                benchmark_repeat=benchmark_repeat,
            )
            for sample_rate, result in detailed_results
        ]
    else:
        artifact_by_rate: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(detailed_results))) as executor:
            future_to_rate = {
                executor.submit(
                    _write_release_rate_artifact,
                    sample_rate=sample_rate,
                    result=result,
                    rates=rates,
                    target_sample_rate=target_sample_rate,
                    fir_output_dir=fir_output_dir,
                    wav_output_dir=wav_output_dir,
                    profile=profile,
                    oversample_factor=oversample_factor,
                    benchmark_warmup=benchmark_warmup,
                    benchmark_repeat=benchmark_repeat,
                ): sample_rate
                for sample_rate, result in detailed_results
            }
            for future in as_completed(future_to_rate):
                artifact_by_rate[future_to_rate[future]] = future.result()
        artifacts = [artifact_by_rate[sample_rate] for sample_rate, _ in detailed_results]

    selected_eq_path = fir_output_dir / "selected_eq.txt"
    notice_path = fir_output_dir / OPRA_NOTICE_FILENAME
    attribution_path = fir_output_dir / OPRA_ATTRIBUTION_FILENAME
    cross_rate_path = fir_output_dir / "fir_cross_rate_consistency.json"
    release_verdict_path = fir_output_dir / "fir_release_verdict.json"
    manifest_path = fir_output_dir / "build_manifest.json"

    selected_eq_path.write_text(eq_to_txt(eq), encoding="utf-8")
    _write_opra_notice(fir_output_dir, product["id"], db_source)
    _write_opra_attribution(
        fir_output_dir,
        product=product,
        eq=eq,
        vendors=vendors,
        db_source=db_source,
    )

    cross_rate_summary = cross_rate_consistency(results)
    cross_rate_payload = asdict(cross_rate_summary) if cross_rate_summary is not None else None
    if cross_rate_payload is not None:
        rate_target_hashes = {
            str(sample_rate): result.verification.target_hash_sha256
            for sample_rate, result in results.items()
        }
        cross_rate_payload["rate_target_hashes_derived_from_master"] = rate_target_hashes
        cross_rate_payload["targets_derived_from_canonical_master"] = True
        cross_rate_payload["target_projection"] = _target_projection_metadata(
            rates=tuple(sorted(int(sample_rate) for sample_rate in results)),
            target_sample_rate=target_sample_rate,
        )
        cross_rate_payload["canonical_master_target_hash_sha256"] = request_fingerprint(
            {
                "target_policy": "canonical_union_fft_bin_centers",
                "target_sample_rate": target_sample_rate,
                "target_projection": _target_projection_metadata(
                    rates=tuple(sorted(int(sample_rate) for sample_rate in results)),
                    target_sample_rate=target_sample_rate,
                ),
                "eq_id": eq["id"],
                "fft_size": int(fft_size),
                "rates": sorted(int(sample_rate) for sample_rate in results),
                "rate_target_hashes": rate_target_hashes,
            }
        )
        cross_rate_path.write_text(
            metadata_json_dumps(cross_rate_payload, reproducible=True),
            encoding="utf-8",
        )

    release_verdict = _build_release_verdict(results, cross_rate_payload)
    if not bool(release_verdict["export_safe"]):
        raise RuntimeError("Release verdict marks export as unsafe")
    if not bool(release_verdict["playback_safe"]):
        raise RuntimeError("Release verdict marks playback as unsafe")
    if str(release_verdict["recommended_use"]) != "production_release":
        raise RuntimeError("Release verdict is not production_release")
    release_verdict_path.write_text(
        metadata_json_dumps(release_verdict, reproducible=True),
        encoding="utf-8",
    )

    _, _, target = parse_eq_id(eq["id"])
    _write_pack_manifest(
        manifest_path,
        pack_root=pack_root,
        fir_output_dir=fir_output_dir,
        wav_output_dir=wav_output_dir,
        product_id=product["id"],
        eq_id=eq["id"],
        target=target,
        db_source=db_source,
        rates=rates,
        fft_size=fft_size,
        headroom_db=headroom_db,
        profile=profile,
        oversample_factor=oversample_factor,
        design_oversample=design_oversample,
        window_type=window_type,
        window_preset=window_preset,
        target_sample_rate=target_sample_rate,
        benchmark_warmup=benchmark_warmup,
        benchmark_repeat=benchmark_repeat,
        keep_existing_artifacts=keep_existing_artifacts,
        release_verdict=release_verdict,
        cross_rate_payload=cross_rate_payload,
        artifacts=artifacts,
    )
    wav_zip_path = pack_root / "wav_pack.zip"
    proof_zip_path = pack_root / "proof_files.zip"
    wav_zip_files = [Path(str(artifact["wav_path"])) for artifact in artifacts]
    proof_zip_files = [
        path
        for artifact in artifacts
        for path in (
            Path(str(artifact["fir_path"])),
            Path(str(artifact["pipeline_json"])),
            Path(str(artifact["benchmark_json"])),
            Path(str(artifact["analysis_json"])),
        )
    ]
    proof_zip_files.extend(
        [
            selected_eq_path,
            notice_path,
            attribution_path,
            release_verdict_path,
            manifest_path,
        ]
    )
    if cross_rate_payload is not None:
        proof_zip_files.append(cross_rate_path)
    _write_directory_zip(wav_output_dir, wav_zip_path, file_paths=wav_zip_files)
    _write_directory_zip(fir_output_dir, proof_zip_path, file_paths=proof_zip_files)


def _write_eq_pack(output_dir: Path, product_id: str, results: dict[int, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = _safe_name(product_id)

    for sample_rate, fir in results.items():
        wav_path = output_dir / f"{base_name}_{sample_rate}.wav"
        write(wav_path, sample_rate, canonical_wav_array(fir))
        print(f"  [OK] {wav_path}")


def _write_verification_artifacts(output_dir: Path, product_id: str, results: dict[int, object], elapsed_ms_per_rate: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = _safe_name(product_id)

    for sample_rate, result in results.items():
        base_out = output_dir / f"{base_name}_{sample_rate}.txt"
        save_fir(base_out, result.fir_final)
        write_pipeline_report(
            base_out=base_out,
            result=result,
            export_json=True,
            plot=False,
            elapsed_ms=elapsed_ms_per_rate,
            mode=None,
            target_validation=None,
        )


def _apply_opra_post_scale_to_results(results: dict[int, np.ndarray | PipelineResult]) -> dict[int, np.ndarray | PipelineResult]:
    adjusted_results: dict[int, np.ndarray | PipelineResult] = {}
    for sample_rate, result in results.items():
        if not isinstance(result, PipelineResult):
            adjusted_results[sample_rate] = result
            continue
        desired_margin_db = max(
            float(result.spec.profile_config.playback_true_peak_margin_db),
            float(result.verification.true_peak_min_safe_margin_db),
        )
        adjusted_result, attenuation_db = _apply_opra_post_scale(result, desired_margin_db)
        if attenuation_db > 0.0:
            print(f"  [INFO] Applied OPRA post true-peak safety scaling: {attenuation_db:.3f} dB at {sample_rate} Hz")
        adjusted_results[sample_rate] = adjusted_result
    return adjusted_results


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
    if export_safe and playback_safe:
        recommended_use = "production_release"
    else:
        recommended_use = "review_before_release"
    return {
        "export_safe": bool(export_safe),
        "playback_safe": bool(playback_safe),
        "cross_rate_consistent": bool(cross_rate_consistent),
        "recommended_use": recommended_use,
    }


def _write_shared_reports(
    output_dir: Path,
    product_id: str,
    results: dict[int, object],
) -> None:
    summary = cross_rate_consistency(results)
    if summary is None:
        return

    base_name = _safe_name(product_id)
    cross_rate_payload = asdict(summary)
    (output_dir / f"{base_name}_cross_rate_consistency.json").write_text(
        json.dumps(cross_rate_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{base_name}_release_verdict.json").write_text(
        json.dumps(_build_release_verdict(results, cross_rate_payload), indent=2),
        encoding="utf-8",
    )


def _write_opra_notice(output_dir: Path, product_id: str, db_source: str) -> None:
    notice_path = output_dir / OPRA_NOTICE_FILENAME
    notice_path.write_text(_build_opra_notice(product_id, db_source), encoding="utf-8")


def cmd_show(args: argparse.Namespace) -> int:
    products, eqs_by_product, vendors = load_opra_jsonl(args.db)
    product = find_product(products, args.query)
    eq = select_eq(product, eqs_by_product, vendors, target=args.target, measurement=args.measurement)
    vendor, _, target = parse_eq_id(eq["id"])

    print("Product:", product["id"])
    print("Vendor:", vendor)
    print("Measurement:", extract_measurement(eq))
    print("Target:", target)
    print()
    print(eq_to_txt(eq))
    return 0


def cmd_eq_pack(args: argparse.Namespace) -> int:
    products, eqs_by_product, vendors = load_opra_jsonl(args.db)
    product = find_product(products, args.query)
    eq = select_eq(product, eqs_by_product, vendors, target=args.target)
    pack_root = _build_default_pack_root(Path(DEFAULT_PROFILE_ROOT), eq["id"], product_id=product["id"])
    print(f"[PROCESS] {product['id']} -> {pack_root}")
    _write_release_pack(
        pack_root,
        product=product,
        eq=eq,
        vendors=vendors,
        db_source=str(args.db),
        rates=DEFAULT_RATES,
        fft_size=STRICT_OPRA_FFT_SIZE,
        headroom_db=STRICT_OPRA_HEADROOM_DB,
        profile=DEFAULT_PROFILE_NAME,
        oversample_factor=STRICT_OPRA_OVERSAMPLE_FACTOR,
        design_oversample=STRICT_OPRA_DESIGN_OVERSAMPLE,
        window_type=DEFAULT_WINDOW_TYPE,
        window_preset=DEFAULT_WINDOW_PRESET,
        target_sample_rate=args.target_sample_rate,
        benchmark_warmup=STRICT_OPRA_BENCHMARK_WARMUP,
        benchmark_repeat=STRICT_OPRA_BENCHMARK_REPEAT,
        fir_dirname=DEFAULT_FIR_DIRNAME,
        wav_dirname=DEFAULT_WAV_DIRNAME,
        keep_existing_artifacts=False,
    )
    return 0


def cmd_eq_pack_all(args: argparse.Namespace) -> int:
    products, eqs_by_product, vendors = load_opra_jsonl(args.db)
    output_root = _arg_or_default(args, "output", None)
    base_root = Path(output_root) if output_root is not None else Path(DEFAULT_PROFILE_ROOT)
    all_profiles = bool(_arg_or_default(args, "all_profiles", False))
    fail_on_skip = bool(_arg_or_default(args, "fail_on_skip", False))
    shard_count, shard_index = _resolve_pack_shard_args(args)
    skipped: list[str] = []
    processed = 0

    if all_profiles:
        jobs = _iter_profile_map_pack_jobs(
            products,
            eqs_by_product,
            vendors,
            root_dir=base_root,
        )
        for job_number, (product_id, product, eq, pack_root) in enumerate(jobs):
            if job_number % shard_count != shard_index:
                continue
            print(f"[PROCESS] {eq['id']} -> {pack_root}")
            try:
                _write_release_pack(
                    pack_root,
                    product=product,
                    eq=eq,
                    vendors=vendors,
                    db_source=str(args.db),
                    rates=DEFAULT_RATES,
                    fft_size=STRICT_OPRA_FFT_SIZE,
                    headroom_db=STRICT_OPRA_HEADROOM_DB,
                    profile=DEFAULT_PROFILE_NAME,
                    oversample_factor=STRICT_OPRA_OVERSAMPLE_FACTOR,
                    design_oversample=STRICT_OPRA_DESIGN_OVERSAMPLE,
                    window_type=DEFAULT_WINDOW_TYPE,
                    window_preset=DEFAULT_WINDOW_PRESET,
                    target_sample_rate=args.target_sample_rate,
                    benchmark_warmup=STRICT_OPRA_BENCHMARK_WARMUP,
                    benchmark_repeat=STRICT_OPRA_BENCHMARK_REPEAT,
                    fir_dirname=DEFAULT_FIR_DIRNAME,
                    wav_dirname=DEFAULT_WAV_DIRNAME,
                    keep_existing_artifacts=False,
                )
                processed += 1
            except Exception as exc:
                skipped.append(str(eq["id"]))
                print(f"[SKIP] {eq['id']} (FIR generation failed: {exc})")

        if fail_on_skip and skipped:
            raise RuntimeError(f"OPRA pack-all skipped {len(skipped)} profile(s); first skipped: {skipped[0]}")
        if fail_on_skip and processed == 0:
            raise RuntimeError("OPRA pack-all generated no packs")
        return 0

    job_number = 0
    for product_id, product in products.items():
        eqs = eqs_by_product.get(product_id, [])
        if not eqs:
            continue
        current_job_number = job_number
        job_number += 1
        if current_job_number % shard_count != shard_index:
            continue

        try:
            eq = select_eq(product, eqs_by_product, vendors)
        except ValueError as exc:
            skipped.append(product_id)
            print(f"[SKIP] {product_id} ({exc})")
            continue

        pack_root = _resolve_multi_pack_root(base_root, eq["id"], product_id=product_id)
        print(f"[PROCESS] {product_id} -> {pack_root}")
        try:
            _write_release_pack(
                pack_root,
                product=product,
                eq=eq,
                vendors=vendors,
                db_source=str(args.db),
                rates=DEFAULT_RATES,
                fft_size=STRICT_OPRA_FFT_SIZE,
                headroom_db=STRICT_OPRA_HEADROOM_DB,
                profile=DEFAULT_PROFILE_NAME,
                oversample_factor=STRICT_OPRA_OVERSAMPLE_FACTOR,
                design_oversample=STRICT_OPRA_DESIGN_OVERSAMPLE,
                window_type=DEFAULT_WINDOW_TYPE,
                window_preset=DEFAULT_WINDOW_PRESET,
                target_sample_rate=args.target_sample_rate,
                benchmark_warmup=STRICT_OPRA_BENCHMARK_WARMUP,
                benchmark_repeat=STRICT_OPRA_BENCHMARK_REPEAT,
                fir_dirname=DEFAULT_FIR_DIRNAME,
                wav_dirname=DEFAULT_WAV_DIRNAME,
                keep_existing_artifacts=False,
            )
            processed += 1
        except Exception as exc:
            skipped.append(product_id)
            print(f"[SKIP] {product_id} (FIR generation failed: {exc})")
            continue

    if fail_on_skip and skipped:
        raise RuntimeError(f"OPRA pack-all skipped {len(skipped)} product(s); first skipped: {skipped[0]}")
    if fail_on_skip and processed == 0:
        raise RuntimeError("OPRA pack-all generated no packs")
    return 0


def cmd_tier3_pack(args: argparse.Namespace) -> int:
    products, eqs_by_product, vendors = load_opra_jsonl(args.db)
    product = find_product(products, args.query)
    eq = select_eq(product, eqs_by_product, vendors, target=args.target)
    source_rate = ensure_positive_sample_rate(args.source_rate)
    design_rate = ensure_positive_sample_rate(args.design_rate)
    target_sample_rate = design_rate if args.target_sample_rate is None else ensure_positive_sample_rate(args.target_sample_rate)
    pack_root = (
        Path(args.output)
        if args.output is not None
        else _tier3_pack_root(
            eq["id"],
            product_id=product["id"],
            source_rate=source_rate,
            design_rate=design_rate,
        )
    )
    print(f"[PROCESS TIER3] {product['id']} {source_rate}->{design_rate} -> {pack_root}")
    _write_tier3_pack(
        pack_root,
        product=product,
        eq=eq,
        vendors=vendors,
        db_source=str(args.db),
        source_rate=source_rate,
        design_rate=design_rate,
        target_sample_rate=target_sample_rate,
        chain_response_path=Path(args.chain_response),
        chain_response_scale=args.chain_response_scale,
        resampler_name=args.resampler_name,
        kernel_id=args.kernel_id,
        transition_hz=args.chain_transition_hz,
    )
    return 0


def cmd_profile_map(args: argparse.Namespace) -> int:
    products, eqs_by_product, vendors = load_opra_jsonl(args.db)
    root_dir = Path(args.root)
    payload = _build_profile_directory_map(
        products,
        eqs_by_product,
        vendors,
        root_dir=root_dir,
    )

    if args.mkdirs:
        for profile in payload["profiles"].values():
            (root_dir / Path(profile["directory_relpath"])).mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(metadata_json_dumps(payload, reproducible=True), encoding="utf-8")
    print(
        f"[OK] Wrote OPRA profile directory map: {output_path} "
        f"({payload['profile_count']} profiles, {payload['auto_selected_count']} auto-selected, "
        f"{payload['manual_only_count']} manual-only)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OPRA EQ selector and FIR pack exporter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_show = subparsers.add_parser("show", help="Resolve one product and print the selected EQ")
    parser_show.add_argument("db", nargs="?", default=DEFAULT_OPRA_DB_URL)
    parser_show.add_argument("query")
    parser_show.add_argument("--target")
    parser_show.add_argument("--measurement")
    parser_show.set_defaults(func=cmd_show)

    parser_pack = subparsers.add_parser("eq-pack", help="Build a FIR WAV pack for one selected product")
    parser_pack.add_argument("db", nargs="?", default=DEFAULT_OPRA_DB_URL)
    parser_pack.add_argument("query")
    parser_pack.add_argument("--target", required=True)
    parser_pack.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Evaluate OPRA EQ at this fixed playback sample rate before strict multi-rate discretization",
    )
    parser_pack.set_defaults(func=cmd_eq_pack)

    parser_all = subparsers.add_parser("eq-pack-all", help="Build FIR WAV packs from the OPRA database")
    parser_all.add_argument("db", nargs="?", default=DEFAULT_OPRA_DB_URL)
    parser_all.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Evaluate OPRA EQ at this fixed playback sample rate before strict multi-rate discretization",
    )
    parser_all.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Root directory for generated product packs; defaults to fir_profiles",
    )
    parser_all.add_argument(
        "--all-profiles",
        action="store_true",
        help="Build every valid OPRA EQ profile instead of only the auto-selected/default profile per product",
    )
    parser_all.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Return a failing exit if any selected OPRA product/profile cannot be exported",
    )
    parser_all.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split pack generation into this many deterministic shards",
    )
    parser_all.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Generate only this zero-based shard index",
    )
    parser_all.set_defaults(func=cmd_eq_pack_all)

    parser_tier3 = subparsers.add_parser(
        "tier3-pack",
        help="Build one chain-specific Tier 3 FIR compensated for an explicit measured resampler response",
    )
    parser_tier3.add_argument("db", nargs="?", default=DEFAULT_OPRA_DB_URL)
    parser_tier3.add_argument("query")
    parser_tier3.add_argument("--target", required=True)
    parser_tier3.add_argument("--source-rate", type=int, required=True)
    parser_tier3.add_argument("--design-rate", type=int, required=True)
    parser_tier3.add_argument("--target-sample-rate", type=int, default=None)
    parser_tier3.add_argument("--chain-response", type=Path, required=True)
    parser_tier3.add_argument("--chain-response-scale", choices=["linear", "db"], default="linear")
    parser_tier3.add_argument(
        "--chain-transition-hz",
        type=float,
        default=0.0,
        help="Optional explicit cosine blend width below the active band edge; default 0 keeps exact inverse compensation",
    )
    parser_tier3.add_argument("--resampler-name", default="measured_resampler")
    parser_tier3.add_argument("--kernel-id", default=None)
    parser_tier3.add_argument("--output", type=Path, default=None)
    parser_tier3.set_defaults(func=cmd_tier3_pack)

    parser_map = subparsers.add_parser(
        "profile-map",
        help="Write a stable directory mapping for every valid EQ profile in the database",
    )
    parser_map.add_argument("db", nargs="?", default=DEFAULT_OPRA_DB_URL)
    parser_map.add_argument("--output", default="opra_profile_directory_map.json")
    parser_map.add_argument(
        "--root",
        default="fir_profiles",
        help="Root directory that the relative profile paths are anchored to",
    )
    parser_map.add_argument(
        "--mkdirs",
        action="store_true",
        help="Create the mapped directory tree on disk as well as writing the JSON manifest",
    )
    parser_map.set_defaults(func=cmd_profile_map)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
