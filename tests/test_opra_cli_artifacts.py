import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io.wavfile import write

import fir_dsp.opra_cli as opra_cli
from fir_dsp.api import generate_fir_pipeline
from fir_dsp.core import coeff_hash_sha256
from fir_dsp.opra_cli import _build_profile_directory_map, _write_verification_artifacts, main
from fir_dsp.profiles import PipelineProfile


def test_opra_verification_txt_round_trips_to_reported_coeff_hash(tmp_path):
    result = generate_fir_pipeline(
        magnitude=np.ones(513, dtype=np.float64),
        fft_size=1024,
        headroom_db=6.0,
        true_peak=True,
        oversample_factor=8,
        return_details=True,
    )

    _write_verification_artifacts(tmp_path, "test/product", {48_000: result}, elapsed_ms_per_rate=1.0)

    txt_path = tmp_path / "test_product_48000.txt"
    json_path = tmp_path / "test_product_48000.json"
    loaded = np.loadtxt(txt_path, dtype=np.float64)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))

    assert np.array_equal(loaded, result.fir_final)
    assert coeff_hash_sha256(loaded) == metadata["verification"]["coeff_hash_sha256"]


def test_opra_wav_payload_reuses_precomputed_export_metadata(tmp_path, monkeypatch):
    fir = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    precomputed = {
        "artifact_contract": {
            "wav_subtype": "FLOAT",
            "wav_bits_per_sample": 32,
            "channels": 1,
            "frames": 3,
            "sample_rate": 48_000,
            "duration_ms": 0.0625,
        },
        "export_parity": {
            "wav_txt_max_abs_diff": 0.0,
            "wav_txt_rms_diff": 0.0,
            "wav_txt_allclose": True,
        },
        "exported_wav_reconstruction_error": {
            "max_abs_error_db": 0.0,
            "rms_error_db": 0.0,
            "p95_abs_error_db": 0.0,
        },
        "float_safety": {
            "has_nan": False,
            "has_inf": False,
            "has_denormals": False,
            "min_nonzero_abs_coeff": 0.25,
            "zero_tap_count": 0,
        },
    }

    def fail_reconstruction_error(*_args, **_kwargs):
        raise AssertionError("reconstruction_error should be reused from pipeline metadata")

    monkeypatch.setattr(opra_cli, "reconstruction_error", fail_reconstruction_error)

    payload = opra_cli._write_wav_payload_from_fir(
        fir=fir,
        output_file=tmp_path / "fir.wav",
        sample_rate=48_000,
        precomputed_export_metadata=precomputed,
    )

    assert (tmp_path / "fir.wav").exists()
    assert payload["artifact_contract"] == precomputed["artifact_contract"]
    assert payload["exported_wav_reconstruction_error"] == precomputed["exported_wav_reconstruction_error"]


def test_opra_wav_payload_fast_path_matches_full_validation(tmp_path):
    fir = np.sin(np.linspace(0.0, 20.0, 4096, dtype=np.float64)) * 0.25
    slow_path = tmp_path / "slow.wav"
    fast_path = tmp_path / "fast.wav"

    slow_payload = opra_cli._write_wav_payload_from_fir(
        fir=fir,
        output_file=slow_path,
        sample_rate=48_000,
    )
    precomputed = {
        "artifact_contract": slow_payload["artifact_contract"],
        "export_parity": slow_payload["export_parity"],
        "exported_wav_reconstruction_error": slow_payload["exported_wav_reconstruction_error"],
        "float_safety": slow_payload["float_safety"],
    }
    fast_payload = opra_cli._write_wav_payload_from_fir(
        fir=fir,
        output_file=fast_path,
        sample_rate=48_000,
        precomputed_export_metadata=precomputed,
    )

    assert fast_payload == slow_payload
    assert fast_path.read_bytes() == slow_path.read_bytes()


def test_opra_wav_hash_writer_matches_scipy_direct_bytes(tmp_path):
    fir = np.array([0.0, 0.5, -0.25, 0.125], dtype=np.float64)
    hashed_path = tmp_path / "hashed.wav"
    direct_path = tmp_path / "direct.wav"

    payload, wav_sha256 = opra_cli._write_wav_payload_and_hash_from_fir(
        fir=fir,
        output_file=hashed_path,
        sample_rate=48_000,
    )
    write(direct_path, 48_000, opra_cli.canonical_wav_array(fir))

    assert hashed_path.read_bytes() == direct_path.read_bytes()
    assert wav_sha256 == opra_cli._sha256_file(hashed_path)
    assert payload == opra_cli._write_wav_payload_from_fir(
        fir=fir,
        output_file=tmp_path / "payload_only.wav",
        sample_rate=48_000,
    )


def test_profile_directory_parts_cache_preserves_sanitized_paths():
    opra_cli._profile_directory_parts.cache_clear()

    eq_id = "Vendor Name!:Fancy Product 100%::target/name harman+2026"

    assert opra_cli._profile_directory_parts(eq_id) == (
        "Vendor_Name",
        "Fancy_Product_100",
        "target_name_harman_2026",
    )
    assert opra_cli._build_default_pack_root(Path("root"), eq_id) == (
        Path("root") / "Vendor_Name" / "Fancy_Product_100" / "target_name_harman_2026"
    )

    cache_before = opra_cli._profile_directory_parts.cache_info()
    assert opra_cli._profile_directory_parts(eq_id) == (
        "Vendor_Name",
        "Fancy_Product_100",
        "target_name_harman_2026",
    )
    cache_after = opra_cli._profile_directory_parts.cache_info()

    assert cache_after.hits == cache_before.hits + 1


def test_profile_directory_map_includes_valid_profiles_and_marks_default_selection(tmp_path):
    products = {
        "vendor::product": {
            "id": "vendor::product",
            "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
        }
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eqs_by_product = {
        "vendor::product": [
            {
                "id": "vendor:product::autoeq_crinacle",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Crinacle Rig",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
            {
                "id": "vendor:product::autoeq_crinacle_blue",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Crinacle Rig",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
        ]
    }

    payload = _build_profile_directory_map(
        products,
        eqs_by_product,
        vendors,
        root_dir=tmp_path / "fir_profiles",
    )

    assert payload["profile_count"] == 2
    canonical = payload["profiles"]["vendor:product::autoeq_crinacle"]
    variant = payload["profiles"]["vendor:product::autoeq_crinacle_blue"]
    assert canonical["directory_relpath"] == "vendor/product/autoeq_crinacle"
    assert variant["directory_relpath"] == "vendor/product/autoeq_crinacle_blue"
    assert canonical["default_selected"] is True
    assert canonical["selection_mode"] == "auto"
    assert variant["default_selected"] is False


def test_profile_directory_map_marks_manual_only_variants(tmp_path):
    products = {
        "vendor::product": {
            "id": "vendor::product",
            "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
        }
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eqs_by_product = {
        "vendor::product": [
            {
                "id": "vendor:product::autoeq_regan_cipher_anc_off",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Regan Cipher",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
            {
                "id": "vendor:product::autoeq_regan_cipher_anc_on",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Regan Cipher",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
        ]
    }

    payload = _build_profile_directory_map(
        products,
        eqs_by_product,
        vendors,
        root_dir=tmp_path / "fir_profiles",
    )

    assert payload["profile_count"] == 2
    assert payload["auto_selected_count"] == 0
    assert payload["manual_only_count"] == 2
    for profile in payload["profiles"].values():
        assert profile["selection_mode"] == "manual_required"
        assert profile["default_selected"] is False


def test_eq_pack_all_all_profiles_uses_profile_map_paths(tmp_path, monkeypatch):
    products = {
        "vendor::product": {
            "id": "vendor::product",
            "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
        }
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eqs_by_product = {
        "vendor::product": [
            {
                "id": "vendor:product::autoeq_crinacle_blue",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Crinacle Rig",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
            {
                "id": "vendor:product::autoeq_crinacle",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Crinacle Rig",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
        ]
    }
    root_dir = tmp_path / "profiles"
    calls = []

    monkeypatch.setattr(
        "fir_dsp.opra_cli.load_opra_jsonl",
        lambda _db: (products, eqs_by_product, vendors),
    )

    def fake_write_release_pack(pack_root, **kwargs):
        calls.append(
            {
                "pack_root": Path(pack_root).relative_to(root_dir).as_posix(),
                "product_id": kwargs["product"]["id"],
                "eq_id": kwargs["eq"]["id"],
                "rates": kwargs["rates"],
                "fft_size": kwargs["fft_size"],
                "headroom_db": kwargs["headroom_db"],
                "profile": kwargs["profile"],
                "window_type": kwargs["window_type"],
                "window_preset": kwargs["window_preset"],
                "target_sample_rate": kwargs["target_sample_rate"],
                "keep_existing_artifacts": kwargs["keep_existing_artifacts"],
            }
        )

    monkeypatch.setattr(opra_cli, "_write_release_pack", fake_write_release_pack)

    rc = main(
        [
            "eq-pack-all",
            "db.jsonl",
            "--output",
            str(root_dir),
            "--all-profiles",
            "--fail-on-skip",
            "--target-sample-rate",
            "48000",
        ]
    )

    assert rc == 0
    assert calls == [
        {
            "pack_root": "vendor/product/autoeq_crinacle",
            "product_id": "vendor::product",
            "eq_id": "vendor:product::autoeq_crinacle",
            "rates": opra_cli.DEFAULT_RATES,
            "fft_size": opra_cli.STRICT_OPRA_FFT_SIZE,
            "headroom_db": opra_cli.STRICT_OPRA_HEADROOM_DB,
            "profile": opra_cli.DEFAULT_PROFILE_NAME,
            "window_type": opra_cli.DEFAULT_WINDOW_TYPE,
            "window_preset": opra_cli.DEFAULT_WINDOW_PRESET,
            "target_sample_rate": 48000,
            "keep_existing_artifacts": False,
        },
        {
            "pack_root": "vendor/product/autoeq_crinacle_blue",
            "product_id": "vendor::product",
            "eq_id": "vendor:product::autoeq_crinacle_blue",
            "rates": opra_cli.DEFAULT_RATES,
            "fft_size": opra_cli.STRICT_OPRA_FFT_SIZE,
            "headroom_db": opra_cli.STRICT_OPRA_HEADROOM_DB,
            "profile": opra_cli.DEFAULT_PROFILE_NAME,
            "window_type": opra_cli.DEFAULT_WINDOW_TYPE,
            "window_preset": opra_cli.DEFAULT_WINDOW_PRESET,
            "target_sample_rate": 48000,
            "keep_existing_artifacts": False,
        },
    ]

    calls.clear()
    rc = main(
        [
            "eq-pack-all",
            "db.jsonl",
            "--output",
            str(root_dir),
            "--all-profiles",
            "--shard-count",
            "2",
            "--shard-index",
            "1",
        ]
    )

    assert rc == 0
    assert [call["eq_id"] for call in calls] == ["vendor:product::autoeq_crinacle_blue"]


def test_eq_pack_all_fail_on_skip_raises_for_all_profiles(tmp_path, monkeypatch):
    products = {
        "vendor::product": {
            "id": "vendor::product",
            "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
        }
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eqs_by_product = {
        "vendor::product": [
            {
                "id": "vendor:product::autoeq_crinacle",
                "data": {
                    "type": "parametric_eq",
                    "product_id": "vendor::product",
                    "details": "Measured by Crinacle Rig",
                    "parameters": {
                        "gain_db": 0.0,
                        "bands": [
                            {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                        ],
                    },
                },
            },
        ]
    }

    monkeypatch.setattr(
        "fir_dsp.opra_cli.load_opra_jsonl",
        lambda _db: (products, eqs_by_product, vendors),
    )
    monkeypatch.setattr(
        opra_cli,
        "_write_release_pack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("release gate failed")),
    )

    with pytest.raises(RuntimeError, match="skipped 1 profile"):
        main(
            [
                "eq-pack-all",
                "db.jsonl",
                "--output",
                str(tmp_path / "profiles"),
                "--all-profiles",
                "--fail-on-skip",
            ]
        )


def test_eq_pack_uses_mapped_directory_and_creates_zips(tmp_path, monkeypatch):
    db_path = tmp_path / "db.jsonl"
    db_path.write_text("", encoding="utf-8")
    root_dir = tmp_path / "profiles"
    products = {
        "vendor::product": {
            "id": "vendor::product",
            "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
        }
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eq = {
        "id": "vendor:product::autoeq_crinacle",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Crinacle Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                ],
            },
        },
    }
    eqs_by_product = {"vendor::product": [eq]}
    monkeypatch.setattr(
        "fir_dsp.opra_cli.load_opra_jsonl",
        lambda _db: (products, eqs_by_product, vendors),
    )
    monkeypatch.setattr(
        "fir_dsp.opra_cli.find_product",
        lambda _products, _query: products["vendor::product"],
    )
    monkeypatch.setattr(
        "fir_dsp.opra_cli.select_eq",
        lambda _product, _eqs_by_product, _vendors, target=None, measurement=None: eq,
    )
    monkeypatch.setattr(
        "fir_dsp.opra_cli.generate_fir_multi_rate",
        lambda magnitude, rates, fft_size, headroom_db, true_peak, oversample_factor, window_type, window_preset, design_oversample, target_sample_rate, return_details, profile: {
            int(sample_rate): generate_fir_pipeline(
                magnitude=np.ones((int(fft_size) // 2) + 1, dtype=np.float64),
                fft_size=int(fft_size),
                headroom_db=float(headroom_db),
                sample_rate=int(sample_rate),
                true_peak=bool(true_peak),
                oversample_factor=int(oversample_factor),
                window_type=window_type,
                window_preset=window_preset,
                design_oversample=int(design_oversample),
                return_details=bool(return_details),
                profile=profile,
            )
            for sample_rate in rates
        },
    )
    monkeypatch.setattr(
        "fir_dsp.opra_cli._apply_opra_post_scale_to_results",
        lambda results: results,
    )
    monkeypatch.setattr(opra_cli, "DEFAULT_PROFILE_ROOT", str(root_dir))
    monkeypatch.setattr(opra_cli, "DEFAULT_RATES", (48_000,))
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_FFT_SIZE", 4096)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_HEADROOM_DB", 12.0)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_BENCHMARK_WARMUP", 0)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_BENCHMARK_REPEAT", 1)

    rc = main(
        [
            "eq-pack",
            str(db_path),
            "vendor::product",
            "--target",
            "autoeq_crinacle",
            "--target-sample-rate",
            "48000",
        ]
    )

    assert rc == 0
    pack_root = root_dir / "vendor" / "product" / "autoeq_crinacle"
    fir_dir = pack_root / "fir_pack"
    wav_dir = pack_root / "wav_pack"
    assert (fir_dir / "fir_48000.txt").exists()
    assert (fir_dir / "fir_48000.json").exists()
    assert (fir_dir / "fir_48000_benchmark.json").exists()
    assert (fir_dir / "fir_48000_analysis.json").exists()
    assert (fir_dir / "selected_eq.txt").exists()
    assert (fir_dir / "NOTICE_OPRA.txt").exists()
    assert (fir_dir / "ATTRIBUTION_OPRA.json").exists()
    assert (fir_dir / "fir_release_verdict.json").exists()
    assert (fir_dir / "build_manifest.json").exists()
    assert (wav_dir / "fir_48000.wav").exists()
    assert (pack_root / "wav_pack.zip").exists()
    assert (pack_root / "proof_files.zip").exists()
    pipeline_metadata = json.loads((fir_dir / "fir_48000.json").read_text(encoding="utf-8"))
    manifest = json.loads((fir_dir / "build_manifest.json").read_text(encoding="utf-8"))
    assert pipeline_metadata["profile"] == "default"
    assert pipeline_metadata["true_peak"] is True
    assert pipeline_metadata["oversample_factor"] == 8
    assert pipeline_metadata["design_oversample"] == 1
    assert pipeline_metadata["window_type"] is None
    assert pipeline_metadata["window_preset"] is None
    assert manifest["profile"] == "default"
    assert manifest["window_type"] is None
    assert manifest["window_preset"] is None
    assert manifest["target_sample_rate"] == 48000
    assert manifest["target_projection"] == {
        "measurement_domain": "continuous_source_eq",
        "target_sample_rate": 48000,
        "reference_target_rate": 48000,
        "projection_stage": "pre_design",
        "projection_grid": "canonical_union_fft_bin_centers",
        "interpolation_mode": "log",
        "design_sample_rates": [48000],
    }
    assert manifest["artifacts"][0]["design_sample_rate"] == 48000
    assert manifest["artifacts"][0]["target_projection"]["design_sample_rate"] == 48000
    assert manifest["artifacts"][0]["wav_sha256"] == opra_cli._sha256_file(wav_dir / "fir_48000.wav")
    wav_summary = manifest["artifacts"][0]["wav_export_summary"]
    assert wav_summary["artifact_contract"] == pipeline_metadata["artifact_contract"]
    assert wav_summary["export_parity"] == pipeline_metadata["export_parity"]
    assert wav_summary["exported_wav_reconstruction_error"] == pipeline_metadata["exported_wav_reconstruction_error"]
    assert wav_summary["float_safety"] == pipeline_metadata["float_safety"]

    with zipfile.ZipFile(pack_root / "wav_pack.zip") as wav_zip:
        assert sorted(wav_zip.namelist()) == ["fir_48000.wav"]
    with zipfile.ZipFile(pack_root / "proof_files.zip") as proof_zip:
        names = sorted(proof_zip.namelist())
        assert "build_manifest.json" in names
        assert "fir_48000.txt" in names
        assert "fir_48000.json" in names


def test_directory_zip_accepts_tracked_file_list_matching_directory_scan(tmp_path):
    source_dir = tmp_path / "proof"
    (source_dir / "nested").mkdir(parents=True)
    (source_dir / "b.txt").write_text("b", encoding="utf-8")
    (source_dir / "a.txt").write_text("a", encoding="utf-8")
    (source_dir / "nested" / "c.txt").write_text("c", encoding="utf-8")
    tracked_files = [
        source_dir / "nested" / "c.txt",
        source_dir / "b.txt",
        source_dir / "a.txt",
    ]

    scan_zip_path = tmp_path / "scan.zip"
    tracked_zip_path = tmp_path / "tracked.zip"
    opra_cli._write_directory_zip(source_dir, scan_zip_path)
    opra_cli._write_directory_zip(source_dir, tracked_zip_path, file_paths=tracked_files)

    with zipfile.ZipFile(scan_zip_path) as scan_zip, zipfile.ZipFile(tracked_zip_path) as tracked_zip:
        assert tracked_zip.namelist() == scan_zip.namelist()
        assert {
            name: tracked_zip.read(name)
            for name in tracked_zip.namelist()
        } == {
            name: scan_zip.read(name)
            for name in scan_zip.namelist()
        }


def test_release_pack_manifest_keeps_sorted_artifacts_when_parallel_writes_finish_out_of_order(tmp_path, monkeypatch):
    product = {
        "id": "vendor::product",
        "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eq = {
        "id": "vendor:product::autoeq_crinacle",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Crinacle Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 0.0, "q": 0.7},
                ],
            },
        },
    }
    monkeypatch.setattr(opra_cli, "DEFAULT_RATES", (48_000, 44_100))
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_FFT_SIZE", 1024)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_HEADROOM_DB", 12.0)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_BENCHMARK_WARMUP", 0)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_BENCHMARK_REPEAT", 1)
    monkeypatch.setattr(opra_cli, "_apply_opra_post_scale_to_results", lambda results: results)

    def generate_details(
        magnitude,
        rates,
        fft_size,
        headroom_db,
        true_peak,
        oversample_factor,
        window_type,
        window_preset,
        design_oversample,
        target_sample_rate,
        return_details,
        profile,
    ):
        return {
            int(sample_rate): generate_fir_pipeline(
                magnitude=np.ones((int(fft_size) // 2) + 1, dtype=np.float64),
                fft_size=int(fft_size),
                headroom_db=float(headroom_db),
                sample_rate=int(sample_rate),
                true_peak=bool(true_peak),
                oversample_factor=int(oversample_factor),
                window_type=window_type,
                window_preset=window_preset,
                design_oversample=int(design_oversample),
                return_details=bool(return_details),
                profile=profile,
            )
            for sample_rate in rates
        }

    completion_order: list[int] = []

    def fake_write_rate_artifact(**kwargs):
        sample_rate = int(kwargs["sample_rate"])
        if sample_rate == 44_100:
            time.sleep(0.05)
        completion_order.append(sample_rate)
        fir_output_dir = kwargs["fir_output_dir"]
        wav_output_dir = kwargs["wav_output_dir"]
        return {
            "sample_rate": sample_rate,
            "design_sample_rate": sample_rate,
            "target_projection": {},
            "fir_path": str(fir_output_dir / f"fir_{sample_rate}.txt"),
            "wav_path": str(wav_output_dir / f"fir_{sample_rate}.wav"),
            "pipeline_json": str(fir_output_dir / f"fir_{sample_rate}.json"),
            "benchmark_json": str(fir_output_dir / f"fir_{sample_rate}_benchmark.json"),
            "analysis_json": str(fir_output_dir / f"fir_{sample_rate}_analysis.json"),
        }

    zip_calls = []

    def capture_zip(source_dir, zip_path, *, file_paths=None):
        zip_calls.append(
            (
                Path(source_dir).name,
                Path(zip_path).name,
                None if file_paths is None else sorted(Path(path).name for path in file_paths),
            )
        )

    monkeypatch.setattr(opra_cli, "generate_fir_multi_rate", generate_details)
    monkeypatch.setattr(opra_cli, "_write_release_rate_artifact", fake_write_rate_artifact)
    monkeypatch.setattr(opra_cli, "_write_directory_zip", capture_zip)

    opra_cli._write_release_pack(
        tmp_path,
        product=product,
        eq=eq,
        vendors=vendors,
        db_source="db.jsonl",
        rates=opra_cli.DEFAULT_RATES,
        fft_size=opra_cli.STRICT_OPRA_FFT_SIZE,
        headroom_db=opra_cli.STRICT_OPRA_HEADROOM_DB,
        profile=opra_cli.DEFAULT_PROFILE_NAME,
        oversample_factor=opra_cli.STRICT_OPRA_OVERSAMPLE_FACTOR,
        design_oversample=opra_cli.STRICT_OPRA_DESIGN_OVERSAMPLE,
        window_type=None,
        window_preset=None,
        target_sample_rate=None,
        benchmark_warmup=opra_cli.STRICT_OPRA_BENCHMARK_WARMUP,
        benchmark_repeat=opra_cli.STRICT_OPRA_BENCHMARK_REPEAT,
        fir_dirname=opra_cli.DEFAULT_FIR_DIRNAME,
        wav_dirname=opra_cli.DEFAULT_WAV_DIRNAME,
        keep_existing_artifacts=False,
    )

    manifest = json.loads((tmp_path / "fir_pack" / "build_manifest.json").read_text(encoding="utf-8"))
    assert completion_order[0] == 48_000
    assert [artifact["sample_rate"] for artifact in manifest["artifacts"]] == [44_100, 48_000]
    assert zip_calls == [
        ("wav_pack", "wav_pack.zip", ["fir_44100.wav", "fir_48000.wav"]),
        (
            "fir_pack",
            "proof_files.zip",
            [
                "ATTRIBUTION_OPRA.json",
                "NOTICE_OPRA.txt",
                "build_manifest.json",
                "fir_44100.json",
                "fir_44100.txt",
                "fir_44100_analysis.json",
                "fir_44100_benchmark.json",
                "fir_48000.json",
                "fir_48000.txt",
                "fir_48000_analysis.json",
                "fir_48000_benchmark.json",
                "fir_cross_rate_consistency.json",
                "fir_release_verdict.json",
                "selected_eq.txt",
            ],
        ),
    ]


def test_tier3_pack_records_chain_model_and_uses_separate_root(tmp_path, monkeypatch):
    db_path = tmp_path / "db.jsonl"
    db_path.write_text("", encoding="utf-8")
    root_dir = tmp_path / "tier3_profiles"
    chain_path = tmp_path / "chain_response.txt"
    chain_path.write_text("0 1.0\n22050 0.98\n", encoding="utf-8")
    products = {
        "vendor::product": {
            "id": "vendor::product",
            "data": {"subtype": "over_ear", "vendor_id": "vendor", "name": "Product"},
        }
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    eq = {
        "id": "vendor:product::autoeq_crinacle",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Crinacle Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 1.0, "q": 0.7},
                ],
            },
        },
    }
    eqs_by_product = {"vendor::product": [eq]}
    monkeypatch.setattr(
        "fir_dsp.opra_cli.load_opra_jsonl",
        lambda _db: (products, eqs_by_product, vendors),
    )
    monkeypatch.setattr(
        "fir_dsp.opra_cli.find_product",
        lambda _products, _query: products["vendor::product"],
    )
    monkeypatch.setattr(
        "fir_dsp.opra_cli.select_eq",
        lambda _product, _eqs_by_product, _vendors, target=None, measurement=None: eq,
    )
    monkeypatch.setattr(opra_cli, "DEFAULT_TIER3_PROFILE_ROOT", str(root_dir))
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_FFT_SIZE", 4096)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_HEADROOM_DB", 12.0)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_BENCHMARK_WARMUP", 0)
    monkeypatch.setattr(opra_cli, "STRICT_OPRA_BENCHMARK_REPEAT", 1)

    rc = main(
        [
            "tier3-pack",
            str(db_path),
            "vendor::product",
            "--target",
            "autoeq_crinacle",
            "--source-rate",
            "44100",
            "--design-rate",
            "48000",
            "--chain-response",
            str(chain_path),
            "--resampler-name",
            "roon_measured",
            "--kernel-id",
            "test-kernel",
        ]
    )

    assert rc == 0
    pack_root = root_dir / "vendor" / "product" / "autoeq_crinacle" / "44100_to_48000"
    fir_dir = pack_root / "fir_pack"
    manifest = json.loads((fir_dir / "build_manifest.json").read_text(encoding="utf-8"))
    pipeline_metadata = json.loads((fir_dir / "fir_48000.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == 3
    assert manifest["projection_stage"] == "post_chain_compensated"
    assert manifest["source_rate"] == 44100
    assert manifest["design_sample_rate"] == 48000
    assert manifest["chain_model"]["resampler"]["source_rate"] == 44100
    assert manifest["chain_model"]["resampler"]["target_rate"] == 48000
    assert manifest["chain_model"]["resampler"]["method"] == "measured_response"
    assert manifest["chain_model"]["resampler"]["kernel_identity"] == "test-kernel"
    assert manifest["chain_model"]["resampler"]["active_compensation_min_hz"] == 0.0
    assert manifest["chain_model"]["resampler"]["active_compensation_max_hz"] == 22050.0
    assert manifest["chain_model"]["resampler"]["design_nyquist_hz"] == 24000.0
    assert manifest["chain_model"]["resampler"]["above_active_band_policy"] == "unity_no_source_content"
    assert manifest["chain_model"]["resampler"]["minimum_allowed_chain_magnitude_db"] == -120.0
    assert manifest["chain_model"]["resampler"]["edge_transition_policy"] == "disabled_exact_inverse"
    assert manifest["chain_model"]["resampler"]["conditioning"]["max_inverse_gain_db"] > 0.0
    assert manifest["chain_model"]["resampler"]["conditioning"]["inverse_gain_warning"] is False
    assert manifest["chain_closure_scope"] == {
        "frequency_min_hz": 0.0,
        "frequency_max_hz": 22050.0,
        "bin_count": 1882,
        "policy": "active_source_content_band_only",
    }
    assert manifest["chain_closure_error"]["max_abs_error_db"] <= 0.1
    assert pipeline_metadata["tier"] == 3
    assert pipeline_metadata["target_projection"]["projection_stage"] == "post_chain_compensated"
    assert pipeline_metadata["target_projection"]["chain_model"]["resampler"]["kernel_identity"] == "test-kernel"
    assert pipeline_metadata["target_projection"]["compensation"]["compensation_band_hz"] == {
        "min": 0.0,
        "max": 22050.0,
    }
    assert pipeline_metadata["target_projection"]["compensation"]["outside_compensation_band_policy"] == "unity_no_source_content"
    assert (pack_root / "wav_pack.zip").exists()
    assert (pack_root / "proof_files.zip").exists()


def test_tier3_chain_transition_is_opt_in_and_fingerprinted(tmp_path):
    chain_path = tmp_path / "chain_response.txt"
    chain_path.write_text("0 1.0\n21550 0.9\n22050 0.8\n", encoding="utf-8")
    freqs_hz, magnitude = opra_cli._load_tier3_chain_response(chain_path, scale="linear")
    fft_freqs, exact_chain, active_max = opra_cli._project_tier3_chain_response(
        response_freqs_hz=freqs_hz,
        response_magnitude=magnitude,
        source_rate=44100,
        design_rate=48000,
        fft_size=4096,
    )
    _, blended_chain, _ = opra_cli._project_tier3_chain_response(
        response_freqs_hz=freqs_hz,
        response_magnitude=magnitude,
        source_rate=44100,
        design_rate=48000,
        fft_size=4096,
        transition_hz=500.0,
    )
    conditioning = opra_cli._tier3_chain_conditioning(
        freqs_hz=fft_freqs,
        chain_magnitude=blended_chain,
        active_max_hz=active_max,
    )
    model = opra_cli._tier3_chain_model(
        source_rate=44100,
        design_rate=48000,
        chain_response_path=chain_path,
        chain_response_scale="linear",
        chain_response_magnitude=blended_chain,
        active_max_hz=active_max,
        resampler_name="measured",
        kernel_id="kernel",
        transition_hz=500.0,
        conditioning=conditioning,
    )

    active_mask = (fft_freqs >= 21550.0) & (fft_freqs <= 22050.0)
    assert not np.array_equal(exact_chain[active_mask], blended_chain[active_mask])
    assert model["resampler"]["edge_transition_width_hz"] == 500.0
    assert model["resampler"]["edge_transition_policy"] == "cosine_blend_to_unity"


def test_tier3_chain_response_rejects_low_frequency_extrapolation(tmp_path):
    chain_path = tmp_path / "chain_response.txt"
    chain_path.write_text("20 1.0\n22050 0.99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must start at 0 Hz"):
        opra_cli._load_tier3_chain_response(chain_path, scale="linear")


def test_tier3_chain_response_rejects_unstable_inverse_nulls(tmp_path):
    chain_path = tmp_path / "chain_response.txt"
    chain_path.write_text("0 1.0\n22050 -121.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="below -120 dB"):
        opra_cli._load_tier3_chain_response(chain_path, scale="db")


def test_tier3_chain_response_must_cover_active_content_band(tmp_path):
    chain_path = tmp_path / "chain_response.txt"
    chain_path.write_text("0 1.0\n20000 0.99\n", encoding="utf-8")
    freqs_hz, magnitude = opra_cli._load_tier3_chain_response(chain_path, scale="linear")

    with pytest.raises(ValueError, match="active source/content Nyquist"):
        opra_cli._project_tier3_chain_response(
            response_freqs_hz=freqs_hz,
            response_magnitude=magnitude,
            source_rate=44100,
            design_rate=48000,
            fft_size=4096,
        )


def test_tier3_chain_model_fingerprint_changes_with_measured_response(tmp_path):
    chain_a = tmp_path / "chain_a.txt"
    chain_b = tmp_path / "chain_b.txt"
    chain_a.write_text("0 1.0\n22050 0.99\n", encoding="utf-8")
    chain_b.write_text("0 1.0\n22050 0.97\n", encoding="utf-8")
    freqs_a, mag_a = opra_cli._load_tier3_chain_response(chain_a, scale="linear")
    freqs_b, mag_b = opra_cli._load_tier3_chain_response(chain_b, scale="linear")
    _, projected_a, active_a = opra_cli._project_tier3_chain_response(
        response_freqs_hz=freqs_a,
        response_magnitude=mag_a,
        source_rate=44100,
        design_rate=48000,
        fft_size=4096,
    )
    _, projected_b, active_b = opra_cli._project_tier3_chain_response(
        response_freqs_hz=freqs_b,
        response_magnitude=mag_b,
        source_rate=44100,
        design_rate=48000,
        fft_size=4096,
    )

    model_a = opra_cli._tier3_chain_model(
        source_rate=44100,
        design_rate=48000,
        chain_response_path=chain_a,
        chain_response_scale="linear",
        chain_response_magnitude=projected_a,
        active_max_hz=active_a,
        resampler_name="measured",
        kernel_id="kernel",
        transition_hz=0.0,
        conditioning=opra_cli._tier3_chain_conditioning(
            freqs_hz=np.fft.rfftfreq(4096, d=1.0 / 48000.0),
            chain_magnitude=projected_a,
            active_max_hz=active_a,
        ),
    )
    model_b = opra_cli._tier3_chain_model(
        source_rate=44100,
        design_rate=48000,
        chain_response_path=chain_b,
        chain_response_scale="linear",
        chain_response_magnitude=projected_b,
        active_max_hz=active_b,
        resampler_name="measured",
        kernel_id="kernel",
        transition_hz=0.0,
        conditioning=opra_cli._tier3_chain_conditioning(
            freqs_hz=np.fft.rfftfreq(4096, d=1.0 / 48000.0),
            chain_magnitude=projected_b,
            active_max_hz=active_b,
        ),
    )

    assert opra_cli.request_fingerprint({"chain_model": model_a}) != opra_cli.request_fingerprint({"chain_model": model_b})


@pytest.mark.parametrize(
    "override_args",
    (
        ["--profile", "default"],
        ["--fft-size", "4096"],
        ["--headroom", "12.0"],
        ["--rates", "48000"],
        ["--window", "hann"],
        ["--window-preset", "safe"],
        ["--oversample-factor", "4"],
        ["--design-os", "2"],
        ["--benchmark-repeat", "1"],
        ["--root", "custom_root"],
        ["--output", "custom_output"],
        ["--measurement", "crinacle"],
    ),
)
def test_eq_pack_rejects_release_config_overrides(override_args):
    with pytest.raises(SystemExit):
        main(["eq-pack", "db.jsonl", "vendor::product", "--target", "target", *override_args])


def test_eq_pack_requires_explicit_target():
    with pytest.raises(SystemExit):
        main(["eq-pack", "db.jsonl", "vendor::product"])


def test_eq_pack_accepts_target_sample_rate_as_only_dsp_authority_option():
    args = opra_cli.build_parser().parse_args(
        ["eq-pack", "db.jsonl", "vendor::product", "--target", "target", "--target-sample-rate", "48000"]
    )

    assert args.target_sample_rate == 48000


def test_release_pack_hard_rejects_non_default_profile(tmp_path):
    with pytest.raises(RuntimeError, match="locked to the registered default profile"):
        opra_cli._write_release_pack(
            tmp_path,
            product={},
            eq={},
            vendors={},
            db_source="db.jsonl",
            rates=opra_cli.DEFAULT_RATES,
            fft_size=opra_cli.STRICT_OPRA_FFT_SIZE,
            headroom_db=opra_cli.STRICT_OPRA_HEADROOM_DB,
            profile=PipelineProfile(name="custom"),
            oversample_factor=opra_cli.STRICT_OPRA_OVERSAMPLE_FACTOR,
            design_oversample=opra_cli.STRICT_OPRA_DESIGN_OVERSAMPLE,
            window_type=None,
            window_preset=None,
            target_sample_rate=None,
            benchmark_warmup=opra_cli.STRICT_OPRA_BENCHMARK_WARMUP,
            benchmark_repeat=opra_cli.STRICT_OPRA_BENCHMARK_REPEAT,
            fir_dirname=opra_cli.DEFAULT_FIR_DIRNAME,
            wav_dirname=opra_cli.DEFAULT_WAV_DIRNAME,
            keep_existing_artifacts=False,
        )
