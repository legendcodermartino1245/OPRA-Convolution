from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .artifact_metrics import (
    artifact_contract,
    canonical_wav_array,
    export_parity,
    float_safety,
    reconstruction_error,
)
from .validation import ensure_bool, ensure_positive_sample_rate

FLOAT_EPSILON = 1e-12


def txt_to_array(input_path: str | Path) -> np.ndarray:
    values: list[float] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("#", ";", "//")):
                continue
            try:
                value = float(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric sample at line {line_number} in {input_path}: {line!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Impulse-response sample at line {line_number} in {input_path} must be finite"
                )
            values.append(value)

    if not values:
        raise ValueError(f"No numeric impulse-response samples found in {input_path}")

    return np.asarray(values, dtype=np.float64)


def write_wav_float32(output_file: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    wavfile.write(str(output_file), sample_rate, canonical_wav_array(signal))


def txt_to_wav(
    input_file: str | Path,
    output_file: str | Path,
    sample_rate: int = 44100,
    normalize: bool = False,
) -> dict[str, object]:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    normalize = ensure_bool("normalize", normalize)
    signal = txt_to_array(input_file)
    processed = signal.copy()

    peak_before = float(np.max(np.abs(signal)))

    applied_normalization = False
    if normalize:
        peak = peak_before
        if peak > FLOAT_EPSILON:
            processed = processed / peak
            applied_normalization = True

    peak_written = float(np.max(np.abs(processed)))
    write_wav_float32(output_file, processed, sample_rate)
    written_sample_rate, wav_samples = wavfile.read(str(output_file))
    wav_samples = np.asarray(wav_samples, dtype=np.float32)
    wav_roundtrip = np.asarray(wav_samples, dtype=np.float64)

    return {
        "sample_rate": sample_rate,
        "normalized": bool(applied_normalization),
        "peak_before": peak_before,
        "peak_written": peak_written,
        "sample_format": "float32",
        "artifact_contract": artifact_contract(wav_samples, written_sample_rate).__dict__,
        "export_parity": export_parity(processed, wav_roundtrip).__dict__,
        "exported_wav_reconstruction_error": reconstruction_error(
            processed,
            wav_roundtrip,
            sample_rate=written_sample_rate,
            fft_size=max(processed.size, 4096),
        ).__dict__,
        "float_safety": float_safety(canonical_wav_array(wav_samples)).__dict__,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TXT impulse response to WAV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize before WAV export. Disabled by default to avoid hidden gain changes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be > 0")

    try:
        result = txt_to_wav(
            input_file=args.input,
            output_file=args.output,
            sample_rate=args.sample_rate,
            normalize=args.normalize,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print("[INFO] Preamp support removed from txt2wav; this tool now performs raw export only")
    print(f"[INFO] Peak before export: {result['peak_before']:.8f}")
    if result["normalized"]:
        print("[INFO] Normalization applied")
    else:
        print("[INFO] No normalization applied")
    print(f"[INFO] Peak written: {result['peak_written']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
