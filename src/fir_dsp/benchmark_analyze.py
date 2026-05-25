from __future__ import annotations

import argparse
import json
import numbers
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .analyze import analyze_fir_quality, load_fir
from .validation import ensure_positive_int, ensure_positive_sample_rate


@dataclass(frozen=True)
class AnalyzeBenchmarkSummary:
    fir_path: str
    sample_rate: int
    taps: int
    oversample_factor: int
    warmup_runs: int
    measured_runs: int
    min_ms: float
    median_ms: float
    mean_ms: float
    max_ms: float
    stdev_ms: float
    python_version: str
    platform: str


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be >= 0")
    return checked


def _positive_int(name: str, value: Any) -> int:
    return ensure_positive_int(name, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FIR analyzer runtime")
    parser.add_argument("--fir", type=Path, required=True, help="Path to FIR coefficient text file")
    parser.add_argument("--sample-rate", type=int, required=True, help="FIR sample rate in Hz")
    parser.add_argument("--oversample-factor", type=int, default=8, help="Oversampling factor for true-peak analysis")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing")
    parser.add_argument("--repeat", type=int, default=10, help="Measured iterations")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path")
    return parser


def benchmark_analyzer(
    fir_path: Path,
    sample_rate: int,
    oversample_factor: int,
    warmup: int,
    repeat: int,
) -> AnalyzeBenchmarkSummary:
    sample_rate = ensure_positive_sample_rate(sample_rate)
    oversample_factor = ensure_positive_int("oversample_factor", oversample_factor)
    warmup = _non_negative_int("warmup", warmup)
    repeat = _positive_int("repeat", repeat)
    fir = load_fir(fir_path)

    for _ in range(warmup):
        analyze_fir_quality(fir, sample_rate=sample_rate, oversample_factor=oversample_factor)

    timings_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        analyze_fir_quality(fir, sample_rate=sample_rate, oversample_factor=oversample_factor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    mean_ms = float(statistics.fmean(timings_ms))
    median_ms = float(statistics.median(timings_ms))
    stdev_ms = float(statistics.pstdev(timings_ms)) if len(timings_ms) > 1 else 0.0

    return AnalyzeBenchmarkSummary(
        fir_path=str(fir_path),
        sample_rate=sample_rate,
        taps=int(fir.size),
        oversample_factor=oversample_factor,
        warmup_runs=warmup,
        measured_runs=len(timings_ms),
        min_ms=float(min(timings_ms)),
        median_ms=median_ms,
        mean_ms=mean_ms,
        max_ms=float(max(timings_ms)),
        stdev_ms=stdev_ms,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = benchmark_analyzer(
        fir_path=args.fir,
        sample_rate=args.sample_rate,
        oversample_factor=args.oversample_factor,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    payload = asdict(summary)
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
