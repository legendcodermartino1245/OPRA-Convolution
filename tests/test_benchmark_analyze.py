from __future__ import annotations

from pathlib import Path

import pytest

from fir_dsp.benchmark_analyze import benchmark_analyzer


def test_benchmark_analyzer_rejects_bool_iteration_counts(tmp_path: Path) -> None:
    fir_path = tmp_path / "fir.txt"
    fir_path.write_text("1.0\n0.0\n", encoding="utf-8")

    with pytest.raises(TypeError, match="warmup must be an integer, not bool"):
        benchmark_analyzer(fir_path, sample_rate=48_000, oversample_factor=1, warmup=True, repeat=1)

    with pytest.raises(TypeError, match="repeat must be an integer, not bool"):
        benchmark_analyzer(fir_path, sample_rate=48_000, oversample_factor=1, warmup=0, repeat=True)
