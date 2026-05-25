from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .profiles import PipelineProfile, resolve_profile
from .preamp import validate_preamp_db
from .validation import ensure_bool, ensure_non_negative_float, ensure_positive_int, ensure_positive_sample_rate

if TYPE_CHECKING:
    from .profiles import PipelineProfile


def _validate_fft_size_contract(fft_size: Any) -> int:
    checked = ensure_positive_int("fft_size", fft_size)
    if checked < 2 or checked % 2 != 0:
        raise ValueError("fft_size must be an even integer >= 2")
    if (checked & (checked - 1)) != 0:
        raise ValueError("fft_size must be a power of two")
    return checked


@dataclass(frozen=True)
class DesignRequest:
    sample_rate: int
    filters: list[dict]
    preamp_db: float = -6.0
    taps: int = 6145
    fft_mult: int = 32
    floor_db: float = -90.0
    fmin: float = 16.0
    n_log: int = 131072
    shelf_q_mode: str = "eapo"


@dataclass(frozen=True)
class DesignArtifact:
    sample_rate: int
    taps: int
    impulse_response: np.ndarray
    freqs: np.ndarray
    magnitude_linear: np.ndarray


@dataclass(frozen=True)
class AnalysisSummary:
    sample_rate: int
    taps: int
    true_peak_dbfs: float
    coeff_hash_sha256: str
    impulse_peak_index: int
    energy_20pct: float


@dataclass(frozen=True)
class BenchmarkResult:
    sample_rate: int
    taps: int
    elapsed_seconds: float
    runs: int
    coeff_hash_sha256: str


@dataclass(frozen=True)
class PlotArtifacts:
    magnitude_plot: Path
    impulse_plot: Path


@dataclass(frozen=True)
class WindowSpec:
    name: str | None
    beta: float | None = None
    preset: str | None = None


@dataclass(frozen=True)
class ResponseErrorSummary:
    aligned_max_abs_error: float
    aligned_rms_error: float
    aligned_mean_abs_error: float
    max_abs_error_db: float
    rms_error_db: float
    p95_abs_error_db: float


@dataclass(frozen=True)
class LatencySummary:
    sample_rate: int
    taps: int
    peak_index: int
    peak_latency_ms: float
    nominal_linear_phase_latency_ms: float
    abs_centroid_ms: float
    energy_centroid_ms: float


@dataclass(frozen=True)
class VerificationSummary:
    fir_peak_linear: float
    fir_peak_dbfs: float
    fir_true_peak_linear: float
    fir_true_peak_dbfs: float
    true_peak_target_linear: float
    true_peak_target_dbfs: float
    true_peak_margin_linear: float
    true_peak_margin_db: float
    true_peak_margin_warning: bool
    true_peak_min_safe_margin_db: float
    target_hash_sha256: str
    coeff_hash_sha256: str
    request_fingerprint_sha256: str
    gain_stage_preamp_source: str
    preamp_applied_as_gain_stage: bool
    source_preamp_db: float | None
    source_preamp_present: bool
    source_preamp_used_for_target_shape: bool
    source_preamp_origin: str | None
    final_baked_headroom_db: float
    final_gain_policy: str
    true_peak_policy: str
    true_peak_target_is_baked_safety_ceiling: bool


@dataclass(frozen=True)
class GainTraceabilitySummary:
    source_preamp_db: float | None
    source_preamp_present: bool
    source_preamp_used_for_target_shape: bool
    source_preamp_origin: str | None
    gain_stage_preamp_source: str
    preamp_applied_as_gain_stage: bool
    final_baked_headroom_db: float
    final_gain_policy: str
    true_peak_policy: str
    true_peak_target_is_baked_safety_ceiling: bool


@dataclass(frozen=True)
class PipelineSpec:
    sample_rate: int
    fft_size: int
    input_scale: str
    requested_headroom_db: float
    true_peak: bool
    oversample_factor: int
    design_oversample: int
    window: WindowSpec
    gain_stage_preamp_source: str
    preamp_applied_as_gain_stage: bool
    profile: PipelineProfile
    source_preamp_db: float | None = None
    source_preamp_present: bool = False
    source_preamp_used_for_target_shape: bool = False
    source_preamp_origin: str | None = None
    post_scale_attenuation_db: float = 0.0
    minimum_phase: bool = True
    interpolation_mode: str = "log"
    target_projection: dict[str, Any] | None = None

    @property
    def profile_config(self) -> PipelineProfile:
        return resolve_profile(self.profile)

    def copy(self) -> "PipelineSpec":
        return replace(self)

    @property
    def normalization_headroom_db(self) -> float:
        post_scale = float(self.post_scale_attenuation_db)
        if not self.true_peak:
            return float(self.requested_headroom_db) + post_scale
        return float(self.requested_headroom_db) + float(self.profile_config.playback_true_peak_margin_db) + post_scale

    @property
    def minimum_true_peak_margin_db(self) -> float:
        return float(self.profile_config.minimum_true_peak_margin_db)

    @property
    def true_peak_target_dbfs(self) -> float:
        return -float(self.requested_headroom_db)

    @property
    def normalization_target_dbfs(self) -> float:
        return -float(self.normalization_headroom_db)

    @classmethod
    def from_profile(
        cls,
        *,
        sample_rate: int,
        fft_size: int,
        input_scale: str,
        requested_headroom_db: float,
        true_peak: bool,
        oversample_factor: int,
        design_oversample: int,
        window: WindowSpec,
        gain_stage_preamp_source: str,
        preamp_applied_as_gain_stage: bool,
        source_preamp_db: float | None = None,
        source_preamp_present: bool = False,
        source_preamp_used_for_target_shape: bool = False,
        source_preamp_origin: str | None = None,
        post_scale_attenuation_db: float = 0.0,
        profile: PipelineProfile | str,
        minimum_phase: bool = True,
        interpolation_mode: str = "log",
        target_projection: dict[str, Any] | None = None,
    ) -> "PipelineSpec":
        if not isinstance(profile, (str, PipelineProfile)):
            raise TypeError("profile must be a registered profile name string")
        profile_config = resolve_profile(profile)
        return cls(
            sample_rate=ensure_positive_sample_rate(sample_rate),
            fft_size=_validate_fft_size_contract(fft_size),
            input_scale=input_scale,
            requested_headroom_db=ensure_non_negative_float("requested_headroom_db", requested_headroom_db),
            true_peak=ensure_bool("true_peak", true_peak),
            oversample_factor=ensure_positive_int("oversample_factor", oversample_factor),
            design_oversample=ensure_positive_int("design_oversample", design_oversample),
            window=window,
            gain_stage_preamp_source=gain_stage_preamp_source,
            preamp_applied_as_gain_stage=ensure_bool("preamp_applied_as_gain_stage", preamp_applied_as_gain_stage),
            source_preamp_db=validate_preamp_db(source_preamp_db),
            source_preamp_present=ensure_bool("source_preamp_present", source_preamp_present),
            source_preamp_used_for_target_shape=ensure_bool("source_preamp_used_for_target_shape", source_preamp_used_for_target_shape),
            source_preamp_origin=source_preamp_origin,
            post_scale_attenuation_db=ensure_non_negative_float("post_scale_attenuation_db", post_scale_attenuation_db),
            profile=profile_config,
            minimum_phase=ensure_bool("minimum_phase", minimum_phase),
            interpolation_mode=interpolation_mode,
            target_projection=target_projection,
        )


@dataclass(frozen=True)
class PipelineResult:
    sample_rate: int
    fft_size: int
    input_scale: str
    headroom_db: float
    true_peak: bool
    oversample_factor: int
    design_oversample: int
    window: WindowSpec
    spec: PipelineSpec
    profile: PipelineProfile
    target_magnitude: np.ndarray
    fir_linear: np.ndarray
    fir_windowed: np.ndarray
    fir_final: np.ndarray
    actual_magnitude: np.ndarray
    actual_magnitude_db: np.ndarray
    target_magnitude_db: np.ndarray
    error: ResponseErrorSummary
    latency: LatencySummary
    verification: VerificationSummary
    gain_traceability: GainTraceabilitySummary
    system_validation: Any
