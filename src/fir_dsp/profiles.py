from __future__ import annotations

from dataclasses import dataclass

from .validation import ensure_bool, ensure_non_negative_float


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    enforce_peq_preamp: bool = False
    forbid_manual_preamp: bool = False
    true_peak_required: bool = False
    deterministic_required: bool = True
    enforce_no_windowing: bool = False
    minimum_true_peak_margin_db: float = 0.0
    playback_true_peak_margin_db: float = 0.0
    strict_doctor: bool = False
    description: str = ""
    tap_scaling_mode: str = "neutral"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, PipelineProfile):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("profile name must be a non-empty string")
        for field_name in (
            "enforce_peq_preamp",
            "forbid_manual_preamp",
            "true_peak_required",
            "deterministic_required",
            "enforce_no_windowing",
            "strict_doctor",
        ):
            object.__setattr__(self, field_name, ensure_bool(field_name, getattr(self, field_name)))
        object.__setattr__(
            self,
            "minimum_true_peak_margin_db",
            ensure_non_negative_float("minimum_true_peak_margin_db", self.minimum_true_peak_margin_db),
        )
        object.__setattr__(
            self,
            "playback_true_peak_margin_db",
            ensure_non_negative_float("playback_true_peak_margin_db", self.playback_true_peak_margin_db),
        )

    @property
    def closed_system(self) -> bool:
        return self.enforce_peq_preamp or self.true_peak_required or self.playback_true_peak_margin_db > 0.0

    @property
    def enforce_absolute_peak(self) -> bool:
        return self.enforce_peq_preamp

    @property
    def design_method(self) -> str:
        return "fft"


DEFAULT_PROFILE = PipelineProfile(
    name="default",
    enforce_peq_preamp=True,
    forbid_manual_preamp=True,
    true_peak_required=True,
    deterministic_required=True,
    enforce_no_windowing=True,
    minimum_true_peak_margin_db=1.0,
    playback_true_peak_margin_db=1.0,
    strict_doctor=True,
    tap_scaling_mode="tight",
    description="Strict deterministic FIR pipeline: trusted pipeline preamp only, no windowing, true-peak required, 1 dB playback safety margin, deterministic verification enforced.",
)

PROFILE_REGISTRY: dict[str, PipelineProfile] = {
    DEFAULT_PROFILE.name: DEFAULT_PROFILE,
}


def resolve_profile(profile: str | PipelineProfile | None) -> PipelineProfile:
    if profile is None:
        return DEFAULT_PROFILE
    if isinstance(profile, PipelineProfile):
        return profile
    try:
        return PROFILE_REGISTRY[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown profile '{profile}'. Expected one of {sorted(PROFILE_REGISTRY)}") from exc
