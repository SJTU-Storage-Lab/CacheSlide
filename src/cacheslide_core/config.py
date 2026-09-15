"""Explicit settings: an untrained positional encoder is never enabled implicitly."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from .policy import WCAConfig


@dataclass(frozen=True)
class CacheSlideSettings:
    artifact_path: str
    cache_root: str
    profile_path: str | None = None
    cpu_budget_bytes: int = 536_870_912
    disk_budget_bytes: int = 2_147_483_648
    max_prompt_tokens: int = 32768
    max_profile_elements: int = 4_194_304
    query_chunk_size: int = 64
    calibration_layer: int = 0
    correction_fraction: float = 0.26
    convergence_threshold: float = 0.12
    convergence_mode: str = "paper_cosine_lt"
    weight_update: str = "previous_layer"
    clamp_alpha: bool = False
    selected_attention: str = "updated_and_self"
    ccpe_position_policy: str = "strict_contextual"

    @classmethod
    def from_mapping(cls, value: dict) -> CacheSlideSettings:
        if not isinstance(value, dict):
            raise ValueError("additional_config.cacheslide must be an object")
        if set(value) - {f.name for f in fields(cls)}:
            raise ValueError("unknown CacheSlide engine setting")
        return cls(**value)

    def __post_init__(self) -> None:
        for name in ("artifact_path", "cache_root"):
            path = getattr(self, name)
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError(f"{name} must be an absolute local path")
        if self.profile_path is not None and (
            not isinstance(self.profile_path, str)
            or not Path(self.profile_path).is_absolute()
        ):
            raise ValueError("profile_path must be an absolute local path")
        for name in (
            "cpu_budget_bytes",
            "disk_budget_bytes",
            "max_prompt_tokens",
            "max_profile_elements",
            "query_chunk_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.calibration_layer) is not int or self.calibration_layer < 0:
            raise ValueError("calibration_layer must be a nonnegative layer index")
        if not isinstance(
            self.selected_attention, str
        ) or self.selected_attention not in {
            "updated_and_self",
            "full_causal",
        }:
            raise ValueError("unknown selected_attention policy")
        if not isinstance(
            self.ccpe_position_policy, str
        ) or self.ccpe_position_policy not in {
            "strict_contextual",
            "mixed_bias_override",
        }:
            raise ValueError("unknown ccpe_position_policy")
        self.wca_config()

    def wca_config(self) -> WCAConfig:
        return WCAConfig(
            correction_fraction=self.correction_fraction,
            convergence_threshold=self.convergence_threshold,
            convergence_mode=self.convergence_mode,
            weight_update=self.weight_update,
            clamp_alpha=self.clamp_alpha,
        )
