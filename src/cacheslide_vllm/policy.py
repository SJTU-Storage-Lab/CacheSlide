"""Validated algorithm policy, independent of Torch, CUDA, and vLLM imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WCAConfig:
    correction_fraction: float = 0.26
    epsilon: float = 1e-8
    gate_interval: int = 4
    convergence_threshold: float = 0.12
    convergence_mode: Literal["paper_cosine_lt", "distance_lt"] = "paper_cosine_lt"
    weight_update: Literal["previous_layer", "same_layer"] = "previous_layer"
    clamp_alpha: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.correction_fraction) or not (
            0 <= self.correction_fraction <= 1
        ):
            raise ValueError("correction_fraction must be in [0, 1]")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if type(self.gate_interval) is not int or self.gate_interval < 1:
            raise ValueError("gate_interval must be a positive integer")
        if not math.isfinite(self.convergence_threshold):
            raise ValueError("convergence_threshold must be finite")
        if self.convergence_mode not in ("paper_cosine_lt", "distance_lt"):
            raise ValueError("unknown convergence_mode")
        if self.weight_update not in ("previous_layer", "same_layer"):
            raise ValueError("unknown weight_update")
