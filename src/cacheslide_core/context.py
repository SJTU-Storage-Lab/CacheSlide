"""Immutable request contexts shared by engine-specific lifecycle adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any, depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("CacheSlide request metadata exceeds 32 nesting levels")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if not -(2**63) <= value < 2**64:
            raise ValueError("CacheSlide metadata integers must fit MessagePack")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("CacheSlide metadata keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, depth + 1) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, depth + 1) for item in value)
    raise ValueError("CacheSlide metadata must contain finite JSON values")


@dataclass(frozen=True, slots=True)
class StepContext:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    extra_args: Mapping[str, Any] | None = None
    replay_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("CacheSlide requires a nonempty request ID")
        for name in ("prompt_token_ids", "positions"):
            values = tuple(getattr(self, name))
            if not values or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(f"CacheSlide {name} must contain nonnegative integers")
            object.__setattr__(self, name, values)
        if any(b != a + 1 for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError("CacheSlide query positions must be consecutive")
        if self.extra_args is not None and not isinstance(self.extra_args, Mapping):
            raise ValueError("SamplingParams.extra_args must be a mapping")
        object.__setattr__(self, "extra_args", _freeze_json(self.extra_args))
        if self.replay_token_ids is not None:
            replay = tuple(self.replay_token_ids)
            if (
                any(type(value) is not int or value < 0 for value in replay)
                or len(replay) <= len(self.prompt_token_ids)
                or replay[: len(self.prompt_token_ids)] != self.prompt_token_ids
                or self.positions != tuple(range(len(replay)))
            ):
                raise ValueError(
                    "replay must cover the full prompt and generated suffix"
                )
            object.__setattr__(self, "replay_token_ids", replay)

    @property
    def current_positions(self) -> tuple[int, ...]:
        return self.positions

    @property
    def last_query_position(self) -> int:
        return self.positions[-1]


_CURRENT_STEP: ContextVar[StepContext | None] = ContextVar(
    "cacheslide_current_step", default=None
)


def current_step() -> StepContext | None:
    return _CURRENT_STEP.get()


@contextmanager
def step_scope(step: StepContext | None):
    token = _CURRENT_STEP.set(step)
    try:
        yield step
    finally:
        _CURRENT_STEP.reset(token)
