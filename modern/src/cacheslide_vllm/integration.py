"""Request context and instance-scoped adapters for the pinned native runner."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
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


def _native_context():
    from vllm.forward_context import get_forward_context

    return get_forward_context()


def _make_step(runner: Any, positions: Any, inputs_embeds: Any, context: Any):
    metadata = context.attn_metadata
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("CacheSlide requires one native attention metadata batch")
    request_ids = tuple(runner.input_batch.req_ids)
    if not request_ids:
        return None
    if len(request_ids) != 1:
        raise ValueError("CacheSlide currently supports exactly one scheduled request")
    if not metadata or any(item is None for item in metadata.values()):
        raise ValueError("CacheSlide requires complete attention metadata")
    counts = {int(item.num_actual_tokens) for item in metadata.values()}
    if len(counts) != 1:
        raise ValueError(
            "CacheSlide attention layers disagree on scheduled token count"
        )
    count = counts.pop()
    request_id = request_ids[0]
    request = runner.requests[request_id]
    if (
        inputs_embeds is not None
        or request.prompt_token_ids is None
        or getattr(request, "prompt_embeds", None) is not None
        or getattr(request, "mm_features", ())
        or getattr(request, "lora_request", None) is not None
    ):
        raise ValueError(
            "CacheSlide requires plain token IDs without multimodal or LoRA"
        )
    params = request.sampling_params
    if params is None or getattr(params, "prompt_logprobs", None) is not None:
        raise ValueError("CacheSlide supports generation without prompt logprobs")
    if positions is None or positions.ndim != 1 or not 0 < count <= len(positions):
        raise ValueError(
            "CacheSlide requires one absolute position per scheduled token"
        )
    step = StepContext(
        request_id=request_id,
        prompt_token_ids=tuple(request.prompt_token_ids),
        positions=tuple(positions[:count].detach().cpu().tolist()),
        extra_args=getattr(params, "extra_args", None),
    )
    prompt_length = len(step.prompt_token_ids)
    if step.positions[0] < prompt_length:
        if step.positions != tuple(range(prompt_length)):
            raise ValueError(
                "CacheSlide requires an unchunked, uncached prompt prefill"
            )
    elif len(step.positions) != 1:
        raise ValueError("CacheSlide requires exactly one query token during decode")
    return step


def attach_runner(
    runner: Any, *, context_getter: Callable[[], Any] | None = None
) -> None:
    """Attach to this runner instance without patching native class methods."""
    if getattr(runner, "_cacheslide_adapter_installed", False):
        return
    get_context = context_getter or _native_context
    original_forward = runner._model_forward
    original_cleanup = runner._on_request_state_removed
    request_snapshots: dict[str, StepContext] = {}

    @wraps(original_forward)
    def forward(
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        step = _make_step(runner, positions, inputs_embeds, get_context())
        if step is not None:
            previous = request_snapshots.setdefault(step.request_id, step)
            if (
                step.prompt_token_ids != previous.prompt_token_ids
                or step.extra_args != previous.extra_args
            ):
                raise ValueError(
                    "CacheSlide request inputs cannot change during decoding"
                )
        with step_scope(step):
            result = original_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        if step is not None:
            if not hasattr(result, "shape") or result.shape[0] != len(positions):
                raise RuntimeError(
                    "CacheSlide model must preserve every native scheduled output row"
                )
        return result

    @wraps(original_cleanup)
    def cleanup(request_id, request_state):
        request_snapshots.pop(request_id, None)
        try:
            return original_cleanup(request_id, request_state)
        finally:
            if getattr(runner, "model", None) is not None:
                model = runner.get_model()
                release = getattr(model, "cacheslide_release_request", None)
                if release is None:
                    raise RuntimeError(
                        "CacheSlide model lacks its request cleanup hook"
                    )
                release(request_id)

    runner._model_forward = forward
    runner._on_request_state_removed = cleanup
    runner._cacheslide_adapter_installed = True
