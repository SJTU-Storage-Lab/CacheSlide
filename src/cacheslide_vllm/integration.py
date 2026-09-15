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


def _native_context():
    from vllm.forward_context import get_forward_context

    return get_forward_context()


def _request_inputs(request: Any, inputs_embeds: Any = None):
    if (
        inputs_embeds is not None
        or request.prompt_token_ids is None
        or getattr(request, "prompt_embeds", None) is not None
        or getattr(request, "mm_features", ())
        or getattr(request, "lora_request", None) is not None
        or any(
            value is not True
            for value in (getattr(request, "prompt_is_token_ids", None) or ())
        )
    ):
        raise ValueError(
            "CacheSlide requires plain token IDs without multimodal or LoRA"
        )
    params = request.sampling_params
    if params is None or getattr(params, "prompt_logprobs", None) is not None:
        raise ValueError("CacheSlide supports generation without prompt logprobs")
    return tuple(request.prompt_token_ids), getattr(params, "extra_args", None)


def _validate_step_positions(step: StepContext) -> None:
    prompt_length = len(step.prompt_token_ids)
    if step.positions[0] < prompt_length:
        length = len(step.replay_token_ids or step.prompt_token_ids)
        if step.positions != tuple(range(length)):
            raise ValueError(
                "CacheSlide requires an unchunked, uncached prompt prefill"
            )
    elif len(step.positions) != 1:
        raise ValueError("CacheSlide requires exactly one query token during decode")


def _make_step(
    runner: Any, positions: Any, inputs_embeds: Any, context: Any, input_ids=None
):
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
    prompt, extras = _request_inputs(request, inputs_embeds)
    if positions is None or positions.ndim != 1 or not 0 < count <= len(positions):
        raise ValueError(
            "CacheSlide requires one absolute position per scheduled token"
        )
    query_positions = tuple(positions[:count].detach().cpu().tolist())
    replay = None
    if query_positions[0] == 0 and count > len(prompt):
        replay = prompt + tuple(getattr(request, "output_token_ids", ()))
        if (
            input_ids is None
            or tuple(input_ids[:count].detach().cpu().tolist()) != replay
        ):
            raise ValueError(
                "native replay input IDs disagree with the request history"
            )
    step = StepContext(
        request_id=request_id,
        prompt_token_ids=prompt,
        positions=query_positions,
        extra_args=extras,
        replay_token_ids=replay,
    )
    _validate_step_positions(step)
    return step


def attach_runner(
    runner: Any,
    *,
    context_getter: Callable[[], Any] | None = None,
    use_v2: bool | None = None,
) -> None:
    """Attach to this runner instance without patching native class methods."""
    if getattr(runner, "_cacheslide_adapter_installed", False):
        return
    v1 = all(
        callable(getattr(runner, name, None))
        for name in ("_model_forward", "_on_request_state_removed")
    )
    v2 = all(
        callable(getattr(runner, name, None))
        for name in (
            "execute_model",
            "prepare_inputs",
            "add_requests",
            "_remove_request",
        )
    )
    if use_v2 is None:
        if v1 == v2:
            raise ValueError("cannot identify the audited native model runner")
        use_v2 = v2
    if (use_v2 and not v2) or (not use_v2 and not v1):
        raise ValueError("native model runner does not match the configured interface")
    if use_v2:
        _attach_v2_runner(runner)
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
        step = _make_step(runner, positions, inputs_embeds, get_context(), input_ids)
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
    runner._cacheslide_runner_version = "v1"


def _release_request(runner: Any, request_id: str) -> None:
    if getattr(runner, "model", None) is not None:
        release = getattr(runner.get_model(), "cacheslide_release_request", None)
        if release is None:
            raise RuntimeError("CacheSlide model lacks its request cleanup hook")
        release(request_id)


def _attach_v2_runner(runner: Any) -> None:
    """Bind V2's actual input-batch and removal paths on this instance only."""
    original_add = runner.add_requests
    original_remove = runner._remove_request
    original_prepare = runner.prepare_inputs
    original_execute = runner.execute_model
    snapshots: dict[str, tuple[StepContext, tuple[int, ...]]] = {}
    executing = False
    dummy = False

    @wraps(original_remove)
    def remove(request_id):
        snapshots.pop(request_id, None)
        try:
            return original_remove(request_id)
        finally:
            _release_request(runner, request_id)

    @wraps(original_add)
    def add(scheduler_output):
        additions = {}
        for request in scheduler_output.scheduled_new_reqs:
            prompt, extras = _request_inputs(request)
            prefill = tuple(request.prefill_token_ids or ())
            if (
                not prompt
                or prefill[: len(prompt)] != prompt
                or any(type(value) is not int or value < 0 for value in prefill)
                or request.num_computed_tokens != 0
            ):
                raise ValueError(
                    "V2 requires complete uncached token-ID prefill history"
                )
            snapshot = StepContext(
                request.req_id, prompt, tuple(range(len(prompt))), extras
            )
            additions[request.req_id] = snapshot, prefill
        result = original_add(scheduler_output)
        # Native add_requests invokes _remove_request before installing a new
        # generation, so publish only after that removal and native add succeed.
        snapshots.update(additions)
        return result

    @wraps(original_prepare)
    def prepare(scheduler_output, batch_req_state, batch_desc):
        batch = original_prepare(scheduler_output, batch_req_state, batch_desc)
        if not executing or dummy:
            raise RuntimeError("V2 real input preparation escaped its execution scope")
        if batch.num_reqs != 1 or len(batch.req_ids) != 1:
            raise ValueError(
                "CacheSlide currently supports exactly one scheduled request"
            )
        if (
            batch.num_draft_tokens != 0
            or batch.num_tokens_after_padding != batch.num_tokens
            or batch.positions.ndim != 1
            or len(batch.positions) != batch.num_tokens
            or batch.input_ids.ndim != 1
            or len(batch.input_ids) != batch.num_tokens
        ):
            raise ValueError(
                "CacheSlide requires unpadded, nonspeculative V2 token rows"
            )
        snapshot, prefill = snapshots[batch.req_ids[0]]
        positions = tuple(batch.positions.detach().cpu().tolist())
        replay = None
        if positions and positions[0] == 0:
            if positions != tuple(range(len(prefill))):
                raise ValueError(
                    "CacheSlide requires an unchunked, uncached prompt prefill"
                )
            if tuple(batch.input_ids.detach().cpu().tolist()) != prefill:
                raise ValueError(
                    "native prefill input IDs disagree with the request history"
                )
            if len(prefill) > len(snapshot.prompt_token_ids):
                replay = prefill
        step = StepContext(
            snapshot.request_id,
            snapshot.prompt_token_ids,
            positions,
            snapshot.extra_args,
            replay_token_ids=replay,
        )
        _validate_step_positions(step)
        _CURRENT_STEP.set(step)
        return batch

    @wraps(original_execute)
    def execute(
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        context_len=0,
    ):
        nonlocal executing, dummy
        if executing:
            raise RuntimeError("CacheSlide does not allow concurrent V2 execution")
        executing, dummy = True, dummy_run
        try:
            with step_scope(None):
                return original_execute(
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    is_profile,
                    context_len,
                )
        finally:
            executing = dummy = False

    runner.add_requests = add
    runner._remove_request = remove
    runner.prepare_inputs = prepare
    runner.execute_model = execute
    runner._cacheslide_adapter_installed = True
    runner._cacheslide_runner_version = "v2"
