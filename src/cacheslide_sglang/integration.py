"""Strict native request binding and the receipt-validated offline Engine API."""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cacheslide_core.config import CacheSlideSettings
from cacheslide_core.context import StepContext, step_scope
from cacheslide_core.contracts import RequestPlan, digest

from .compat import CompatibilityError, validate_engine_config, verify_installed_sglang
from .receipts import ReceiptError, ReceiptStore, canonical_json, validate_result

_CONFIG_ENV = "CACHESLIDE_SGLANG_LAUNCH"
_REQUEST: ContextVar[SGRequestContext | None] = ContextVar("sg_request", default=None)
_BATCH: ContextVar[Any] = ContextVar("sg_forward_batch", default=None)
_WARMUP: ContextVar[bool] = ContextVar("sg_warmup", default=False)
_ENGINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class LaunchConfig:
    model_path: str
    settings: CacheSlideSettings
    receipt_dir: str
    run_id: str

    @property
    def config_digest(self) -> str:
        return digest(asdict(self))


def launch_config() -> LaunchConfig:
    raw = os.environ.get(_CONFIG_ENV, "")
    if not raw or len(raw.encode()) > 65536:
        raise CompatibilityError(
            "missing or oversized trusted CacheSlide launch configuration"
        )
    try:
        value = json.loads(raw)
        if set(value) != {"model_path", "settings", "receipt_dir", "run_id"}:
            raise ValueError("unknown launch fields")
        value["settings"] = CacheSlideSettings.from_mapping(value["settings"])
        result = LaunchConfig(**value)
        if (
            not Path(result.model_path).is_absolute()
            or not Path(result.receipt_dir).is_absolute()
        ):
            raise ValueError("launch paths must be absolute")
        if (
            not isinstance(result.run_id, str)
            or not result.run_id
            or len(result.run_id) > 256
        ):
            raise ValueError("invalid run identity")
        return result
    except (TypeError, ValueError, KeyError) as exc:
        raise CompatibilityError(
            "invalid trusted CacheSlide launch configuration"
        ) from exc


@dataclass
class SGRequestContext:
    request_id: str
    req_pool_index: int
    req_generation: int
    runner: Any
    plan: RequestPlan
    request: Any
    nonce: str
    retired: bool = False
    metrics: dict | None = None
    receipt_published: bool = False


def current_request() -> SGRequestContext | None:
    return _REQUEST.get()


def current_forward_batch():
    batch = _BATCH.get()
    if batch is None:
        raise ValueError("no bound native SGLang ForwardBatch")
    return batch


@contextmanager
def request_scope(request: SGRequestContext):
    token = _REQUEST.set(request)
    try:
        yield request
    finally:
        _REQUEST.reset(token)


@contextmanager
def warmup_scope():
    token = _WARMUP.set(True)
    try:
        yield
    finally:
        _WARMUP.reset(token)


def _ids(values) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError("native token IDs/positions must be nonnegative integers")
    return result


def validate_epoch(context: SGRequestContext) -> None:
    req = context.request
    if context.retired or req.kv.req_pool_idx != context.req_pool_index:
        raise ValueError("stale CacheSlide request slot")
    generation = int(
        context.runner.req_to_token_pool.req_generation[context.req_pool_index]
    )
    if generation != context.req_generation:
        raise ValueError("stale CacheSlide request generation")


@contextmanager
def bind_model_forward(input_ids, positions, forward_batch):
    """Bind the final native batch, after EagerRunner copied its input buffers."""
    context = current_request()
    batch_token = _BATCH.set(forward_batch)
    try:
        if context is None:
            if not _WARMUP.get():
                raise ValueError("unbound native forward is not an authorized warmup")
            with step_scope(None):
                yield None
            return
        validate_epoch(context)
        if _ids(forward_batch.req_pool_indices) != (context.req_pool_index,):
            raise ValueError("native ForwardBatch has the wrong request slot")
        prompt = context.plan.token_ids
        outputs = _ids(context.request.output_ids)
        actual_ids, actual_positions = _ids(input_ids), _ids(positions)
        mode = forward_batch.forward_mode
        replay = None
        if mode.is_decode():
            if not outputs or actual_ids != outputs[-1:]:
                raise ValueError("decode token differs from native request history")
            expected_positions = (len(prompt) + len(outputs) - 1,)
        elif mode.is_extend():
            complete = prompt + outputs
            if actual_ids != complete:
                raise ValueError(
                    "prefill/replay must contain the complete native token history"
                )
            expected_positions = tuple(range(len(complete)))
            if outputs:
                replay = complete
            if _ids(forward_batch.extend_prefix_lens) != (0,):
                raise ValueError("native prefix/chunked prefill is unsupported")
        else:
            raise ValueError("unsupported native forward mode")
        if actual_positions != expected_positions or _ids(forward_batch.seq_lens) != (
            expected_positions[-1] + 1,
        ):
            raise ValueError(
                "native positions/sequence length disagree with request history"
            )
        step = StepContext(
            context.request_id,
            prompt,
            actual_positions,
            {"cacheslide": json.loads(context.plan.to_json())},
            replay,
        )
        with step_scope(step):
            yield step
    finally:
        _BATCH.reset(batch_token)


def runtime_for(runner):
    runtime = getattr(getattr(runner.model, "model", None), "cacheslide_runtime", None)
    if runtime is None:
        raise CompatibilityError("native model has no CacheSlide runtime")
    return runtime


def plan_digest(plan: RequestPlan) -> str:
    return digest([json.loads(plan.to_json()), plan.token_ids])


def publish_receipt(
    context: SGRequestContext, *, status: str, released: bool, error: str | None = None
):
    from .plugin import attestation

    config = launch_config()
    req = context.request
    reason = getattr(req, "finished_reason", None)
    receipt = {
        "schema_version": 1,
        "engine": "sglang",
        "run_id": config.run_id,
        "request_id": context.request_id,
        "nonce": context.nonce,
        "status": status,
        "resources_released": released,
        "req_generation": context.req_generation,
        "config_digest": config.config_digest,
        "plan_digest": plan_digest(context.plan),
        "input_digest": digest(context.plan.token_ids),
        "output_ids": list(getattr(req, "output_ids_through_stop", req.output_ids)),
        "finish_reason": reason.to_json() if reason is not None else None,
        "metrics": context.metrics or {},
        "error": error,
        "attestation": attestation(),
    }
    path = ReceiptStore(config.receipt_dir).publish(receipt)
    context.receipt_published = True
    return path


def _run_scheduler_checked(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    from .plugin import assert_applied, register

    verify_installed_sglang()
    register()
    load_plugins()
    HookRegistry.apply_hooks()
    assert_applied(role="scheduler")
    return run_scheduler_process(*args, **kwargs)


def _release_native_resources(native):
    """Bound the official close RPC before Engine.shutdown kills its workers."""
    import zmq

    socket = native.send_to_rpc
    options = (zmq.SNDTIMEO, zmq.RCVTIMEO)
    previous = {option: socket.getsockopt(option) for option in options}
    try:
        for option in options:
            socket.setsockopt(option, 10000)
        native.collective_rpc("release_host_resources")
    finally:
        for option, value in previous.items():
            socket.setsockopt(option, value)


class CacheSlideEngine:
    def __init__(self, native, config: LaunchConfig):
        self.native = native
        self.config = config
        self.receipts = ReceiptStore(config.receipt_dir)
        self.closed = False
        self._owns_engine_lock = False
        self._generating = threading.Lock()
        from .plugin import attestation

        expected = attestation(config)
        infos = native._scheduler_init_result.scheduler_infos
        if len(infos) != 1 or infos[0].get("cacheslide") != expected:
            raise CompatibilityError(
                "missing or mismatched CacheSlide worker attestation"
            )

    def generate(
        self,
        plan: RequestPlan,
        *,
        max_new_tokens: int = 16,
        request_id: str | None = None,
    ) -> dict:
        if self.closed:
            raise RuntimeError("CacheSlide Engine is closed")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(plan, RequestPlan):
            raise ValueError("generate requires a validated RequestPlan")
        plan = RequestPlan.parse(plan.to_json(), plan.token_ids)
        rid, nonce = request_id or uuid.uuid4().hex, uuid.uuid4().hex
        ReceiptStore._filename(self.config.run_id, rid, nonce)
        if not self._generating.acquire(blocking=False):
            raise RuntimeError("only one request may be active")
        try:
            result = self.native.generate(
                input_ids=list(plan.token_ids),
                rid=rid,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "n": 1,
                    "ignore_eos": True,
                    "custom_params": {
                        "cacheslide_plan_json": plan.to_json(),
                        "cacheslide_plan_sha256": plan_digest(plan),
                        "cacheslide_run_id": self.config.run_id,
                        "cacheslide_request_nonce": nonce,
                    },
                },
                return_logprob=False,
                stream=False,
            )
            receipt = self.receipts.read(self.config.run_id, rid, nonce)
            validate_result(
                receipt,
                result,
                plan_digest=plan_digest(plan),
                input_digest=digest(plan.token_ids),
            )
            from .plugin import attestation

            if receipt.get("config_digest") != self.config.config_digest or receipt.get(
                "attestation"
            ) != attestation(self.config):
                raise ReceiptError("receipt engine/configuration attestation mismatch")
            result.setdefault("meta_info", {})["cacheslide_receipt"] = receipt
            return result
        finally:
            self._generating.release()

    def shutdown(self):
        if not self.closed:
            if not self._generating.acquire(blocking=False):
                raise RuntimeError("cannot shut down an active generation")
            try:
                self.closed = True
                try:
                    _release_native_resources(self.native)
                finally:
                    self.native.shutdown()
            finally:
                if self._owns_engine_lock:
                    self._owns_engine_lock = False
                    _ENGINE_LOCK.release()
                self._generating.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


def create_engine(
    model_path: str | Path,
    settings: CacheSlideSettings,
    *,
    receipt_dir: str | Path,
    dtype: str = "float32",
    context_length: int | None = None,
    max_total_tokens: int | None = None,
    mem_fraction_static: float = 0.5,
) -> CacheSlideEngine:
    """Verify and install every hook before constructing the native Engine."""
    if not isinstance(settings, CacheSlideSettings):
        raise ValueError("settings must be CacheSlideSettings")
    model_path = Path(model_path)
    if not model_path.is_absolute() or not model_path.is_dir():
        raise ValueError("model_path must be an absolute local model directory")
    for name, value in (
        ("context_length", context_length),
        ("max_total_tokens", max_total_tokens),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(name + " must be a positive integer")
    if type(mem_fraction_static) not in (int, float) or not 0 < mem_fraction_static < 1:
        raise ValueError("mem_fraction_static must be between zero and one")
    verify_installed_sglang()
    receipts = ReceiptStore(receipt_dir)
    config = LaunchConfig(
        str(model_path), settings, str(receipts.root), uuid.uuid4().hex
    )
    args = dict(
        model_path=str(model_path),
        dtype=dtype,
        device="cuda",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        nnodes=1,
        max_running_requests=1,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        disable_prefill_cuda_graph=True,
        disable_decode_cuda_graph=True,
        skip_tokenizer_init=True,
        attention_backend="cacheslide",
        prefill_attention_backend="cacheslide",
        decode_attention_backend="cacheslide",
        kv_cache_dtype="auto",
        num_continuous_decode_steps=1,
        mem_fraction_static=mem_fraction_static,
        json_model_override_args=json.dumps(
            {"architectures": ["CacheSlideLlamaForCausalLM"]}
        ),
    )
    if context_length is not None:
        args["context_length"] = context_length
    if max_total_tokens is not None:
        args["max_total_tokens"] = max_total_tokens
    validate_engine_config(SimpleNamespace(**args))
    if not _ENGINE_LOCK.acquire(blocking=False):
        raise RuntimeError("only one CacheSlide SGLang Engine may exist per process")
    environment = {
        _CONFIG_ENV: canonical_json(asdict(config)).decode(),
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "cacheslide_sglang.models",
        "SGLANG_PLUGINS": "cacheslide",
    }
    previous = {key: os.environ.get(key) for key in environment}
    native = None
    try:
        os.environ.update(environment)
        from sglang.srt.plugins import load_plugins
        from sglang.srt.plugins.hook_registry import HookRegistry

        from .plugin import assert_applied, register

        register()
        load_plugins()
        # load_plugins is once-only and swallows registration/apply exceptions.
        # An ordinary Engine may have loaded our inactive plugin earlier.
        HookRegistry.apply_hooks()
        assert_applied(role="parent")
        from sglang.srt.entrypoints.engine import Engine

        class CheckedEngine(Engine):
            run_scheduler_process_func = staticmethod(_run_scheduler_checked)

        native = CheckedEngine(**args)
        engine = CacheSlideEngine(native, config)
        engine._owns_engine_lock = True
        return engine
    except BaseException:
        try:
            if native is not None:
                try:
                    _release_native_resources(native)
                finally:
                    native.shutdown()
        finally:
            _ENGINE_LOCK.release()
        raise
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
