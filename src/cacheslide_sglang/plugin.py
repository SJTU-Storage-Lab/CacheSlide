"""Official SGLang hooks with independently verified installation and cleanup."""

from __future__ import annotations

import json
import os
import pkgutil
from dataclasses import asdict

from cacheslide_core.contracts import RequestPlan, digest

from . import integration as api
from .compat import CompatibilityError, compatibility_manifest, validate_engine_config
from .receipts import canonical_json

_REGISTERED = False


def _active() -> bool:
    # An installed entry point must not alter an ordinary SGLang Engine. Reject
    # malformed explicit launch data rather than interpreting it as opt-out.
    if api._CONFIG_ENV not in os.environ:
        return False
    api.launch_config()
    return True


def attestation(config=None) -> dict:
    config = config or api.launch_config()
    manifest = compatibility_manifest()
    return {
        "adapter": "cacheslide_sglang",
        "protocol": 1,
        "sglang_version": manifest["sglang_version"],
        "source_commit": manifest["sglang_git_commit"],
        "source_digest": digest(manifest["sha256"]),
        "config_digest": digest(asdict(config)),
        "run_id": config.run_id,
        "model_class": "CacheSlideLlamaForCausalLM",
        "backend": "cacheslide",
    }


def _runtime_metrics(context) -> dict:
    metrics = api.runtime_for(context.runner).last_metrics
    if metrics.get("request_id") != context.request_id:
        raise CompatibilityError(
            "native forward did not execute its CacheSlide runtime"
        )
    return json.loads(canonical_json(metrics))


def _worker_forward(original, worker, batch=None, *args, **kwargs):
    if not _active():
        return original(worker, batch, *args, **kwargs)
    if batch is None or len(batch.reqs) != 1:
        raise ValueError("CacheSlide requires one real ScheduleBatch request")
    if api.current_request() is not None:
        raise ValueError("nested native CacheSlide request")
    req = batch.reqs[0]
    config = api.launch_config()
    custom = getattr(req.sampling_params, "custom_params", None)
    required = {
        "cacheslide_plan_json",
        "cacheslide_plan_sha256",
        "cacheslide_run_id",
        "cacheslide_request_nonce",
    }
    if not isinstance(custom, dict) or set(custom) - {"__req__"} != required:
        raise ValueError(
            "native request requires the exact CacheSlide metadata envelope"
        )
    if custom["cacheslide_run_id"] != config.run_id:
        raise ValueError("request belongs to another CacheSlide launch")
    metadata = custom["cacheslide_plan_json"]
    if not isinstance(metadata, str):
        raise ValueError("request plan must be bounded encoded JSON")
    plan = RequestPlan.parse(metadata, api._ids(req.origin_input_ids))
    if api.plan_digest(plan) != custom["cacheslide_plan_sha256"]:
        raise ValueError("native request plan digest mismatch")
    nonce = custom["cacheslide_request_nonce"]
    api.ReceiptStore._filename(config.run_id, req.rid, nonce)
    for name in (
        "input_embeds",
        "session",
        "multimodal_inputs",
        "positional_embed_overrides",
        "lora_id",
        "beam_group",
    ):
        if getattr(req, name, None) is not None:
            raise ValueError(name + " is unsupported by CacheSlide")
    if (
        getattr(req.sampling_params, "n", 1) != 1
        or getattr(req.sampling_params, "max_new_tokens", 0) < 1
    ):
        raise ValueError(
            "only one generated sequence with positive length is supported"
        )
    runner = worker.model_runner
    row = req.kv.req_pool_idx
    if type(row) is not int or row <= 0:
        raise ValueError("CacheSlide requires a live native request row")
    generation = int(runner.req_to_token_pool.req_generation[row])
    context = getattr(req, "_cacheslide_context", None)
    if context is None or context.retired:
        context = api.SGRequestContext(
            req.rid, row, generation, runner, plan, req, nonce
        )
        req._cacheslide_context = context
    elif (
        context.request_id,
        context.req_pool_index,
        context.req_generation,
        context.nonce,
        context.plan,
    ) != (req.rid, row, generation, nonce, plan):
        raise ValueError("native request identity changed without retirement")
    api.validate_epoch(context)
    with api.request_scope(context):
        try:
            result = original(worker, batch, *args, **kwargs)
            context.metrics = _runtime_metrics(context)
            logits = result.logits_output
            if logits.customized_info is None:
                logits.customized_info = {}
            if "cacheslide" in logits.customized_info:
                raise ValueError("conflicting CacheSlide metric producer")
            logits.customized_info["cacheslide"] = [context.metrics]
            return result
        except BaseException as exc:
            runtime = api.runtime_for(runner)
            runtime.release(context.request_id)
            context.retired = True
            context.metrics = dict(runtime.last_metrics)
            api.publish_receipt(
                context, status="failed", released=False, error=type(exc).__name__
            )
            raise


def _release(original, req, tree_cache, *args, **kwargs):
    if not _active():
        return original(req, tree_cache, *args, **kwargs)
    context = getattr(req, "_cacheslide_context", None)
    if context is None:
        return original(req, tree_cache, *args, **kwargs)
    if context.retired:
        # A failed forward already invalidated its own state; native still owns KV.
        return original(req, tree_cache, *args, **kwargs)
    api.validate_epoch(context)
    runtime = api.runtime_for(context.runner)
    # Drain/invalidate request-owned operations BEFORE native token-slot recycling.
    runtime.release(context.request_id)
    context.metrics = dict(runtime.last_metrics)
    context.retired = True
    try:
        result = original(req, tree_cache, *args, **kwargs)
    except BaseException as exc:
        api.publish_receipt(
            context, status="failed", released=False, error=type(exc).__name__
        )
        raise
    if req.finished():
        reason = req.finished_reason.to_json()
        status = "cancelled" if reason.get("type") == "abort" else "complete"
        api.publish_receipt(context, status=status, released=True)
    # A nonterminal release is retraction: discard slot maps, not request history.
    return result


def _abort(original, scheduler, recv_req):
    if not _active():
        return original(scheduler, recv_req)
    # Queued aborts do not pass release_kv_cache. Running aborts must not retire
    # early: native marks them and performs its final forward/release later.
    queued = [
        req
        for req in scheduler.waiting_queue
        if recv_req.abort_all or req.rid.startswith(recv_req.rid)
    ]
    result = original(scheduler, recv_req)
    for req in queued:
        context = getattr(req, "_cacheslide_context", None)
        if context is not None and context.retired and not context.receipt_published:
            api.publish_receipt(context, status="cancelled", released=True)
    return result


def _warmup(original, *args, **kwargs):
    if not _active():
        return original(*args, **kwargs)
    with api.warmup_scope():
        return original(*args, **kwargs)


def _scheduler_init(original, scheduler, server_args, *args, **kwargs):
    if not _active():
        return original(scheduler, server_args, *args, **kwargs)
    validate_engine_config(server_args)
    assert_applied(role="scheduler")
    return original(scheduler, server_args, *args, **kwargs)


def _ready(original, scheduler):
    if not _active():
        return original(scheduler)
    assert_applied(role="scheduler")
    result = original(scheduler)
    runner = scheduler.model_worker.model_runner
    api.runtime_for(runner)
    from .backend import CacheSlideAttentionBackend

    if not isinstance(runner.attn_backend, CacheSlideAttentionBackend):
        raise CompatibilityError("worker is not using the CacheSlide backend")
    return {**result, "cacheslide": attestation()}


def _close_host(original, scheduler):
    if not _active():
        return original(scheduler)
    try:
        api.runtime_for(scheduler.model_worker.model_runner).close()
    finally:
        result = original(scheduler)
    return result


_HOOKS = {
    (
        "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation"
    ): _worker_forward,
    "sglang.srt.mem_cache.common.release_kv_cache": _release,
    "sglang.srt.managers.scheduler.Scheduler.abort_request": _abort,
    "sglang.srt.managers.scheduler.Scheduler.__init__": _scheduler_init,
    "sglang.srt.managers.scheduler.Scheduler.get_init_info": _ready,
    "sglang.srt.managers.scheduler.Scheduler.release_host_resources": _close_host,
    "sglang.srt.model_executor.runner.base_runner.BaseRunner.warmup": _warmup,
}


def register():
    """Register through the official plugin API; perform no GPU initialization."""
    global _REGISTERED
    if not _active():
        return
    if _REGISTERED:
        return
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .backend import register_backend
    from .models.llama import CacheSlideLlamaForCausalLM

    register_backend()
    existing = ModelRegistry.models.get("CacheSlideLlamaForCausalLM")
    if existing is None:
        ModelRegistry.register("cacheslide_sglang.models", strict=True)
    elif existing is not CacheSlideLlamaForCausalLM:
        raise CompatibilityError("CacheSlide model registration conflicts")
    for target, callback in _HOOKS.items():
        HookRegistry.register(target, callback, HookType.AROUND)
    _REGISTERED = True


def _has_callback(function, callback) -> bool:
    seen = set()
    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        for cell in getattr(function, "__closure__", None) or ():
            try:
                if cell.cell_contents is callback:
                    return True
            except ValueError:
                pass
        function = getattr(function, "__wrapped__", None)
    return False


def assert_applied(*, role: str):
    """Check actual wrappers, not the loader flag (which also marks failures)."""
    if role not in {"parent", "scheduler"}:
        raise ValueError("unknown installation role")
    if not _REGISTERED:
        raise CompatibilityError("CacheSlide plugin did not register")
    for target, callback in _HOOKS.items():
        try:
            applied = _has_callback(pkgutil.resolve_name(target), callback)
        except (ImportError, AttributeError, ValueError):
            applied = False
        if not applied:
            raise CompatibilityError("CacheSlide hook was not applied: " + target)
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.models.registry import ModelRegistry

    from .models.llama import CacheSlideLlamaForCausalLM

    if (
        ModelRegistry.models.get("CacheSlideLlamaForCausalLM")
        is not CacheSlideLlamaForCausalLM
    ):
        raise CompatibilityError("CacheSlide model registration missing")
    factory = ATTENTION_BACKENDS.get("cacheslide")
    if factory is None or factory.__module__ != "cacheslide_sglang.backend":
        raise CompatibilityError("CacheSlide attention backend registration missing")
