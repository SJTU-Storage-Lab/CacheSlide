import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from cacheslide_sglang import integration as api
from cacheslide_sglang import plugin
from cacheslide_sglang.compat import CompatibilityError, verify_compatibility
from tests.test_sglang_integration import launch as launch
from tests.test_sglang_integration import native_context


def request_fixture():
    context = native_context()
    req = context.request
    req.sampling_params = SimpleNamespace(
        n=1,
        max_new_tokens=1,
        custom_params={
            "cacheslide_plan_json": context.plan.to_json(),
            "cacheslide_plan_sha256": api.plan_digest(context.plan),
            "cacheslide_run_id": "run",
            "cacheslide_request_nonce": "nonce",
            "__req__": req,
        },
    )
    req.finished = lambda: True
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
    events = []
    runtime = api.runtime_for(context.runner)
    runtime.last_metrics.update(
        {
            "cache_hit": False,
            "fallback": False,
            "prompt_tokens": 2,
            "layer_rows": [2],
            "storage": {"live_bytes": 0},
        }
    )
    runtime.release = lambda rid: events.append(("runtime released", rid))
    runtime.close = lambda: events.append("runtime closed")
    return req, SimpleNamespace(model_runner=context.runner), events


def test_worker_binds_real_request_metrics_and_release_precedes_native_reuse(launch):
    req, worker, events = request_fixture()
    result = SimpleNamespace(logits_output=SimpleNamespace(customized_info=None))

    def forward(worker, batch):
        assert api.current_request().request is req
        return result

    assert (
        plugin._worker_forward(forward, worker, SimpleNamespace(reqs=[req])) is result
    )
    assert api.current_request() is None
    assert result.logits_output.customized_info["cacheslide"][0]["prompt_tokens"] == 2
    req.output_ids = [3]

    def free(req, tree):
        assert events == [("runtime released", "rid")]
        req.kv.req_pool_idx = None
        events.append("native freed")

    plugin._release(free, req, object())
    assert req._cacheslide_context.retired
    receipt = api.ReceiptStore(launch.receipt_dir).read("run", "rid", "nonce")
    assert receipt["status"] == "complete" and receipt["resources_released"] is True
    assert receipt["output_ids"] == [3]
    assert receipt["metrics"]["layer_rows"] == [2]


def test_forward_failure_retires_before_native_reuse_and_publishes_failure(launch):
    req, worker, events = request_fixture()

    def forward(*_):
        raise RuntimeError("injected model fault")

    with pytest.raises(RuntimeError, match="model fault"):
        plugin._worker_forward(forward, worker, SimpleNamespace(reqs=[req]))
    assert api.current_request() is None
    assert req._cacheslide_context.retired
    assert events == [("runtime released", "rid")]
    receipt = api.ReceiptStore(launch.receipt_dir).read("run", "rid", "nonce")
    assert receipt["status"] == "failed" and receipt["resources_released"] is False
    plugin._release(lambda *_: events.append("native freed"), req, object())
    plugin._abort(
        lambda *_: None,
        SimpleNamespace(waiting_queue=[req]),
        SimpleNamespace(abort_all=True),
    )
    assert len(events) == 2  # Neither cleanup nor terminal receipt was repeated.


def test_retraction_does_not_publish_success_and_rebinds_new_generation(launch):
    req, worker, events = request_fixture()

    def result(*_):
        return SimpleNamespace(logits_output=SimpleNamespace(customized_info=None))

    plugin._worker_forward(result, worker, SimpleNamespace(reqs=[req]))
    old = req._cacheslide_context
    req.finished = lambda: False
    plugin._release(lambda *_: None, req, object())
    assert old.retired and not old.receipt_published
    worker.model_runner.req_to_token_pool.req_generation[1] = 8
    req.output_ids = [3, 4]
    plugin._worker_forward(result, worker, SimpleNamespace(reqs=[req]))
    assert req._cacheslide_context is not old
    assert req._cacheslide_context.req_generation == 8
    assert not list(Path(launch.receipt_dir).glob("*.json"))


def test_noncache_engine_is_transparent_even_after_hooks_registered(monkeypatch):
    monkeypatch.delenv(api._CONFIG_ENV, raising=False)
    monkeypatch.setattr(plugin, "_REGISTERED", True)
    plugin.register()  # Must not even import native modules.
    value = object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return value

    for callback, args in [
        (plugin._worker_forward, (object(), None)),
        (plugin._release, (object(), object())),
        (plugin._abort, (object(), object())),
        (plugin._warmup, (object(),)),
        (plugin._scheduler_init, (object(), object())),
        (plugin._ready, (object(),)),
        (plugin._close_host, (object(),)),
    ]:
        assert callback(original, *args) is value
    assert len(calls) == 7


def test_invalid_explicit_launch_is_not_silent_passthrough(monkeypatch):
    monkeypatch.setenv(api._CONFIG_ENV, "{}")
    with pytest.raises(CompatibilityError, match="launch"):
        plugin.register()


def test_close_runtime_before_native_host_cleanup(launch):
    req, worker, events = request_fixture()
    plugin._close_host(
        lambda _: events.append("native closed"), SimpleNamespace(model_worker=worker)
    )
    assert events == ["runtime closed", "native closed"]


def source_root():
    root = os.environ.get("CACHESLIDE_SGLANG_SOURCE")
    if not root:
        pytest.skip(
            "set CACHESLIDE_SGLANG_SOURCE to execute audited upstream contracts"
        )
    verify_compatibility(root)
    root = Path(root)
    return root / "python" if (root / "python").is_dir() else root


def source_function(path, name, globals=None):
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        node,
    ]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {} if globals is None else dict(globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_actual_upstream_registry_around_and_import_alias_propagation(monkeypatch):
    source = source_root() / "sglang/srt/plugins/hook_registry.py"
    spec = importlib.util.spec_from_file_location(
        "cacheslide_test_official_hooks", source
    )
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    defining = ModuleType("cacheslide_test_defining")
    importing = ModuleType("cacheslide_test_importing")
    defining.release = lambda value: value + 1
    importing.release = defining.release
    monkeypatch.setitem(sys.modules, defining.__name__, defining)
    monkeypatch.setitem(sys.modules, importing.__name__, importing)

    def callback(original, value):
        return original(value) * 10

    hooks.HookRegistry.register(
        defining.__name__ + ".release", callback, hooks.HookType.AROUND
    )
    hooks.HookRegistry.apply_hooks()
    assert plugin._has_callback(defining.release, callback)
    assert importing.release is defining.release
    assert importing.release(2) == 30
    hooks.HookRegistry.apply_hooks()
    assert importing.release(2) == 30  # once-only application


def test_actual_upstream_release_and_metrics_collector_contract(launch):
    root = source_root() / "sglang/srt"
    req, worker, events = request_fixture()
    result = SimpleNamespace(logits_output=SimpleNamespace(customized_info=None))
    plugin._worker_forward(lambda *_: result, worker, SimpleNamespace(reqs=[req]))
    req.output_ids, req.customized_info = [3], None
    collect = source_function(
        root / "managers/scheduler_components/batch_result_processor.py",
        "_maybe_collect_customized_info",
        {"torch": torch},
    )
    collect(None, 0, req, result.logits_output)
    assert req.customized_info["cacheslide"][0]["request_id"] == "rid"
    assert (
        req.customized_info["cacheslide"][0]
        is not result.logits_output.customized_info["cacheslide"][0]
    )
    req.kv.holds_kv, req.kv.is_kv_released = True, False
    req.kv.kv_allocated_len, req.kv.cache_protected_len = 2, 0
    req.kv.mark_kv_released = lambda: events.append("native marked released")
    req.effective_kv_committed_len = lambda: 2
    tree = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(free=lambda _: events.append("row freed")),
        free_kv_row=lambda *_: events.append("native KV freed"),
    )
    finished = source_function(root / "mem_cache/chunk_cache.py", "cache_finished_req")
    tree.cache_finished_req = lambda req, **kwargs: finished(tree, req, **kwargs)
    release = source_function(
        root / "mem_cache/common.py",
        "release_kv_cache",
        {
            "HybridReqToTokenPool": type("Hybrid", (), {}),
            "_release_overallocated_kv_indices": lambda *_: None,
        },
    )
    plugin._release(release, req, tree)
    assert events == [
        ("runtime released", "rid"),
        "native KV freed",
        "row freed",
        "native marked released",
    ]


def test_actual_upstream_shutdown_rpc_and_generation_signature():
    root = source_root() / "sglang/srt"
    engine = ast.parse((root / "entrypoints/engine.py").read_text())
    methods = {
        node.name: node
        for node in ast.walk(engine)
        if isinstance(node, ast.FunctionDef)
    }
    generate = methods["generate"]
    arguments = {arg.arg for arg in generate.args.args + generate.args.kwonlyargs}
    assert {"input_ids", "rid", "sampling_params", "return_logprob"} <= arguments
    assert "return_text_in_logprobs" not in arguments
    assert "kill_process_tree" in ast.unparse(methods["shutdown"])
    rpc = source_function(
        root / "entrypoints/engine.py",
        "collective_rpc",
        {
            "RpcReqInput": lambda **kwargs: kwargs,
            "RpcReqOutput": dict,
            "sock_send": lambda socket, obj: socket.append(obj),
            "sock_recv": lambda *args, **kwargs: SimpleNamespace(
                success=True, message=""
            ),
            "zmq": SimpleNamespace(BLOCKY=70),
        },
    )
    # Prove the official method routes the requested Scheduler method; skip its
    # final concrete IPC type assertion by supplying that same response class.
    rpc.__globals__["RpcReqOutput"] = SimpleNamespace
    socket = []
    rpc(SimpleNamespace(send_to_rpc=socket), "release_host_resources")
    assert socket == [{"method": "release_host_resources", "parameters": {}}]
