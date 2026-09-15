import json
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import pytest
import torch

from cacheslide_core.config import CacheSlideSettings
from cacheslide_core.context import current_step
from cacheslide_core.contracts import RequestPlan
from cacheslide_sglang import integration as api
from cacheslide_sglang import plugin
from cacheslide_sglang.compat import CompatibilityError


def make_plan():
    return RequestPlan.parse(
        {
            "version": 1,
            "operation": "recompute",
            "namespace": "test",
            "task_id": "case",
            "chunks": [{"id": "fixed", "role": "reuse", "start": 0, "end": 2}],
        },
        [1, 2],
    )


@pytest.fixture
def launch(tmp_path, monkeypatch):
    config = api.LaunchConfig(
        str(tmp_path),
        CacheSlideSettings(str(tmp_path / "artifact"), str(tmp_path / "cache")),
        str(tmp_path / "receipts"),
        "run",
    )
    monkeypatch.setenv(api._CONFIG_ENV, json.dumps(asdict(config)))
    return config


def native_context(plan=None):
    plan = plan or make_plan()
    req = SimpleNamespace(
        rid="rid",
        origin_input_ids=list(plan.token_ids),
        output_ids=[],
        kv=SimpleNamespace(req_pool_idx=1),
    )
    runtime = SimpleNamespace(last_metrics={"request_id": "rid"})
    runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_generation=[0, 7]),
        model=SimpleNamespace(model=SimpleNamespace(cacheslide_runtime=runtime)),
    )
    return api.SGRequestContext("rid", 1, 7, runner, plan, req, "nonce")


def batch(*, decode=False, positions=(0, 1)):
    return SimpleNamespace(
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([positions[-1] + 1]),
        extend_prefix_lens=torch.tensor([0]),
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_extend=lambda: not decode
        ),
    )


def test_prefill_decode_replay_and_finally_cleanup():
    context = native_context()
    with api.request_scope(context):
        final_batch = batch()
        with api.bind_model_forward(
            torch.tensor([1, 2]), torch.tensor([0, 1]), final_batch
        ) as step:
            assert step is current_step()
            assert api.current_forward_batch() is final_batch
        context.request.output_ids = [3]
        with api.bind_model_forward(
            torch.tensor([3]), torch.tensor([2]), batch(decode=True, positions=(2,))
        ) as step:
            assert step.positions == (2,)
        with api.bind_model_forward(
            torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), batch(positions=(0, 1, 2))
        ) as step:
            assert step.replay_token_ids == (1, 2, 3)
        with pytest.raises(RuntimeError, match="fault"):
            with api.bind_model_forward(
                torch.tensor([3]), torch.tensor([2]), batch(decode=True, positions=(2,))
            ):
                raise RuntimeError("fault")
        assert current_step() is None
    assert api.current_request() is None
    with pytest.raises(ValueError, match="no bound"):
        api.current_forward_batch()


def test_unbound_forward_requires_explicit_warmup():
    with pytest.raises(ValueError, match="authorized warmup"):
        with api.bind_model_forward([1], [0], batch(positions=(0,))):
            pass
    with (
        api.warmup_scope(),
        api.bind_model_forward([1], [0], batch(positions=(0,))) as step,
    ):
        assert step is None


@pytest.mark.parametrize("fault", ["token", "position", "slot", "generation", "prefix"])
def test_native_request_validation(fault):
    context, forward = native_context(), batch()
    ids, positions = [1, 2], [0, 1]
    if fault == "token":
        ids = [1, 9]
    elif fault == "position":
        positions = [1, 2]
    elif fault == "slot":
        forward.req_pool_indices = torch.tensor([2])
    elif fault == "generation":
        context.runner.req_to_token_pool.req_generation[1] = 8
    else:
        forward.extend_prefix_lens = torch.tensor([1])
    with api.request_scope(context), pytest.raises(ValueError):
        with api.bind_model_forward(ids, positions, forward):
            pass
    assert current_step() is None


def fake_plugins(monkeypatch):
    for name in ("sglang", "sglang.srt", "sglang.srt.plugins"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["sglang.srt.plugins"].load_plugins = lambda: None
    module = ModuleType("sglang.srt.plugins.hook_registry")
    module.HookRegistry = SimpleNamespace(apply_hooks=lambda: None)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_parent_rejects_unapplied_plugin_before_engine_import(launch, monkeypatch):
    fake_plugins(monkeypatch)
    monkeypatch.setattr(api, "verify_installed_sglang", lambda: {})
    monkeypatch.setattr(plugin, "register", lambda: None)

    def fail(**kwargs):
        raise CompatibilityError("unapplied")

    monkeypatch.setattr(plugin, "assert_applied", fail)
    with pytest.raises(CompatibilityError, match="unapplied"):
        api.create_engine(
            launch.model_path, launch.settings, receipt_dir=launch.receipt_dir
        )
    assert "sglang.srt.entrypoints.engine" not in sys.modules
    assert not api._ENGINE_LOCK.locked()
    assert api.launch_config() == launch


def test_child_rejects_unapplied_plugin_before_native_entry(monkeypatch):
    fake_plugins(monkeypatch)
    module = ModuleType("sglang.srt.managers.scheduler")
    calls = []
    module.run_scheduler_process = lambda *a, **k: calls.append("native")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(api, "verify_installed_sglang", lambda: {})
    monkeypatch.setattr(plugin, "register", lambda: None)

    def fail(**kwargs):
        raise CompatibilityError("child missing")

    monkeypatch.setattr(plugin, "assert_applied", fail)
    with pytest.raises(CompatibilityError, match="child missing"):
        api._run_scheduler_checked()
    assert calls == []


def test_missing_worker_attestation_is_rejected(launch):
    native = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[{"status": "ready"}])
    )
    with pytest.raises(CompatibilityError, match="attestation"):
        api.CacheSlideEngine(native, launch)


def test_generated_ids_and_real_receipt_roundtrip(launch, monkeypatch):
    monkeypatch.setattr(api, "_release_native_resources", lambda _: None)
    plan = make_plan()

    def generate(**kwargs):
        assert kwargs["input_ids"] == [1, 2]
        assert kwargs["return_logprob"] is False
        assert kwargs["sampling_params"]["ignore_eos"] is True
        context = native_context(plan)
        context.request_id = kwargs["rid"]
        context.nonce = kwargs["sampling_params"]["custom_params"][
            "cacheslide_request_nonce"
        ]
        context.request.output_ids = [6]
        context.metrics = {"request_id": context.request_id, "cache_hit": False}
        api.publish_receipt(context, status="complete", released=True)
        return {"output_ids": [6], "meta_info": {"id": kwargs["rid"]}}

    native = SimpleNamespace(
        generate=generate,
        shutdown=lambda: None,
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"cacheslide": plugin.attestation(launch)}]
        ),
    )
    engine = api.CacheSlideEngine(native, launch)
    result = engine.generate(plan, max_new_tokens=1)
    assert result["meta_info"]["cacheslide_receipt"]["resources_released"] is True
    engine.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate(plan)


@pytest.mark.parametrize("fail_rpc", [False, True])
def test_shutdown_bounds_rpc_releases_lock_and_always_kills_native(
    launch, monkeypatch, fail_rpc
):
    zmq = ModuleType("zmq")
    zmq.SNDTIMEO, zmq.RCVTIMEO = 1, 2
    monkeypatch.setitem(sys.modules, "zmq", zmq)
    options, calls = {1: -1, 2: -1}, []
    socket = SimpleNamespace(
        getsockopt=options.__getitem__, setsockopt=options.__setitem__
    )

    def rpc(method):
        assert method == "release_host_resources"
        assert options == {1: 10000, 2: 10000}
        calls.append("rpc")
        if fail_rpc:
            raise TimeoutError("worker unavailable")
        calls.append("writer lock released")

    native = SimpleNamespace(
        send_to_rpc=socket,
        collective_rpc=rpc,
        shutdown=lambda: calls.append("native killed"),
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"cacheslide": plugin.attestation(launch)}]
        ),
    )
    engine = api.CacheSlideEngine(native, launch)
    assert api._ENGINE_LOCK.acquire(blocking=False)
    engine._owns_engine_lock = True
    if fail_rpc:
        with pytest.raises(TimeoutError, match="worker unavailable"):
            engine.shutdown()
    else:
        engine.shutdown()
    assert calls[-1] == "native killed"
    assert options == {1: -1, 2: -1}
    assert not api._ENGINE_LOCK.locked()
    engine.shutdown()
    assert calls.count("native killed") == 1


def test_create_explicitly_applies_hooks_after_already_loaded_plugin(
    launch, monkeypatch
):
    fake_plugins(monkeypatch)
    calls = []
    monkeypatch.setattr(
        api, "verify_installed_sglang", lambda: calls.append("verified")
    )
    monkeypatch.setattr(
        api, "_release_native_resources", lambda _: calls.append("closed")
    )
    monkeypatch.setattr(plugin, "register", lambda: calls.append("registered"))
    registry = sys.modules["sglang.srt.plugins.hook_registry"].HookRegistry
    registry.apply_hooks = lambda: calls.append("applied")

    def check(**kwargs):
        assert calls[-1] == "applied"
        calls.append("asserted")

    monkeypatch.setattr(plugin, "assert_applied", check)
    module = ModuleType("sglang.srt.entrypoints.engine")

    class Engine:
        def __init__(self, **kwargs):
            assert calls[-1] == "asserted"
            assert kwargs["skip_tokenizer_init"] is True
            assert kwargs["attention_backend"] == "cacheslide"
            self._scheduler_init_result = SimpleNamespace(
                scheduler_infos=[{"cacheslide": plugin.attestation()}]
            )

        def shutdown(self):
            calls.append("native killed")

    module.Engine = Engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = api.create_engine(
        launch.model_path, launch.settings, receipt_dir=launch.receipt_dir
    )
    assert api.launch_config() == launch
    engine.shutdown()
    assert calls == [
        "verified",
        "registered",
        "applied",
        "asserted",
        "closed",
        "native killed",
    ]


def test_shutdown_cannot_interrupt_generation(launch):
    native = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"cacheslide": plugin.attestation(launch)}]
        )
    )
    engine = api.CacheSlideEngine(native, launch)
    engine._generating.acquire()
    try:
        with pytest.raises(RuntimeError, match="active generation"):
            engine.shutdown()
        assert not engine.closed
    finally:
        engine._generating.release()
