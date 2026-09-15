"""CPU-only worker lifecycle contract; this does not instantiate native CUDA."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

import cacheslide_vllm


@pytest.fixture
def worker_class(monkeypatch):
    native = types.ModuleType("vllm.v1.worker.gpu_worker")

    class NativeWorker:
        def shutdown(self):
            self.native_shutdown_called = True

    native.Worker = NativeWorker
    monkeypatch.setitem(sys.modules, native.__name__, native)
    path = Path(cacheslide_vllm.__file__).with_name("worker.py")
    spec = importlib.util.spec_from_file_location("cacheslide_vllm._worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CacheSlideWorker


def test_worker_metrics_and_shutdown_drain_owned_runtime(worker_class):
    events = []
    runtime = types.SimpleNamespace(
        last_metrics={"cache_hit": True}, close=lambda: events.append("closed")
    )
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(cacheslide_runtime=runtime)
    )
    worker = object.__new__(worker_class)
    worker.model_runner = types.SimpleNamespace(model=model, get_model=lambda: model)
    receipt = worker.cacheslide_metrics()
    receipt["cache_hit"] = False
    assert runtime.last_metrics["cache_hit"]
    worker.shutdown()
    assert events == ["closed"] and worker.native_shutdown_called


def test_native_shutdown_still_runs_if_store_cleanup_fails(worker_class):
    def fail():
        raise OSError("simulated cache I/O failure")

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            cacheslide_runtime=types.SimpleNamespace(close=fail)
        )
    )
    worker = object.__new__(worker_class)
    worker.model_runner = types.SimpleNamespace(model=model, get_model=lambda: model)
    with pytest.raises(OSError, match="simulated"):
        worker.shutdown()
    assert worker.native_shutdown_called


def test_partial_worker_initialization_can_shutdown(worker_class):
    worker = object.__new__(worker_class)
    worker.shutdown()
    assert worker.native_shutdown_called
