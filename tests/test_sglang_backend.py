import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def backend(monkeypatch):
    base = ModuleType("sglang.srt.layers.attention.base_attn_backend")

    class AttentionBackend:
        def init_forward_metadata(self, batch):
            self.init_forward_metadata_out_graph(batch)
            self.init_forward_metadata_in_graph(batch)

        def init_forward_metadata_in_graph(self, batch):
            pass

    base.AttentionBackend = AttentionBackend
    registry = ModuleType("sglang.srt.layers.attention.attention_registry")
    registry.ATTENTION_BACKENDS = {}

    def register(name):
        def decorate(factory):
            registry.ATTENTION_BACKENDS[name] = factory
            return factory

        return decorate

    registry.register_attention_backend = register
    args = ModuleType("sglang.srt.server_args")
    args.ATTENTION_BACKEND_CHOICES = ["triton"]
    args.add_attention_backend_choices = args.ATTENTION_BACKEND_CHOICES.extend
    for module in (base, registry, args):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).parents[1] / "src/cacheslide_sglang/backend.py"
    spec = importlib.util.spec_from_file_location("_sg_test_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, registry, args


def test_backend_registers_native_factory_idempotently(backend):
    module, registry, args = backend
    module.register_backend()
    module.register_backend()
    assert args.ATTENTION_BACKEND_CHOICES.count("cacheslide") == 1
    runner = SimpleNamespace(req_to_token_pool=object(), token_to_kv_pool=object())
    instance = registry.ATTENTION_BACKENDS["cacheslide"](runner)
    assert instance.req_to_token_pool is runner.req_to_token_pool
    assert instance.token_to_kv_pool is runner.token_to_kv_pool
    instance.init_forward_metadata(SimpleNamespace(batch_size=1))
    assert instance.forward_metadata is None
    registry.ATTENTION_BACKENDS["cacheslide"] = lambda _: None
    with pytest.raises(RuntimeError, match="another plugin"):
        module.register_backend()


def test_backend_fails_closed_if_radix_or_graph_paths_are_used(backend):
    module, *_ = backend
    instance = module.CacheSlideAttentionBackend(
        SimpleNamespace(req_to_token_pool=object(), token_to_kv_pool=object())
    )
    for name in ("forward", "forward_decode", "forward_extend", "forward_mixed"):
        with pytest.raises(RuntimeError, match="RadixAttention"):
            getattr(instance, name)()
    with pytest.raises(ValueError, match="CUDA graphs"):
        instance.init_forward_metadata_out_graph(SimpleNamespace(batch_size=1), True)
    with pytest.raises(ValueError, match="one request"):
        instance.init_forward_metadata(SimpleNamespace(batch_size=2))
