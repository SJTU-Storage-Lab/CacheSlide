import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import cacheslide_sglang
from cacheslide_core.artifacts import AttentionAdapter
from cacheslide_core.context import StepContext, step_scope
from cacheslide_core.position import cope_attention
from cacheslide_core.storage import StaleCompletionError


class TupleLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class NativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_size, self.kv_size = 4, 2
        self.num_heads, self.num_kv_heads, self.head_dim = 2, 1, 2
        self.qkv_proj = TupleLinear(4, 8, bias=False, dtype=torch.float64)
        self.o_proj = TupleLinear(4, 4, bias=False, dtype=torch.float64)
        self.attn = nn.Identity()

    def load_weights(self, weights):
        return {name for name, _ in weights}


class NativeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = NativeAttention()
        self.received_batches = []

    def forward(self, positions, hidden, forward_batch, residual):
        self.received_batches.append(forward_batch)
        return self.self_attn(positions, hidden, forward_batch), hidden


class NativeNorm(nn.Module):
    def forward(self, hidden, residual):
        return hidden + residual, None


class NativeModel(nn.Module):
    def __init__(self, config=None, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.padding_idx, self.vocab_size = 0, 16
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.start_layer, self.end_layer = 0, 2
        self.layers_to_capture = []
        self.embed_tokens = nn.Embedding(16, 4, dtype=torch.float64)
        self.layers = nn.ModuleList([NativeLayer(), NativeLayer()])
        self.norm = NativeNorm()

    def forward(
        self, ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ):
        hidden = self.embed_tokens(ids) if input_embeds is None else input_embeds
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, forward_batch, residual)
        return self.norm(hidden, residual)[0]


class NativeLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.model = self._init_model(config, quant_config, prefix)
        self.capture_aux_hidden_states = False
        self.logits_processor = SimpleNamespace(logit_scale=None)

    def forward(
        self,
        ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        hidden = self.model(
            ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return {
            "native_logits": hidden * self.logits_processor.logit_scale,
            "batch": forward_batch,
        }

    def load_weights(self, weights):
        return {name for name, _ in weights}


@pytest.fixture
def model_module(monkeypatch):
    modules = []
    native = ModuleType("sglang.srt.models.llama")
    native.LlamaAttention, native.LlamaModel, native.LlamaForCausalLM = (
        NativeAttention,
        NativeModel,
        NativeLM,
    )
    memory = ModuleType("sglang.srt.mem_cache.memory_pool")
    memory.MHATokenToKVPool = type("MHATokenToKVPool", (), {})
    forward = ModuleType("sglang.srt.model_executor.forward_context")
    state = SimpleNamespace(batch=None, request=None, pool=None, request_pool=None)
    forward.get_req_to_token_pool = lambda: state.request_pool
    forward.get_token_to_kv_pool = lambda: state.pool
    integration = ModuleType("cacheslide_sglang.integration")
    integration.current_forward_batch = lambda: state.batch
    integration.current_request = lambda: state.request

    @contextmanager
    def bind(ids, positions, batch):
        previous, state.batch = state.batch, batch
        try:
            yield
        finally:
            state.batch = previous

    integration.bind_model_forward = bind
    compat = ModuleType("cacheslide_sglang.compat")
    compat.verify_installed_sglang = lambda: None
    modules.extend((native, memory, forward, integration, compat))
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cacheslide_sglang, "integration", integration, raising=False)
    path = Path(__file__).parents[1] / "src/cacheslide_sglang/models/llama.py"
    spec = importlib.util.spec_from_file_location("_sg_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, state, memory, integration


def adapter():
    config = dict(
        model_type="llama",
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=16,
    )
    result = AttentionAdapter(config, rank=2, max_positions=8).double()
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.normal_(0, 0.1)
    return result


class AttentionRuntime:
    def __init__(self):
        self.observed = []

    def attention(self, layer, positions, q, k, v, trained):
        self.observed.append((layer, positions, q, k, v))
        return cope_attention(q, k, v, trained.cope, positions)


def test_native_projection_identity_gqa_cope_lora_and_loader_preserved(model_module):
    module, state, *_ = model_module
    state.batch = object()
    native, runtime, trained = NativeAttention(), AttentionRuntime(), adapter()
    wrapper = module.CacheSlideAttention(native, 1, runtime, trained)
    assert wrapper.qkv_proj is native.qkv_proj and wrapper.o_proj is native.o_proj
    assert not hasattr(wrapper, "rotary_emb")
    hidden = torch.randn(3, 4, dtype=torch.float64)
    positions = torch.arange(3)
    qkv = native.qkv_proj(hidden)[0] + trained.qkv_delta(hidden)
    q, k, v = qkv.split((4, 2, 2), -1)
    expected_attention = cope_attention(
        q.reshape(3, 2, 2),
        k.reshape(3, 1, 2),
        v.reshape(3, 1, 2),
        trained.cope,
        positions,
    )
    expected_attention = expected_attention.reshape(3, 4)
    expected = native.o_proj(expected_attention)[0] + trained.output_delta(
        expected_attention
    )
    actual = wrapper(positions, hidden, state.batch)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(runtime.observed[0][3], k.reshape(3, 1, 2))
    assert actual.dtype == torch.float64
    assert wrapper.load_weights([("qkv_proj.weight", torch.empty(0))]) == {
        "qkv_proj.weight"
    }
    with pytest.raises(ValueError, match="ForwardBatch"):
        wrapper(positions, hidden, object())


def test_bound_native_layers_keep_parameter_paths_and_exact_forward_batch(
    model_module, monkeypatch
):
    module, state, *_ = model_module
    state.batch = object()
    native = NativeModel()
    original = dict(native.named_parameters())
    adapters = [adapter(), adapter()]
    bundle = SimpleNamespace(layer=lambda i, **kwargs: adapters[i])

    class Runtime(AttentionRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def run(self, model, hidden, positions, step):
            residual = None
            for layer in model.layers:
                hidden, residual = layer(positions, hidden, residual)
            return model.norm(hidden, residual)[0]

    monkeypatch.setattr(module, "CacheSlideRuntime", Runtime)
    model = module.CacheSlideModel(native, bundle, object())
    names = dict(model.named_parameters())
    for name, parameter in original.items():
        assert names[name] is parameter
    assert not any(".native." in name or ".layer." in name for name in names)
    step = StepContext("req", (1, 2, 3), (0, 1, 2))
    with step_scope(step):
        actual = model(torch.tensor([1, 2, 3]), torch.arange(3), state.batch)
    assert actual.shape == (3, 4)
    assert all(layer.received_batches == [state.batch] for layer in native.layers)
    # Native profiling may use repeated dummy positions. An explicit warmup
    # takes the native layer loop with synthetic causal coordinates, no arenas.
    observed_before = len(model.cacheslide_runtime.observed)
    warmup = model(
        torch.tensor([1, 2, 3]), torch.zeros(3, dtype=torch.long), state.batch
    )
    assert warmup.shape == (3, 4)
    for entry in model.cacheslide_runtime.observed[observed_before:]:
        torch.testing.assert_close(entry[1], torch.arange(3))


def test_native_outer_logits_processor_and_load_weights_remain_inherited(
    model_module, monkeypatch
):
    module, state, *_ = model_module
    native = NativeModel()
    monkeypatch.setattr(
        module.CacheSlideLlamaForCausalLM, "_init_model", lambda *args: native
    )
    lm = module.CacheSlideLlamaForCausalLM(SimpleNamespace(logit_scale=2.5))
    assert lm.logits_processor.logit_scale == 2.5
    assert module.CacheSlideLlamaForCausalLM.load_weights is NativeLM.load_weights
    # A small native-forward replacement isolates outer scope restoration and
    # proves logits/sampling are not replaced by a CacheSlide-specific path.
    native.forward = lambda ids, positions, batch, *args, **kwargs: torch.ones(3, 4)
    batch = object()
    result = lm(torch.tensor([1, 2, 3]), torch.arange(3), batch)
    assert result["batch"] is batch and state.batch is None
    torch.testing.assert_close(result["native_logits"], torch.full((3, 4), 2.5))


def test_arena_factory_checks_native_row_generation_and_split_pool_identity(
    model_module,
):
    module, state, memory, *_ = model_module
    state.pool = memory.MHATokenToKVPool()
    pool = state.pool
    pool.kv_cache_layout, pool.use_hnd, pool.is_quantized_kv_cache = "nhd", False, False
    pool.store_dtype = pool.dtype = torch.float32
    pool.page_size = 1
    k, v = torch.zeros(10, 1, 2), torch.zeros(10, 1, 2)
    pool.get_key_buffer = lambda _: k
    pool.get_value_buffer = lambda _: v
    state.request_pool = SimpleNamespace(
        req_to_token=torch.tensor([[0, 0, 0], [4, 8, 3]], dtype=torch.int32),
        req_generation=torch.tensor([0, 7]),
        free_slots=[],
    )
    state.request = SimpleNamespace(
        request_id="req",
        req_pool_index=1,
        req_generation=7,
        runner=SimpleNamespace(
            token_to_kv_pool=pool, req_to_token_pool=state.request_pool
        ),
    )
    state.batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([3]),
        out_cache_loc=torch.tensor([4, 8, 3]),
    )
    step = StepContext("req", (1, 2, 3), (0, 1, 2))
    with step_scope(step):
        arena = module.CacheSlideModel._arena(None, 0, 3, 1)
        assert arena.key_buffer is k and arena.value_buffer is v
        assert arena.snapshot() == {0: 4, 1: 8, 2: 3}
        state.batch.out_cache_loc[0] = 9
        with pytest.raises(ValueError, match="locations"):
            module.CacheSlideModel._arena(None, 0, 3, 1)
        state.batch.out_cache_loc[0] = 4
        state.request_pool.req_generation[1] += 1
        with pytest.raises(StaleCompletionError, match="generation"):
            arena.write_prefill(torch.zeros(3, 1, 2), torch.zeros(3, 1, 2))
