"""CPU contract tests; these never import or execute the native CUDA engine."""

from __future__ import annotations

import ast
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cacheslide_vllm.compat import (
    CompatibilityError,
    compatibility_manifest,
    validate_engine_config,
    verify_vllm_sources,
)
from cacheslide_vllm.integration import (
    StepContext,
    attach_runner,
    current_step,
    step_scope,
)
from cacheslide_vllm.plugin import register


def _config():
    return SimpleNamespace(
        use_v2_model_runner=False,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["CacheSlideLlamaForCausalLM"]),
            enforce_eager=True,
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1, async_scheduling=False, enable_chunked_prefill=False
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode="NONE"),
        attention_config=SimpleNamespace(backend="FLASH_ATTN"),
    )


def test_engine_gates_explicitly_select_v1_without_mutating_configuration(monkeypatch):
    config = _config()
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    with pytest.raises(CompatibilityError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        validate_engine_config(config)
    assert not config.use_v2_model_runner
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    validate_engine_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("scheduler_config", "max_num_seqs", 2, "max_num_seqs"),
        ("parallel_config", "tensor_parallel_size", 2, "tensor_parallel_size"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_size"),
        ("scheduler_config", "async_scheduling", True, "async_scheduling"),
        ("scheduler_config", "enable_chunked_prefill", True, "chunked prefill"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        ("cache_config", "cache_dtype", "fp8", "quantized KV"),
        ("cache_config", "cache_dtype", "float16", "must match the model dtype"),
        ("model_config", "quantization", "awq", "quantized model"),
        ("compilation_config", "mode", 3, "compilation mode"),
        ("compilation_config", "cudagraph_mode", "FULL", "CUDA graph"),
        ("attention_config", "backend", None, "FLASH_ATTN"),
        (None, "lora_config", object(), "lora_config"),
        (None, "speculative_config", object(), "speculative_config"),
        (None, "kv_transfer_config", object(), "kv_transfer_config"),
    ],
)
def test_reject_modes_that_invalidate_sparse_request_or_cache_semantics(
    monkeypatch, section, field, value, message
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config = _config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(CompatibilityError, match=message):
        validate_engine_config(config)


def test_request_context_detaches_metadata_and_restores_after_error():
    extras = {"cacheslide": '{"chunks":[]}', "other": {"rows": [1, 2]}}
    step = StepContext("req", (4, 5), (0, 1), extras)
    extras["other"]["rows"].append(3)
    assert step.extra_args["other"]["rows"] == (1, 2)
    with pytest.raises(TypeError):
        step.extra_args["other"]["rows"] = ()
    assert step.last_query_position == 1
    with step_scope(step):
        with pytest.raises(RuntimeError, match="forward failed"), step_scope(None):
            assert current_step() is None
            raise RuntimeError("forward failed")
        assert current_step() is step
    assert current_step() is None


@pytest.mark.parametrize("value", [float("nan"), 2**65, object(), {1: "bad"}])
def test_metadata_is_finite_and_transportable(value):
    with pytest.raises(ValueError):
        StepContext("req", (1,), (0,), {"cacheslide": value})


class _Model:
    def __init__(self):
        self.released = []

    def cacheslide_release_request(self, request_id):
        self.released.append(request_id)


class _Runner:
    def __init__(self):
        self.input_batch = SimpleNamespace(req_ids=["req"])
        self.requests = {
            "req": SimpleNamespace(
                prompt_token_ids=[4, 5],
                sampling_params=SimpleNamespace(
                    prompt_logprobs=None, extra_args={"cacheslide": "{}"}
                ),
            )
        }
        self.model = _Model()
        self.observed = []
        self.cleanup_calls = []
        self.fail = False
        self.sparse_output = False

    def _model_forward(self, *, positions, **kwargs):
        self.observed.append(current_step())
        if self.fail:
            raise RuntimeError("native forward failed")
        count = 1 if self.sparse_output else len(positions)
        return torch.zeros(count, 3)

    def _on_request_state_removed(self, request_id, state):
        self.cleanup_calls.append((request_id, state))

    def get_model(self):
        return self.model


def _context(count=2):
    return SimpleNamespace(
        attn_metadata={
            "model.layers.0.self_attn.attn": SimpleNamespace(num_actual_tokens=count)
        }
    )


def test_runner_preserves_native_rows_scopes_request_and_does_not_patch_class():
    original = _Runner._model_forward
    runner, untouched = _Runner(), _Runner()
    context = _context()
    attach_runner(runner, context_getter=lambda: context)
    attach_runner(runner, context_getter=lambda: context)
    assert runner._model_forward(positions=torch.arange(2)).shape == (2, 3)
    assert runner.observed[0].request_id == "req"
    assert runner.observed[0].current_positions == (0, 1)
    assert current_step() is None
    assert _Runner._model_forward is original
    untouched._model_forward(positions=torch.arange(2))
    assert untouched.observed == [None]
    runner._on_request_state_removed("req", runner.requests["req"])
    assert runner.model.released == ["req"]
    assert len(runner.cleanup_calls) == 1


def test_cleanup_before_model_load_and_profiling_do_not_require_request_state():
    runner = _Runner()
    runner.model = None
    attach_runner(runner, context_getter=lambda: SimpleNamespace(attn_metadata=None))
    runner._on_request_state_removed("aborted", None)
    runner._model_forward(positions=torch.arange(2))
    assert runner.observed == [None]


def test_failed_forward_clears_request_context_and_sparse_output_is_rejected():
    runner = _Runner()
    attach_runner(runner, context_getter=_context)
    runner.fail = True
    with pytest.raises(RuntimeError, match="native forward failed"):
        runner._model_forward(positions=torch.arange(2))
    assert current_step() is None
    runner.fail, runner.sparse_output = False, True
    with pytest.raises(RuntimeError, match="preserve every native"):
        runner._model_forward(positions=torch.arange(2))


def test_decode_keeps_absolute_position_and_incomplete_prefill_is_rejected():
    runner = _Runner()
    attach_runner(runner, context_getter=lambda: _context(1))
    runner._model_forward(positions=torch.tensor([2]))
    assert runner.observed[-1].last_query_position == 2
    with pytest.raises(ValueError, match="unchunked, uncached"):
        runner._model_forward(positions=torch.tensor([1]))


def test_request_metadata_is_immutable_until_finish_then_id_can_be_reused():
    runner = _Runner()
    context = _context()
    attach_runner(runner, context_getter=lambda: context)
    runner._model_forward(positions=torch.arange(2))
    context.attn_metadata[next(iter(context.attn_metadata))].num_actual_tokens = 1
    params = runner.requests["req"].sampling_params
    params.extra_args["cacheslide"] = '{"changed":true}'
    with pytest.raises(ValueError, match="cannot change during decoding"):
        runner._model_forward(positions=torch.tensor([2]))
    runner._on_request_state_removed("req", runner.requests["req"])
    runner._model_forward(positions=torch.tensor([2]))
    assert runner.observed[-1].extra_args["cacheslide"] == '{"changed":true}'


def test_registry_registration_is_lazy_and_preserves_native_architecture(monkeypatch):
    native = object()
    entries = {"LlamaForCausalLM": native}
    registry = SimpleNamespace(
        get_supported_archs=lambda: entries.keys(),
        register_model=lambda key, value: entries.__setitem__(key, value),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    register()
    register()
    custom_class = "cacheslide_vllm.model:CacheSlideLlamaForCausalLM"
    assert entries == {
        "LlamaForCausalLM": native,
        "CacheSlideLlamaForCausalLM": custom_class,
    }


@pytest.fixture
def native_source():
    source = os.environ.get("CACHESLIDE_VLLM_SOURCE")
    if not source:
        pytest.skip("Set CACHESLIDE_VLLM_SOURCE for read-only native source contracts")
    root = Path(source)
    assert root.is_dir(), "CACHESLIDE_VLLM_SOURCE must be an explicit vLLM source root"
    return root


def _method(tree, class_name, method_name):
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )


def test_exact_native_source_fingerprint_and_forward_signatures(native_source):
    manifest = verify_vllm_sources(native_source, version="0.29.0")
    assert manifest["vllm_git_commit"] == "98dff2a81d747d1dba01a47f939f48c3526d4206"
    runner = ast.parse(
        (native_source / "vllm/v1/worker/gpu_model_runner.py").read_text()
    )
    method = _method(runner, "GPUModelRunner", "_model_forward")
    assert [arg.arg for arg in method.args.args] == [
        "self",
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
    ]
    cleanup = _method(runner, "GPUModelRunner", "_on_request_state_removed")
    assert [arg.arg for arg in cleanup.args.args] == ["self", "req_id", "req_state"]
    llama = ast.parse(
        (native_source / "vllm/model_executor/models/llama.py").read_text()
    )
    layer = _method(llama, "LlamaDecoderLayer", "forward")
    assert [arg.arg for arg in layer.args.args] == [
        "self",
        "positions",
        "hidden_states",
        "residual",
    ]
    attention = ast.parse(
        (native_source / "vllm/v1/attention/backends/flash_attn.py").read_text()
    )
    update = _method(attention, "FlashAttentionImpl", "do_kv_cache_update")
    assert [arg.arg for arg in update.args.args] == [
        "self",
        "layer",
        "key",
        "value",
        "kv_cache",
        "slot_mapping",
    ]
    assert "kv_cache.transpose(1, 2).split(self.head_size, dim=-1)" in ast.unparse(
        update
    )


def test_changed_native_sources_and_versions_fail_closed(native_source, tmp_path):
    for relative in compatibility_manifest()["sha256"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native_source / relative, target)
    target = tmp_path / "vllm/model_executor/models/llama.py"
    target.write_text(target.read_text() + "\n# modified native source\n")
    with pytest.raises(CompatibilityError, match=r"llama.py \(modified\)"):
        verify_vllm_sources(tmp_path, version="0.29.0")
    with pytest.raises(CompatibilityError, match="requires vLLM 0.29.0"):
        verify_vllm_sources(native_source, version="0.30.0")
