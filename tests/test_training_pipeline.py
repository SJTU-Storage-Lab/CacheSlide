import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from cacheslide_vllm.artifacts import AdapterBundle, backbone_manifest
from cacheslide_vllm.config import CacheSlideSettings
from cacheslide_vllm.reference import ReferenceLlama
from cacheslide_vllm.training_data import (
    TokenCorpus,
    assert_disjoint,
    prepare_corpus,
)
from cacheslide_vllm.training_pipeline import (
    PretrainingConfig,
    TrainingForward,
    evaluate,
    learning_rate_at,
    main,
    run_pretraining,
)


@pytest.fixture
def model_dir(tmp_path):
    """Tiny HF-format checkpoint for mechanics only, never an experiment result."""
    path = tmp_path / "backbone"
    path.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }
    generator = torch.Generator().manual_seed(19)
    weights = {
        "model.embed_tokens.weight": torch.randn(16, 8, generator=generator) * 0.2,
        "lm_head.weight": torch.randn(16, 8, generator=generator) * 0.2,
        "model.norm.weight": torch.ones(8),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for name, shape in {
            "self_attn.q_proj.weight": (8, 8),
            "self_attn.k_proj.weight": (4, 8),
            "self_attn.v_proj.weight": (4, 8),
            "self_attn.o_proj.weight": (8, 8),
            "mlp.gate_proj.weight": (12, 8),
            "mlp.up_proj.weight": (12, 8),
            "mlp.down_proj.weight": (8, 12),
        }.items():
            weights[prefix + name] = torch.randn(shape, generator=generator) * 0.2
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            weights[prefix + name] = torch.ones(8)
    (path / "config.json").write_text(json.dumps(config))
    save_file(weights, str(path / "model.safetensors"))
    return path


def write_rows(path, rows):
    path.write_text("".join(json.dumps({"token_ids": ids}) + "\n" for ids in rows))
    return path


def small_config(**kwargs):
    return replace(
        PretrainingConfig(
            max_steps=4,
            warmup_steps=0,
            learning_rate=0.02,
            gradient_accumulation_steps=2,
            sequence_length=8,
            rank=2,
            max_positions=8,
            query_chunk_size=2,
            loss_chunk_size=2,
            eval_every=2,
            checkpoint_every=2,
        ),
        **kwargs,
    )


def test_preparation_preserves_all_transitions_and_document_boundaries(tmp_path):
    source = write_rows(tmp_path / "source.jsonl", [[1, 2, 3, 4, 5, 6], [7, 8]])
    result = prepare_corpus(source, tmp_path / "prepared", sequence_length=2)
    data = TokenCorpus(
        tmp_path / "prepared/tokens.jsonl", max_sequence_length=2, vocab_size=16
    )
    assert [data[i] for i in range(len(data))] == [[1, 2, 3], [3, 4, 5], [5, 6], [7, 8]]
    assert result["supervised_tokens"] == 6
    assert result["paper_corpus_claim"] is False
    assert len(data.document_ids) == 2
    assert data.manifest()["supervised_tokens_per_epoch"] == 6
    with pytest.raises(FileExistsError):
        prepare_corpus(source, tmp_path / "prepared", sequence_length=2)


def test_corpus_leakage_and_post_index_changes_fail_closed(tmp_path):
    train_path = write_rows(tmp_path / "train.jsonl", [[1, 2, 3]])
    train = TokenCorpus(train_path, max_sequence_length=4, vocab_size=16)
    copy = TokenCorpus(
        write_rows(tmp_path / "val.jsonl", [[1, 2, 3]]),
        max_sequence_length=4,
        vocab_size=16,
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(train, copy)
    write_rows(train_path, [[4, 5, 6]])
    with pytest.raises(ValueError, match="changed"):
        train[0]


def test_preparation_preserves_original_document_identity(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "token_ids": [1, 2, 3, 4],
                "document_sha256": "a" * 64,
            }
        )
        + "\n"
    )
    prepare_corpus(source, tmp_path / "prepared", sequence_length=2)
    data = TokenCorpus(
        tmp_path / "prepared/tokens.jsonl", max_sequence_length=2, vocab_size=16
    )
    assert data.document_ids == {"a" * 64}


@pytest.mark.parametrize("checkpointed", [False, True])
def test_checkpointed_forward_matches_reference_loss_and_gradients(
    model_dir, checkpointed
):
    torch.manual_seed(3)
    reference = ReferenceLlama.from_checkpoint(
        model_dir, rank=2, max_positions=8, query_chunk_size=2
    )
    ids = torch.tensor([1, 2, 3, 4, 5])
    expected_loss = F.cross_entropy(reference(ids[:-1]), ids[1:], reduction="sum")
    expected_loss.backward()
    gradients = {
        name: param.grad.clone()
        for name, param in reference.adapters.named_parameters()
    }
    torch.manual_seed(3)
    model = ReferenceLlama.from_checkpoint(
        model_dir, rank=2, max_positions=8, query_chunk_size=2
    )
    forward = TrainingForward(model, small_config(gradient_checkpointing=checkpointed))
    model.train()
    loss = forward.loss_sum(ids)
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    for name, parameter in model.adapters.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name])
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if not name.startswith("adapters.")
    )


def test_heldout_nll_is_target_weighted_not_sequence_mean(model_dir, tmp_path):
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
    forward = TrainingForward(model, small_config())
    data = TokenCorpus(
        write_rows(tmp_path / "eval.jsonl", [[1, 2], [3, 4, 5, 6]]),
        max_sequence_length=8,
        vocab_size=16,
    )
    expected = (
        sum(float(forward.loss_sum(torch.tensor(data[i])).detach()) for i in range(2))
        / 4
    )
    report = evaluate(forward, data)
    assert report["nll"] == pytest.approx(expected)
    assert report["supervised_tokens"] == 4
    assert report["full_validation_split"] is True
    assert evaluate(forward, data, max_sequences=1)["full_validation_split"] is False


def test_training_exports_native_adapter_and_exact_resume(model_dir, tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [[1, 2, 3, 4, 5], [2, 4, 6]])
    val = write_rows(tmp_path / "val.jsonl", [[1, 3, 5, 7]])
    before = backbone_manifest(model_dir)
    config = small_config()
    complete = run_pretraining(model_dir, train, val, tmp_path / "full", config=config)
    partial = run_pretraining(
        model_dir, train, val, tmp_path / "partial", config=config, stop_after_steps=2
    )
    assert partial["status"] == "partial" and partial["adapter"] is None
    assert not (tmp_path / "partial/adapter").exists()
    resumed = run_pretraining(
        model_dir,
        train,
        val,
        tmp_path / "resumed",
        config=config,
        resume=tmp_path / "partial/checkpoints/step-00000002",
    )
    assert complete["training_tokens"] == resumed["training_tokens"] == 24
    assert complete["paper_results_reproduced"] is False
    assert complete["status"] == resumed["status"] == "complete"
    assert backbone_manifest(model_dir) == before
    full_weights = load_file(str(tmp_path / "full/adapter/adapter.safetensors"))
    resumed_weights = load_file(str(tmp_path / "resumed/adapter/adapter.safetensors"))
    for name in full_weights:
        torch.testing.assert_close(
            full_weights[name], resumed_weights[name], rtol=0, atol=0
        )
    assert full_weights["layers.0.qkv_b"].abs().sum() > 0
    assert full_weights["layers.0.cope.position_embeddings"].abs().sum() > 0
    bundle = AdapterBundle(tmp_path / "full/adapter", model_dir)
    assert bundle.metadata["training_steps"] == 4
    assert complete["validation"][-1] == resumed["validation"][-1]


def test_resume_changed_config_or_corpus_is_rejected(model_dir, tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [[1, 2, 3]])
    val = write_rows(tmp_path / "val.jsonl", [[2, 3, 4]])
    config = small_config()
    run_pretraining(
        model_dir, train, val, tmp_path / "partial", config=config, stop_after_steps=2
    )
    checkpoint_path = tmp_path / "partial/checkpoints/step-00000002"
    with pytest.raises(ValueError, match="identity"):
        run_pretraining(
            model_dir,
            train,
            val,
            tmp_path / "changed-config",
            config=replace(config, learning_rate=0.01),
            resume=checkpoint_path,
        )
    write_rows(train, [[1, 4, 3]])
    with pytest.raises(ValueError, match="identity"):
        run_pretraining(
            model_dir,
            train,
            val,
            tmp_path / "changed-corpus",
            config=config,
            resume=checkpoint_path,
        )
    assert not (tmp_path / "changed-corpus").exists()


def test_schedule_has_positive_steps_and_explicit_warmup():
    config = small_config(max_steps=5, warmup_steps=2, min_lr_ratio=0.1)
    config.validate()
    values = [learning_rate_at(config, step) for step in range(1, 6)]
    assert values[:3] == [0.01, 0.02, 0.02]
    assert values[-1] == pytest.approx(0.002)
    assert all(value > 0 for value in values)
    with pytest.raises(ValueError):
        small_config(max_steps=2, warmup_steps=2).validate()


def test_real_text_prepare_accepts_local_tokenizer_api(tmp_path):
    class LocalTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "real corpus text" and add_special_tokens is True
            return [1, 2, 3, 4]

    source = tmp_path / "text.jsonl"
    source.write_text(json.dumps({"text": "real corpus text"}) + "\n")
    result = prepare_corpus(
        source,
        tmp_path / "prepared",
        sequence_length=8,
        tokenizer=LocalTokenizer(),
        tokenizer_provenance={"revision": "local-test"},
    )
    assert result["supervised_tokens"] == 3
    assert result["tokenizer"]["revision"] == "local-test"


def test_prepare_module_cli(tmp_path, capsys):
    source = write_rows(tmp_path / "raw.jsonl", [[1, 2, 3, 4]])
    assert (
        main(
            [
                "prepare",
                "--input",
                str(source),
                "--output",
                str(tmp_path / "prepared"),
                "--sequence-length",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["supervised_tokens"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bfloat16_checkpointed_training(model_dir, tmp_path):
    """Small mechanics test only; run on an explicitly allocated visible GPU."""
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("hardware BF16 support required")
    train = write_rows(tmp_path / "train.jsonl", [[1, 2, 3, 4]])
    validation = write_rows(tmp_path / "validation.jsonl", [[1, 2, 3, 5]])
    result = run_pretraining(
        model_dir,
        train,
        validation,
        tmp_path / "bf16",
        config=small_config(precision="bfloat16"),
        device="cuda:0",
    )
    assert result["status"] == "complete"
    weights = load_file(str(tmp_path / "bf16/adapter/adapter.safetensors"))
    assert all(tensor.dtype == torch.float32 for tensor in weights.values())
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())
    assert weights["layers.0.cope.position_embeddings"].abs().sum() > 0


def _native_mount_constructor(monkeypatch, model_dir, engine):
    """Load the production CacheSlide model module with CPU native shells only.

    Engine-native allocation/dispatch is unavailable on this CPU runner. Its
    shells below expose real frozen projections/norm/MLP weights from the local
    safetensors checkpoint. AdapterBundle, bundle.layer, the actual CacheSlide
    model/attention wrappers, and CacheSlideRuntime are NOT mocked. This is not
    a native engine process, CUDA source-attestation or KV-reuse test.
    """
    allocations = []
    batch = object()

    class NativeAttention(nn.Module):
        def __init__(self, original, *, prefix=""):
            super().__init__()
            for name in ("q_size", "kv_size", "num_heads", "num_kv_heads", "head_dim"):
                setattr(self, name, getattr(original, name))
            self.qkv_proj, self.o_proj = original.qkv_proj, original.o_proj
            self.attn = nn.Identity()

    class NativeLayer(nn.Module):
        def __init__(self, config, *, prefix="", attn_layer_type=None, original):
            super().__init__()
            attention = attn_layer_type or NativeAttention
            self.self_attn = attention(original.self_attn, prefix=prefix)
            self.input_layernorm = original.input_layernorm
            self.post_attention_layernorm = original.post_attention_layernorm
            self.mlp = original.mlp

        def forward(self, positions, hidden, *args):
            residual = args[-1]
            if residual is None:
                residual = hidden
                hidden = self.input_layernorm(hidden)
            else:
                hidden, residual = self.input_layernorm(hidden, residual)
            if engine == "sglang":
                hidden = self.self_attn(positions, hidden, args[0])
            else:
                hidden = self.self_attn(positions, hidden)
            hidden, residual = self.post_attention_layernorm(hidden, residual)
            return self.mlp(hidden), residual

    class NativeModel(nn.Module):
        def __init__(
            self,
            config=None,
            *,
            vllm_config=None,
            prefix="",
            layer_type=None,
            quant_config=None,
        ):
            super().__init__()
            allocations.append(engine)
            frozen = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
            self.config = config
            self.padding_idx, self.vocab_size = 0, frozen.config["vocab_size"]
            self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
            self.start_layer, self.end_layer = 0, len(frozen.layers)
            self.layers_to_capture = []
            self.embed_tokens, self.norm = frozen.embed_tokens, frozen.norm
            layer_type = layer_type or NativeLayer
            self.layers = nn.ModuleList(
                layer_type(
                    config or vllm_config, prefix=f"model.layers.{i}", original=layer
                )
                for i, layer in enumerate(frozen.layers)
            )

        def forward(self, input_ids, positions, *args, **kwargs):
            hidden, residual = self.embed_tokens(input_ids), None
            for layer in self.layers:
                arguments = (batch, residual) if engine == "sglang" else (residual,)
                hidden, residual = layer(positions, hidden, *arguments)
            return self.norm(hidden, residual)[0]

    native = ModuleType(
        "sglang.srt.models.llama"
        if engine == "sglang"
        else "vllm.model_executor.models.llama"
    )
    native.LlamaAttention, native.LlamaModel = NativeAttention, NativeModel
    native.LlamaDecoderLayer, native.LlamaForCausalLM = NativeLayer, nn.Module
    monkeypatch.setitem(sys.modules, native.__name__, native)
    if engine == "sglang":
        memory = ModuleType("sglang.srt.mem_cache.memory_pool")
        memory.MHATokenToKVPool = type("MHATokenToKVPool", (), {})
        context = ModuleType("sglang.srt.model_executor.forward_context")
        context.get_req_to_token_pool = context.get_token_to_kv_pool = lambda: None
        for module in (memory, context):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        relative = "cacheslide_sglang/models/llama.py"
        module_name = "cacheslide_sglang._training_mount_test"
    else:
        context = ModuleType("vllm.model_executor.layers.attention.attention")
        context.get_attention_context = lambda *args: None
        utilities = ModuleType("vllm.model_executor.models.utils")
        utilities.extract_layer_index = lambda prefix: int(prefix.rsplit(".", 1)[1])
        for module in (context, utilities):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        relative = "cacheslide_vllm/model.py"
        module_name = "cacheslide_vllm._training_mount_test"
    path = Path(__file__).parents[1] / "src" / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    hf = SimpleNamespace(**json.loads((model_dir / "config.json").read_text()))

    def construct(artifact, cache_root):
        settings = CacheSlideSettings(
            artifact_path=str(artifact), cache_root=str(cache_root), query_chunk_size=2
        )
        if engine == "sglang":
            monkeypatch.setattr(implementation, "verify_installed_sglang", lambda: None)
            monkeypatch.setattr(
                implementation.integration,
                "launch_config",
                lambda: SimpleNamespace(settings=settings, model_path=str(model_dir)),
            )
            monkeypatch.setattr(
                implementation.integration, "current_forward_batch", lambda: batch
            )
            # Exercise exactly the production artifact loading/architecture path.
            return implementation.CacheSlideLlamaForCausalLM._init_model(None, hf)
        monkeypatch.setattr(implementation, "verify_installed_vllm", lambda: None)
        monkeypatch.setattr(
            implementation, "validate_engine_config", lambda config: None
        )
        config = SimpleNamespace(
            additional_config={
                "cacheslide": {
                    "artifact_path": str(artifact),
                    "cache_root": str(cache_root),
                    "query_chunk_size": 2,
                }
            },
            model_config=SimpleNamespace(model=str(model_dir), hf_config=hf),
        )
        return implementation.CacheSlideModel(vllm_config=config)

    def execute(mounted, ids):
        positions = torch.arange(ids.numel())
        return (
            mounted(ids, positions, batch)
            if engine == "sglang"
            else mounted(ids, positions)
        )

    return construct, execute, allocations


@pytest.mark.parametrize("engine", ["vllm"])
def test_trained_artifact_mounts_through_actual_native_adapter_path(
    model_dir, tmp_path, monkeypatch, engine
):
    """Tiny mechanics only: real training -> production loader -> used weights."""
    train = write_rows(tmp_path / "train.jsonl", [[1, 2, 3, 4, 5]])
    validation = write_rows(tmp_path / "validation.jsonl", [[1, 2, 3, 6]])
    run_pretraining(
        model_dir, train, validation, tmp_path / "training", config=small_config()
    )
    artifact = tmp_path / "training/adapter"
    bundle = AdapterBundle(artifact, model_dir)
    trained = ReferenceLlama.from_checkpoint(
        model_dir, rank=2, max_positions=8, query_chunk_size=2
    )
    trained.load_adapters(bundle)
    ids = torch.tensor([1, 2, 3, 4, 5])
    with torch.no_grad():
        expected = trained(ids)
    construct, execute, allocations = _native_mount_constructor(
        monkeypatch, model_dir, engine
    )
    mounted = construct(artifact, tmp_path / "native-cache")
    try:
        assert all(not parameter.requires_grad for parameter in mounted.parameters())
        assert mounted.cacheslide_runtime.bundle.identity == bundle.identity
        for layer in mounted.cacheslide_adapters:
            assert layer.qkv_b.abs().sum() > 0 and layer.out_b.abs().sum() > 0
            assert layer.cope.position_embeddings.abs().sum() > 0
        with torch.no_grad():
            actual = trained.lm_head(execute(mounted, ids))[0] * trained.logit_scale
        torch.testing.assert_close(actual, expected)
        # Disabling only the mounted LoRA must alter the actual module output.
        with torch.no_grad():
            for layer in mounted.cacheslide_adapters:
                layer.qkv_b.zero_()
                layer.out_b.zero_()
            without_lora = (
                trained.lm_head(execute(mounted, ids))[0] * trained.logit_scale
            )
        assert not torch.allclose(actual, without_lora, rtol=1e-6, atol=1e-7)
        with torch.no_grad():
            for layer in mounted.cacheslide_adapters:
                layer.cope.position_embeddings.zero_()
            without_cope = (
                trained.lm_head(execute(mounted, ids))[0] * trained.logit_scale
            )
        assert not torch.allclose(without_lora, without_cope, rtol=1e-6, atol=1e-7)
    finally:
        mounted.cacheslide_runtime.close()
    # Production verification must reject a changed backbone before allocation.
    allocation_count = len(allocations)
    weights = load_file(str(model_dir / "model.safetensors"))
    weights["model.embed_tokens.weight"][0, 0] += 1
    save_file(weights, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="backbone SHA-256"):
        construct(artifact, tmp_path / "wrong-backbone-cache")
    assert len(allocations) == allocation_count
