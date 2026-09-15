import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from cacheslide_vllm.artifacts import AdapterBundle
from cacheslide_vllm.config import CacheSlideSettings
from cacheslide_vllm.contracts import RequestPlan
from cacheslide_vllm.integration import StepContext
from cacheslide_vllm.profiles import calibrate_profiles
from cacheslide_vllm.reference import ReferenceLlama
from cacheslide_vllm.runtime import CacheSlideRuntime
from cacheslide_vllm.training import train_adapter


def plan(dynamic=(3,), operation="recompute"):
    # Shared A and B are separated by different dynamic text; B is non-prefix.
    tokens = (1, 2, *dynamic, 5, 6, 7, 8, 9)
    split = 2 + len(dynamic)
    return RequestPlan.parse(
        {
            "version": 1,
            "operation": operation,
            "namespace": "test",
            "task_id": "qa",
            "chunks": [
                {"id": "A", "role": "reuse", "start": 0, "end": 2},
                {"id": "dynamic", "role": "recompute", "start": 2, "end": split},
                {"id": "B", "role": "reuse", "start": split, "end": split + 4},
                {
                    "id": "query",
                    "role": "recompute",
                    "start": split + 4,
                    "end": len(tokens),
                },
            ],
        },
        tokens,
    )


def build_runtime(tmp_path, **settings_overrides):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(92)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    weights = {
        "model.embed_tokens.weight": rand(16, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": rand(16, 8),
    }
    for layer in range(6):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = torch.ones(8)
        weights[prefix + "post_attention_layernorm.weight"] = torch.ones(8)
        for name, shape in {
            "q_proj": (8, 8),
            "k_proj": (4, 8),
            "v_proj": (4, 8),
            "o_proj": (8, 8),
        }.items():
            weights[prefix + f"self_attn.{name}.weight"] = rand(*shape)
        for name, shape in {
            "gate_proj": (12, 8),
            "up_proj": (12, 8),
            "down_proj": (8, 12),
        }.items():
            weights[prefix + f"mlp.{name}.weight"] = rand(*shape)
    save_file(weights, str(model_dir / "model.safetensors"))
    torch.manual_seed(9)
    result = train_adapter(
        model_dir,
        [list(plan().token_ids)],
        tmp_path / "adapter",
        steps=2,
        lr=0.01,
        rank=2,
        max_positions=16,
    )
    bundle = AdapterBundle(result.output, model_dir)
    profile_dir = calibrate_profiles(
        model_dir,
        result.output,
        [plan((3,)), plan((4, 10))],
        tmp_path / "profiles",
        max_elements=100_000,
    )
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=16)
    model.load_adapters(bundle)
    settings = dict(
        artifact_path=str(result.output),
        cache_root=str(tmp_path / "cache"),
        profile_path=str(profile_dir),
        query_chunk_size=2,
        cpu_budget_bytes=32768,
        disk_budget_bytes=1048576,
        # Explicit engineering variant used to exercise nonzero WCA candidates.
        calibration_layer=1,
        ccpe_position_policy="mixed_bias_override",
    )
    settings.update(settings_overrides)
    runtime = CacheSlideRuntime(
        CacheSlideSettings(**settings), bundle, model_dtype=torch.float32
    )
    for layer in model.layers:
        layer.self_attn.attention_handler = runtime.attention
    return model, runtime


@torch.inference_mode()
def prefill(model, runtime, request, request_id="r"):
    ids = torch.tensor(request.token_ids)
    positions = torch.arange(len(ids))
    step = StepContext(
        request_id,
        request.token_ids,
        tuple(positions.tolist()),
        {"cacheslide": request.to_json()},
    )
    hidden = runtime.run(model, model.embed_tokens(ids), positions, step)
    logits, _ = model.lm_head(hidden)
    return logits


@torch.inference_mode()
def decode(model, runtime, request, token, position, request_id="r"):
    step = StepContext(
        request_id, request.token_ids, (position,), {"cacheslide": request.to_json()}
    )
    hidden = runtime.run(
        model, model.embed_tokens(torch.tensor([token])), torch.tensor([position]), step
    )
    logits, _ = model.lm_head(hidden)
    return logits


def test_recompute_without_profile_matches_dense_trained_model_and_decode(tmp_path):
    model, runtime = build_runtime(tmp_path, profile_path=None)
    try:
        request = plan()
        actual = prefill(model, runtime, request)
        expected = model(torch.tensor(request.token_ids))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        for i, token in enumerate((3, 4, 5)):
            result = decode(model, runtime, request, token, len(request.token_ids) + i)
            expected = model(torch.tensor((*request.token_ids, *(3, 4, 5)[: i + 1])))
            torch.testing.assert_close(result[-1], expected[-1], atol=3e-6, rtol=3e-6)
    finally:
        runtime.close()


def test_unchanged_nonprefix_reuse_is_exact_and_skips_later_token_rows(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        expected = prefill(model, runtime, plan(operation="populate"))[-1]
        actual = prefill(model, runtime, plan(operation="reuse"))[-1]
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        metrics = runtime.last_metrics
        assert metrics["cache_hit"]
        assert metrics["layer_rows"][:2] == [8, 8]
        assert metrics["computed_token_layers"] < metrics["dense_token_layers"]
        assert runtime.store.stats()
    finally:
        runtime.close()


def test_shifted_dynamic_chunk_reuses_fixed_context_and_decodes(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        prefill(model, runtime, plan(operation="populate"))
        request = plan((4, 10), operation="reuse")
        result = prefill(model, runtime, request)
        assert runtime.last_metrics["cache_hit"]
        assert not runtime.last_metrics["fallback"]
        assert runtime.last_metrics["computed_token_layers"] < 9 * 6
        assert torch.isfinite(result).all()
        for index in range(3):
            result = decode(
                model, runtime, request, int(result[-1].argmax()), 9 + index
            )
            assert result.shape == (1, 16) and torch.isfinite(result).all()
    finally:
        runtime.close()


def test_wca_gate_promotes_tokens_and_restores_cached_layer_inputs(tmp_path):
    model, runtime = build_runtime(
        tmp_path, correction_fraction=0.17, convergence_threshold=2.0
    )
    try:
        prefill(model, runtime, plan(operation="populate"))
        result = prefill(model, runtime, plan((4, 10), operation="reuse"))
        assert runtime.last_metrics["cache_hit"] and torch.isfinite(result).all()
        assert runtime.last_metrics["restored_rows"] > 0
    finally:
        runtime.close()


def test_layout_or_namespace_change_is_full_recompute_miss(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        prefill(model, runtime, plan(operation="populate"))
        request = replace(plan(operation="reuse"), namespace="another")
        result = prefill(model, runtime, request)
        assert not runtime.last_metrics["cache_hit"]
        assert runtime.last_metrics["layer_rows"] == [8] * 6
        assert torch.isfinite(result).all()
    finally:
        runtime.close()


def test_decode_requires_unchanged_metadata_and_consecutive_position(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        request = plan()
        prefill(model, runtime, request)
        with pytest.raises(ValueError, match="unchanged"):
            decode(model, runtime, replace(request, namespace="x"), 1, 8)
        with pytest.raises(ValueError, match="consecutive"):
            decode(model, runtime, request, 1, 9)
        runtime.release("r")
        assert not runtime.requests
        with pytest.raises(ValueError, match="unchanged"):
            decode(model, runtime, request, 1, 8)
    finally:
        runtime.close()
