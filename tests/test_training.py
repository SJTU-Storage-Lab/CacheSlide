import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from cacheslide_vllm.artifacts import AdapterBundle, backbone_manifest
from cacheslide_vllm.position import cope_attention
from cacheslide_vllm.reference import ReferenceLlama
from cacheslide_vllm.training import train_adapter


def tiny_checkpoint(path, *, tied=False, layers=2):
    path.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": tied,
        "hidden_act": "silu",
    }
    generator = torch.Generator().manual_seed(17)

    def random(shape):
        return torch.randn(shape, generator=generator) * 0.15

    weights = {
        "model.embed_tokens.weight": random((16, 8)),
        "model.norm.weight": torch.ones(8),
    }
    if not tied:
        weights["lm_head.weight"] = random((16, 8))
    for layer in range(layers):
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
            weights[prefix + name] = random(shape)
        weights[prefix + "input_layernorm.weight"] = torch.ones(8)
        weights[prefix + "post_attention_layernorm.weight"] = torch.ones(8)
    (path / "config.json").write_text(json.dumps(config))
    save_file(weights, str(path / "model.safetensors"))
    return path, config, weights


def manual_content_attention_reference(ids, config, weights):
    """Independent unfused residual implementation with zero adapter weights."""
    weights = {name: value.double() for name, value in weights.items()}
    hidden = weights["model.embed_tokens.weight"][ids]

    def norm(value, weight):
        return (
            value
            * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
            )
            * weight
        )

    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        normalized = norm(hidden, weights[prefix + "input_layernorm.weight"])
        q = F.linear(normalized, weights[prefix + "self_attn.q_proj.weight"])
        k = F.linear(normalized, weights[prefix + "self_attn.k_proj.weight"])
        v = F.linear(normalized, weights[prefix + "self_attn.v_proj.weight"])
        q = q.reshape(-1, 2, 4)
        k = k.reshape(-1, 1, 4).expand(-1, 2, -1)
        v = v.reshape(-1, 1, 4).expand(-1, 2, -1)
        logits = torch.einsum("qhd,khd->hqk", q, k) / 2
        mask = torch.ones(ids.numel(), ids.numel(), dtype=torch.bool).tril()
        probability = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        attention = torch.einsum("hqk,khd->qhd", probability, v).reshape(-1, 8)
        hidden = hidden + F.linear(
            attention, weights[prefix + "self_attn.o_proj.weight"]
        )
        normalized = norm(hidden, weights[prefix + "post_attention_layernorm.weight"])
        gate = F.linear(normalized, weights[prefix + "mlp.gate_proj.weight"])
        up = F.linear(normalized, weights[prefix + "mlp.up_proj.weight"])
        hidden = hidden + F.linear(
            F.silu(gate) * up, weights[prefix + "mlp.down_proj.weight"]
        )
    hidden = norm(hidden, weights["model.norm.weight"])
    output = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return F.linear(hidden, output)


@pytest.mark.parametrize("tied", [False, True])
def test_frozen_reference_matches_dense_causal_swiglu_and_fused_residual(
    tmp_path, tied
):
    model_dir, config, weights = tiny_checkpoint(tmp_path / "model", tied=tied)
    model = ReferenceLlama.from_checkpoint(
        model_dir, rank=2, max_positions=8, dtype=torch.float64, query_chunk_size=2
    )
    ids = torch.tensor([1, 2, 5, 3])
    expected = manual_content_attention_reference(ids, config, weights)
    actual = model(ids)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(ids[:2]), actual[:2])


def test_only_explicit_attention_adapters_receive_gradients(tmp_path):
    model_dir, _, _ = tiny_checkpoint(tmp_path / "model")
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
    ids = torch.tensor([1, 2, 5, 3])
    loss = F.cross_entropy(model(ids[:-1]), ids[1:])
    loss.backward()
    assert all(
        parameter.requires_grad == name.startswith("adapters.")
        for name, parameter in model.named_parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("adapters.")
    )
    assert (
        sum(
            float(layer.cope.position_embeddings.grad.abs().sum())
            for layer in model.adapters
        )
        > 0
    )
    assert sum(float(layer.qkv_b.grad.abs().sum()) for layer in model.adapters) > 0
    assert sum(float(layer.out_b.grad.abs().sum()) for layer in model.adapters) > 0


def test_reference_and_training_honor_native_logit_scale(tmp_path):
    model_dir, config, weights = tiny_checkpoint(tmp_path / "model")
    ids = torch.tensor([1, 2, 5, 3])
    unscaled = ReferenceLlama(config, weights, rank=2, max_positions=8)
    expected_raw = unscaled(ids[:-1]).detach()
    config["logit_scale"] = 2.5
    (model_dir / "config.json").write_text(json.dumps(config))
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
    torch.testing.assert_close(model(ids[:-1]), expected_raw * 2.5)
    expected_loss = F.cross_entropy(expected_raw * 2.5, ids[1:]).item()
    assert expected_loss != pytest.approx(F.cross_entropy(expected_raw, ids[1:]).item())
    result = train_adapter(
        model_dir, [ids], tmp_path / "scaled-adapter", steps=1, rank=2, max_positions=8
    )
    assert result.losses[0] == pytest.approx(expected_loss)
    bundle = AdapterBundle(result.output, model_dir)
    assert bundle.config["logit_scale"] == 2.5
    model.load_adapters(bundle)
    # Loaded adapter inference applies the same scale after the frozen head.
    scaled = model(ids)
    model.logit_scale = 1.0
    torch.testing.assert_close(scaled, model(ids) * 2.5)


def test_observer_and_native_style_attention_handler(tmp_path):
    model_dir, _, _ = tiny_checkpoint(tmp_path / "model")
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
    ids, observed, handled = torch.tensor([1, 2, 5, 3]), [], []
    baseline = model(ids)

    def observer(layer, q, k, cope):
        observed.append((layer, q.shape, k.shape, cope.max_positions))

    def handler(layer, positions, q, k, v, adapter):
        handled.append(layer)
        return cope_attention(q, k, v, adapter.cope, positions)

    for layer in model.layers:
        layer.self_attn.attention_handler = handler
    torch.testing.assert_close(model(ids, observer=observer), baseline)
    assert handled == [0, 1]
    assert observed == [(0, (4, 2, 4), (4, 1, 4), 8), (1, (4, 2, 4), (4, 1, 4), 8)]


def test_training_makes_optimizer_updates_and_exports_verified_adapter(tmp_path):
    model_dir, _, _ = tiny_checkpoint(tmp_path / "model")
    base_before = backbone_manifest(model_dir)
    result = train_adapter(
        model_dir,
        [[1, 2, 3, 4, 5]],
        tmp_path / "adapter",
        steps=12,
        lr=0.04,
        rank=2,
        max_positions=8,
    )
    assert result.training_steps == 12
    assert result.training_tokens == 48
    assert len(result.losses) == 12
    assert result.losses[-1] < result.losses[0]
    assert backbone_manifest(model_dir) == base_before
    weights = load_file(str(result.output / "adapter.safetensors"))
    assert weights["layers.0.cope.position_embeddings"].abs().sum() > 0
    assert weights["layers.0.qkv_b"].abs().sum() > 0
    bundle = AdapterBundle(result.output, model_dir)
    model = ReferenceLlama.from_checkpoint(model_dir, rank=1, max_positions=3)
    model.load_adapters(bundle)
    assert model.adapters[0].rank == 2
    assert model.adapters[0].cope.max_positions == 8
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.layers[0].self_attn.adapter is model.adapters[0]
    assert torch.isfinite(model(torch.tensor([1, 2, 3]))).all()
    with pytest.raises(FileExistsError):
        train_adapter(model_dir, [[1, 2]], result.output, steps=1)


def test_no_zero_step_or_invalid_token_training_export(tmp_path):
    with pytest.raises(ValueError, match="zero-step"):
        train_adapter(tmp_path / "missing", [[1, 2]], tmp_path / "output", steps=0)
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="at least two"):
        train_adapter(tmp_path / "missing", [[1]], tmp_path / "output", steps=1)
    with pytest.raises(ValueError, match="lr"):
        train_adapter(
            tmp_path / "missing",
            [[1, 2]],
            tmp_path / "output",
            steps=1,
            lr=float("nan"),
        )


def test_local_loader_rejects_unexpected_weights_and_invalid_ids(tmp_path):
    model_dir, _, weights = tiny_checkpoint(tmp_path / "model")
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
    with pytest.raises(ValueError, match="in-vocabulary"):
        model(torch.tensor([16]))
    with pytest.raises(ValueError, match="in-vocabulary"):
        model(torch.tensor([[1, 2]]))
    weights["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(8)
    save_file(weights, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=8)
