import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from cacheslide_core.artifacts import (
    AdapterBundle,
    AttentionAdapter,
    backbone_files,
    file_sha256,
    save_adapter,
    validate_llama_config,
)


def tiny_config():
    return {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
    }


def make_backbone(tmp_path):
    model = tmp_path / "backbone"
    model.mkdir()
    config = tiny_config()
    (model / "config.json").write_text(json.dumps(config))
    # Artifact validation hashes weights; execution is covered by training tests.
    save_file({"example.weight": torch.zeros(1)}, str(model / "model.safetensors"))
    return model, config


def save_test_adapter(tmp_path, model, config):
    layers = [AttentionAdapter(config, 2, 8) for _ in range(2)]
    output = save_adapter(
        tmp_path / "adapter",
        model,
        layers,
        training_steps=1,
        training_tokens=3,
        losses=[2.0],
    )
    return output, layers


@pytest.mark.parametrize(
    "epsilon", [0, -1, float("nan"), float("inf"), True, None, "1e-6", [], 10**1000]
)
def test_normalization_epsilon_must_be_finite_and_positive(epsilon):
    config = tiny_config()
    config["rms_norm_eps"] = epsilon
    with pytest.raises(ValueError, match="rms_norm_eps"):
        validate_llama_config(config)


@pytest.mark.parametrize(
    "scale", [float("nan"), float("inf"), True, None, "2.0", [], 10**1000]
)
def test_logit_scale_must_be_finite_numeric(scale):
    config = tiny_config()
    config["logit_scale"] = scale
    with pytest.raises(ValueError, match="logit_scale"):
        validate_llama_config(config)


@pytest.mark.parametrize("scale", [-2.0, 0.0, 1, 2.5])
def test_finite_logit_scales_preserve_native_forward_semantics(scale):
    config = tiny_config()
    config["logit_scale"] = scale
    assert validate_llama_config(config)["logit_scale"] == scale


def test_default_head_geometry_cannot_silently_truncate_hidden_size():
    config = tiny_config()
    config["hidden_size"] = 9
    with pytest.raises(ValueError, match="divide evenly"):
        validate_llama_config(config)
    # Explicit independent head dimensions are valid, e.g. Mistral variants.
    config["head_dim"] = 4
    assert validate_llama_config(config)["head_dim"] == 4


def test_malformed_model_and_shard_values_raise_validation_error(tmp_path):
    config = tiny_config()
    config["model_type"] = []
    with pytest.raises(ValueError):
        validate_llama_config(config)
    model, _ = make_backbone(tmp_path)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "shard.safetensors", "b": []}})
    )
    with pytest.raises(ValueError, match="shard names"):
        backbone_files(model)


@pytest.mark.parametrize(
    "field,invalid",
    [("rank", True), ("rank", 0), ("max_positions", True), ("max_positions", 1.0)],
)
def test_adapter_geometry_validation_precedes_tensor_allocation(field, invalid):
    geometry = {"rank": 2, "max_positions": 8, field: invalid}
    with pytest.raises(ValueError):
        AttentionAdapter(tiny_config(), **geometry)


def test_valid_artifact_loads_without_consuming_rng_state(tmp_path):
    model, config = make_backbone(tmp_path)
    output, layers = save_test_adapter(tmp_path, model, config)
    rng_before = torch.random.get_rng_state().clone()
    bundle = AdapterBundle(output, model)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    restored = bundle.layer(0, dtype=torch.float64)
    for name, tensor in layers[0].state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor.double())
    for index in (-1, 2, True):
        with pytest.raises(ValueError, match="layer index"):
            bundle.layer(index)


@pytest.mark.parametrize("defect", ["shape", "dtype", "nan", "missing"])
def test_bundle_rejects_invalid_tensors_before_layer_loading(tmp_path, defect):
    model, config = make_backbone(tmp_path)
    output, _ = save_test_adapter(tmp_path, model, config)
    weights_path = output / "adapter.safetensors"
    weights = load_file(str(weights_path))
    name = "layers.0.qkv_a"
    if defect == "shape":
        weights[name] = weights[name][:1].clone()
    elif defect == "dtype":
        weights[name] = weights[name].long()
    elif defect == "nan":
        weights[name][0, 0] = float("nan")
    else:
        del weights[name]
    save_file(weights, str(weights_path))
    manifest_path = output / "manifest.json"
    metadata = json.loads(manifest_path.read_text())
    metadata["adapter_sha256"] = file_sha256(weights_path)
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="tensor"):
        AdapterBundle(output, model)


@pytest.mark.parametrize("defect", ["rank", "positions", "shape", "dtype", "nan"])
def test_export_rejects_invalid_layers_before_creating_output(tmp_path, defect):
    model, config = make_backbone(tmp_path)
    layers = [AttentionAdapter(config, 2, 8) for _ in range(2)]
    if defect == "rank":
        layers[1] = AttentionAdapter(config, 3, 8)
    elif defect == "positions":
        layers[1] = AttentionAdapter(config, 2, 9)
    elif defect == "shape":
        layers[1].qkv_a = nn.Parameter(torch.zeros(2, 7))
    elif defect == "dtype":
        layers[1].qkv_a = nn.Parameter(
            torch.zeros(2, 8, dtype=torch.long), requires_grad=False
        )
    else:
        with torch.no_grad():
            layers[1].qkv_a[0, 0] = float("nan")
    output = tmp_path / "invalid-artifact"
    with pytest.raises(ValueError):
        save_adapter(
            output, model, layers, training_steps=1, training_tokens=3, losses=[2.0]
        )
    assert not output.exists()
