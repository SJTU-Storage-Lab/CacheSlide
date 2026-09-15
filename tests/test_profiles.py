import json
from dataclasses import replace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from cacheslide_core.artifacts import (
    AdapterBundle,
    AttentionAdapter,
    file_sha256,
    save_adapter,
)
from cacheslide_core.contracts import RequestPlan, digest
from cacheslide_core.profiles import ProfileBundle, calibrate_profiles


@pytest.fixture
def tiny_artifacts(tmp_path):
    """A local two-layer GQA checkpoint and one-step adapter fixture."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 24,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(71)

    def random(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    weights = {
        "model.embed_tokens.weight": random(24, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": random(24, 8),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = torch.ones(8)
        weights[prefix + "post_attention_layernorm.weight"] = torch.ones(8)
        for name, shape in {
            "q_proj": (8, 8),
            "k_proj": (4, 8),
            "v_proj": (4, 8),
            "o_proj": (8, 8),
        }.items():
            weights[prefix + f"self_attn.{name}.weight"] = random(*shape)
        for name, shape in {
            "gate_proj": (12, 8),
            "up_proj": (12, 8),
            "down_proj": (8, 12),
        }.items():
            weights[prefix + f"mlp.{name}.weight"] = random(*shape)
    save_file(weights, str(model_dir / "model.safetensors"))
    adapters = [AttentionAdapter(config, rank=2, max_positions=16) for _ in range(2)]
    optimizer = torch.optim.SGD([p for a in adapters for p in a.parameters()], lr=0.1)
    loss = sum(
        (a.cope.position_embeddings - 0.2).square().mean()
        + (a.qkv_b - 0.1).square().mean()
        for a in adapters
    )
    loss.backward()
    optimizer.step()
    adapter_dir = save_adapter(
        tmp_path / "adapter",
        model_dir,
        adapters,
        training_steps=1,
        training_tokens=6,
        losses=[float(loss.detach())],
    )
    return model_dir, adapter_dir, AdapterBundle(adapter_dir, model_dir)


def make_plan(dynamic=(3,), *, fixed=(5, 6), task_id="qa"):
    split = 2 + len(dynamic)
    tokens = (1, 2, *dynamic, *fixed)
    return RequestPlan.parse(
        {
            "version": 1,
            "operation": "calibrate",
            "namespace": "test",
            "task_id": task_id,
            "chunks": [
                {"id": "document-a", "role": "reuse", "start": 0, "end": 2},
                {"id": "question", "role": "recompute", "start": 2, "end": split},
                {
                    "id": "document-b",
                    "role": "reuse",
                    "start": split,
                    "end": len(tokens),
                },
            ],
        },
        tokens,
    )


def rewrite_manifest(path, update=None):
    metadata = json.loads((path / "manifest.json").read_text())
    if update:
        update(metadata)
    metadata.pop("manifest_sha256")
    metadata["manifest_sha256"] = digest(metadata)
    (path / "manifest.json").write_text(json.dumps(metadata))


def rewrite_tensors(path, update):
    tensors = load_file(str(path / "profiles.safetensors"))
    update(tensors)
    save_file(tensors, str(path / "profiles.safetensors"))
    rewrite_manifest(
        path,
        lambda meta: meta.update(
            tensors_sha256=file_sha256(path / "profiles.safetensors")
        ),
    )


def test_calibrate_genuine_contexts_canonical_lookup_and_safe_export(
    tiny_artifacts,
    tmp_path,
):
    from cacheslide_core.reference import ReferenceLlama

    model_dir, adapter_dir, adapter = tiny_artifacts
    short, long = make_plan(), make_plan((3, 4, 8))
    output = calibrate_profiles(
        model_dir, adapter_dir, [short, short, long], tmp_path / "profiles"
    )
    bundle = ProfileBundle(output, adapter.identity)
    assert bundle.identity == digest(bundle.metadata)
    profile = bundle.get(long, 0)
    assert profile is not None and profile.sample_count == 3
    assert profile.checkpoint_id == adapter.identity
    assert profile.trained_profile_version == "v1"
    assert tuple(c.role for c in profile.chunks) == ("document-a", "document-b")
    torch.testing.assert_close(profile.query_positions, torch.arange(4))
    torch.testing.assert_close(profile.key_positions, torch.arange(4))
    model = ReferenceLlama.from_checkpoint(model_dir, rank=2, max_positions=16)
    model.load_adapters(adapter)
    observed = {}

    def observer(layer, query, key, cope):
        fixed = torch.tensor(short.fixed_indices)
        full = cope.position_trace(
            query, key, torch.arange(5), checkpoint_id=adapter.identity
        )
        observed[layer] = full.project(fixed, fixed).positions
        if layer == 0:
            omitted = cope.position_trace(
                query[fixed],
                key[fixed],
                torch.arange(4),
                checkpoint_id=adapter.identity,
            ).positions
            assert not torch.allclose(observed[layer], omitted)

    with torch.inference_mode():
        model.forward(torch.tensor(short.token_ids), observer=observer)
    for layer in range(2):
        profile = bundle.get(long, layer)
        torch.testing.assert_close(profile.canonical_positions, observed[layer])
        selected = profile.lookup(
            profile.chunks,
            torch.tensor([3, 1]),
            checkpoint_id=adapter.identity,
            trained_profile_version="v1",
        )
        torch.testing.assert_close(selected, observed[layer][:, [3, 1]])
    assert bundle.get(make_plan(fixed=(5, 9)), 0) is None
    assert bundle.get(make_plan(task_id="different"), 0) is None
    assert bundle.get(long, 2) is None
    text = (output / "manifest.json").read_text()
    for private_name in ("token_ids", "masked_logits", "question", "requests"):
        assert private_name not in text
    with safe_open(output / "profiles.safetensors", framework="pt") as stored:
        assert len(stored.keys()) == 6
        assert all(
            key.rsplit(".", 1)[-1] in {"positions", "query_positions", "key_positions"}
            for key in stored.keys()
        )
    with pytest.raises(FileExistsError):
        calibrate_profiles(model_dir, adapter_dir, [short], output)


def test_budget_rejected_before_model_or_trace_allocation(
    tiny_artifacts, tmp_path, monkeypatch
):
    from cacheslide_core.reference import ReferenceLlama

    model_dir, adapter_dir, _ = tiny_artifacts

    def forbidden(*args, **kwargs):
        pytest.fail("quadratic calibration was attempted before budget preflight")

    monkeypatch.setattr(ReferenceLlama, "from_checkpoint", forbidden)
    # Each individual trace fits; the two-layer aggregate and source logits do not.
    with pytest.raises(ValueError, match="source traces.*max_elements"):
        calibrate_profiles(
            model_dir,
            adapter_dir,
            [make_plan(), make_plan()],
            tmp_path / "too-big",
            max_elements=100,
        )
    assert not (tmp_path / "too-big").exists()


def test_bundle_adapter_checksums_and_aggregate_budget(tiny_artifacts, tmp_path):
    model_dir, adapter_dir, adapter = tiny_artifacts
    output = calibrate_profiles(model_dir, adapter_dir, [make_plan()], tmp_path / "p")
    with pytest.raises(ValueError, match="adapter identity"):
        ProfileBundle(output, "0" * 64)
    with pytest.raises(ValueError, match="max_elements"):
        ProfileBundle(output, adapter.identity, max_elements=79)
    assert ProfileBundle(output, adapter.identity, max_elements=80).get(make_plan(), 0)
    metadata = json.loads((output / "manifest.json").read_text())
    metadata["trained_profile_version"] = "changed"
    (output / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifest checksum"):
        ProfileBundle(output, adapter.identity)
    rewrite_manifest(output)
    path = output / "profiles.safetensors"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tensor checksum"):
        ProfileBundle(output, adapter.identity)


@pytest.mark.parametrize(
    "corruption", ["nan", "bound", "causal", "shape", "dtype", "ordinals", "extra"]
)
def test_bundle_rejects_corrupt_tensor_layout_even_with_new_checksum(
    tiny_artifacts,
    tmp_path,
    corruption,
):
    model_dir, adapter_dir, adapter = tiny_artifacts
    output = calibrate_profiles(model_dir, adapter_dir, [make_plan()], tmp_path / "p")

    def corrupt(tensors):
        name = next(key for key in tensors if key.endswith(".positions"))
        if corruption == "nan":
            tensors[name][0, 0, 0] = torch.nan
        elif corruption == "bound":
            tensors[name][0, 0, 0] = 16
        elif corruption == "causal":
            tensors[name][0, 0, 1] = 0.2
        elif corruption == "shape":
            tensors[name] = tensors[name][:, :1].contiguous()
        elif corruption == "dtype":
            tensors[name] = tensors[name].double()
        elif corruption == "ordinals":
            tensors[name.removesuffix(".positions") + ".key_positions"][1] = 8
        else:
            tensors["raw_trace"] = torch.ones(1)

    rewrite_tensors(output, corrupt)
    with pytest.raises(ValueError, match="finite|bounded|noncausal|layout|keys"):
        ProfileBundle(output, adapter.identity)


@pytest.mark.parametrize(
    "corruption", ["format", "layer", "chunks", "count", "width", "unknown"]
)
def test_bundle_rejects_invalid_manifest_even_with_new_checksum(
    tiny_artifacts,
    tmp_path,
    corruption,
):
    model_dir, adapter_dir, adapter = tiny_artifacts
    output = calibrate_profiles(model_dir, adapter_dir, [make_plan()], tmp_path / "p")

    def corrupt(metadata):
        entry = next(iter(metadata["profiles"].values()))
        if corruption == "format":
            metadata["format"] = "cacheslide-profiles-v99"
        elif corruption == "layer":
            entry["layer"] = 2
        elif corruption == "chunks":
            entry["chunks"][0]["length"] = 3
        elif corruption == "count":
            entry["sample_count"] = 0
        elif corruption == "width":
            entry["histogram_bin_width"] = -1
        else:
            entry["raw_request"] = "should not be accepted"

    rewrite_manifest(output, corrupt)
    with pytest.raises(ValueError):
        ProfileBundle(output, adapter.identity)


def test_get_checks_ordered_identities_and_profile_version(tiny_artifacts, tmp_path):
    model_dir, adapter_dir, adapter = tiny_artifacts
    output = calibrate_profiles(
        model_dir,
        adapter_dir,
        [make_plan()],
        tmp_path / "p",
        trained_profile_version="trained-v2",
    )
    profile = ProfileBundle(output, adapter.identity).get(make_plan(), 0)
    with pytest.raises(ValueError, match="version"):
        profile.lookup(
            profile.chunks,
            torch.arange(4),
            checkpoint_id=adapter.identity,
            trained_profile_version="v1",
        )
    with pytest.raises(ValueError, match="ordered chunk"):
        profile.lookup(
            tuple(reversed(profile.chunks)),
            torch.arange(4),
            checkpoint_id=adapter.identity,
            trained_profile_version="trained-v2",
        )
    malformed = replace(make_plan(), chunks=tuple(reversed(make_plan().chunks)))
    with pytest.raises(ValueError, match="partition"):
        calibrate_profiles(model_dir, adapter_dir, [malformed], tmp_path / "bad")


def test_profile_files_cannot_escape_bundle_via_symlink(tiny_artifacts, tmp_path):
    model_dir, adapter_dir, adapter = tiny_artifacts
    output = calibrate_profiles(model_dir, adapter_dir, [make_plan()], tmp_path / "p")
    tensor_path = output / "profiles.safetensors"
    moved = tmp_path / "elsewhere.safetensors"
    tensor_path.rename(moved)
    tensor_path.symlink_to(moved)
    with pytest.raises(ValueError, match="non-symlink"):
        ProfileBundle(output, adapter.identity)
