"""Launcher/manifest mechanics with tiny local fixtures; no model experiment."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save, save_file

from cacheslide_core.artifacts import AttentionAdapter, save_adapter

SPEC = importlib.util.spec_from_file_location(
    "paper_pretraining_launcher",
    Path(__file__).parents[1] / "scripts/run_paper_pretraining.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def asset_fixture(tmp_path):
    """Synthetic metadata for integrity tests, never labeled official downloads."""
    asset_root = tmp_path / "assets"
    model = asset_root / "models" / launcher.MODEL_NAME
    model.mkdir(parents=True)
    spec = {
        "models": [
            {
                "name": launcher.MODEL_NAME,
                "repo": launcher.MODEL_REPO,
                "revision": launcher.MODEL_REVISION,
            }
        ],
        "datasets": [],
        "archives": [],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    identity = {
        "owner": "cacheslide-paper-assets-v1",
        "spec_sha256": launcher._sha256(spec_path),
        "selection": "all",
    }
    (asset_root / ".owner.json").write_text(json.dumps(identity))
    model_config = {
        "model_type": "mistral",
        "sliding_window": None,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }
    shapes = {
        "model.embed_tokens.weight": (16, 8),
        "model.norm.weight": (8,),
        "lm_head.weight": (16, 8),
    }
    for name, shape in {
        "self_attn.q_proj.weight": (8, 8),
        "self_attn.k_proj.weight": (4, 8),
        "self_attn.v_proj.weight": (4, 8),
        "self_attn.o_proj.weight": (8, 8),
        "mlp.gate_proj.weight": (12, 8),
        "mlp.up_proj.weight": (12, 8),
        "mlp.down_proj.weight": (8, 12),
        "input_layernorm.weight": (8,),
        "post_attention_layernorm.weight": (8,),
    }.items():
        shapes["model.layers.0." + name] = shape
    shards = [{}, {}]
    weights = {}
    for i, (name, shape) in enumerate(shapes.items()):
        shards[i % 2][name] = torch.ones(shape) * 0.1
        weights[name] = f"model-{i % 2 + 1:05d}-of-00002.safetensors"
    files = {
        "config.json": json.dumps(model_config).encode(),
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": json.dumps({"weight_map": weights}).encode(),
        "model-00001-of-00002.safetensors": save(shards[0]),
        "model-00002-of-00002.safetensors": save(shards[1]),
    }
    entries = []
    for name, content in files.items():
        (model / name).write_bytes(content)
        entries.append(
            {
                "path": f"models/{launcher.MODEL_NAME}/{name}",
                "size": len(content),
                "url": f"https://huggingface.co/{launcher.MODEL_REPO}/resolve/{launcher.MODEL_REVISION}/{name}",
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_blob_sha1": None,
            }
        )
    manifest = {"identity": identity, "spec": spec, "files": entries}
    (asset_root / "download_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "trusted_model.json").write_text(
        json.dumps(
            {
                "repo": launcher.MODEL_REPO,
                "revision": launcher.MODEL_REVISION,
                "files": [
                    {
                        key: entry.get(key)
                        for key in ("path", "size", "sha256", "git_blob_sha1")
                    }
                    for entry in entries
                ],
            }
        )
    )
    config_path = tmp_path / "pilot.json"
    config_path.write_text(json.dumps({"max_steps": 20, "sequence_length": 512}))
    return asset_root, model, spec_path, config_path, manifest


def arguments(fixture, output):
    root, _, spec, config, _ = fixture
    return [
        "--asset-root",
        str(root),
        "--output",
        str(output),
        "--spec",
        str(spec),
        "--config",
        str(config),
        "--trusted-model-manifest",
        str(spec.with_name("trusted_model.json")),
        "--device",
        "cpu",
    ]


def fake_runner(
    monkeypatch,
    fixture,
    *,
    failure=None,
    changed_config=False,
    adapter_status="complete",
    bad_backbone=False,
    adapter_fault=None,
):
    calls = []
    root, model_dir, spec, config, _ = fixture

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == launcher.ROOT
        assert str(launcher.ROOT / "src") in kwargs["env"]["PYTHONPATH"]
        assert kwargs["check"] is False
        if failure and failure in command:
            return SimpleNamespace(returncode=7)
        if "download_paper_assets.py" in Path(command[1]).name:
            return SimpleNamespace(returncode=0)
        target = Path(command[command.index("--output") + 1])
        target.mkdir()
        if Path(command[1]).name == "prepare_paper_data.py":
            for name in ("train.corpus.jsonl", "nll_validation.corpus.jsonl"):
                (target / name).write_text('{"token_ids":[1,2,3]}\n')
        elif "prepare" in command:
            (target / "tokens.jsonl").write_text('{"token_ids":[1,2,3]}\n')
            if changed_config and target.name == "validation_tokens":
                config.write_text('{"max_steps":21,"sequence_length":512}')
        elif "train" in command:
            artifact = target / "adapter"
            model_config = json.loads((model_dir / "config.json").read_text())
            layer = AttentionAdapter(model_config, rank=2, max_positions=8)
            save_adapter(
                artifact,
                model_dir,
                [layer],
                training_steps=20,
                training_tokens=60,
                losses=[1.0] * 20,
            )
            metadata = json.loads((artifact / "manifest.json").read_text())
            if bad_backbone:
                metadata["base_files"]["config.json"] = "0" * 64
            if adapter_fault == "malformed":
                (artifact / "adapter.safetensors").write_bytes(b"not-safetensors")
            elif adapter_fault in {"nan", "wrong_shape"}:
                tensors = load_file(str(artifact / "adapter.safetensors"))
                if adapter_fault == "nan":
                    tensors["layers.0.qkv_b"][0, 0] = float("nan")
                else:
                    tensors["layers.0.qkv_b"] = torch.zeros(1, 1)
                save_file(tensors, str(artifact / "adapter.safetensors"))
            if adapter_fault:
                metadata["adapter_sha256"] = launcher._sha256(
                    artifact / "adapter.safetensors"
                )
            if adapter_fault == "one_step":
                metadata["training_steps"] = 1
            (artifact / "manifest.json").write_text(json.dumps(metadata))
            (target / "report.json").write_text(
                json.dumps(
                    {
                        "status": adapter_status,
                        "adapter": str(artifact),
                        "optimizer_steps": 1 if adapter_fault == "one_step" else 20,
                        "training_tokens": 60,
                    }
                )
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    return calls


def test_complete_model_checks_every_shard_and_hash(tmp_path):
    fixture = asset_fixture(tmp_path)
    root, _, spec, _, _ = fixture
    result = launcher.verify_complete_model(
        root, spec, spec.with_name("trusted_model.json")
    )
    assert len(result["verified_files"]) == 5
    assert len(result["weight_shards"]) == 2


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "checksum",
        "config_only",
        "unlisted_shard",
        "wrong_url",
        "symlink",
        "forged_receipt",
    ],
)
def test_incomplete_or_unverified_model_never_reaches_a_subprocess(
    tmp_path, monkeypatch, fault
):
    fixture = asset_fixture(tmp_path)
    root, model, _, _, manifest = fixture
    shard = model / "model-00002-of-00002.safetensors"
    if fault == "missing":
        shard.unlink()
    elif fault == "checksum":
        shard.write_bytes(b"broken-weight-shard!!")
    elif fault == "config_only":
        manifest["files"] = manifest["files"][:2]
    elif fault == "unlisted_shard":
        manifest["files"] = manifest["files"][:-1]
    elif fault == "wrong_url":
        manifest["files"][0]["url"] = "https://unrelated.example/config.json"
    elif fault == "forged_receipt":
        payload = b"locally forged model shard"
        shard.write_bytes(payload)
        manifest["files"][-1]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest["files"][-1]["size"] = len(payload)
    else:
        external = tmp_path / "outside"
        shard.rename(external)
        shard.symlink_to(external)
    (root / "download_manifest.json").write_text(json.dumps(manifest))
    calls = fake_runner(monkeypatch, fixture)
    output = tmp_path / "run"
    assert launcher.main(arguments(fixture, output)) == 1
    assert not calls
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stages"][0]["name"] == "verify_model"
    assert (output / "logs/verify_model.stderr.log").read_text()
    assert not (output / "result.json").exists()


def test_orchestration_is_explicit_logged_and_never_claims_quality(
    tmp_path, monkeypatch
):
    fixture = asset_fixture(tmp_path)
    calls = fake_runner(monkeypatch, fixture)
    output = tmp_path / "run"
    assert launcher.main(arguments(fixture, output)) == 0
    assert len(calls) == 4
    assert Path(calls[0][1]).name == "prepare_paper_data.py"
    assert calls[0][calls[0].index("--training-count") + 1] == "128"
    assert calls[0][calls[0].index("--max-prompt-tokens") + 1] == "8192"
    assert calls[-1][1:4] == ["-m", launcher.TRAINING_MODULE, "train"]
    assert calls[-1][-2:] == ["--device", "cpu"]
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert [stage["name"] for stage in status["stages"]] == [
        "verify_model",
        "prepare_data",
        "prepare_train",
        "prepare_validation",
        "verify_before_training",
        "train",
        "verify_adapter",
    ]
    assert not status["manages_gpu_keeper"] and not status["starts_native_benchmark"]
    result = json.loads((output / "result.json").read_text())
    assert result["adapter"].endswith("/training/adapter")
    assert not result["quality_validated"] and not result["paper_results_reproduced"]
    assert "--adapter" in result["next_step"]
    assert result["canonical_adapter_load_verified"]
    for stage in status["stages"]:
        for stream in ("stdout", "stderr"):
            assert (output / "logs" / f"{stage['name']}.{stream}.log").is_file()


def test_download_only_occurs_with_explicit_flag(tmp_path, monkeypatch):
    fixture = asset_fixture(tmp_path)
    calls = fake_runner(monkeypatch, fixture)
    assert launcher.main(arguments(fixture, tmp_path / "run") + ["--download"]) == 0
    assert Path(calls[0][1]).name == "download_paper_assets.py"
    assert calls[0][-2:] == ["--only", "all"]


def test_failure_stops_later_stages_and_keeps_logs(tmp_path, monkeypatch):
    fixture = asset_fixture(tmp_path)
    failure = str(launcher.ROOT / "scripts/prepare_paper_data.py")
    calls = fake_runner(monkeypatch, fixture, failure=failure)
    output = tmp_path / "run"
    assert launcher.main(arguments(fixture, output)) == 1
    assert len(calls) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stages"][-1]["exit_code"] == 7
    assert not (output / "result.json").exists()


def test_config_change_during_preparation_blocks_gpu_stage(tmp_path, monkeypatch):
    fixture = asset_fixture(tmp_path)
    calls = fake_runner(monkeypatch, fixture, changed_config=True)
    assert launcher.main(arguments(fixture, tmp_path / "run")) == 1
    assert len(calls) == 3 and all("train" not in command for command in calls)


@pytest.mark.parametrize("fault", ["partial", "wrong_backbone"])
def test_partial_or_wrong_backbone_adapter_cannot_report_success(
    tmp_path, monkeypatch, fault
):
    fixture = asset_fixture(tmp_path)
    fake_runner(
        monkeypatch,
        fixture,
        adapter_status="partial" if fault == "partial" else "complete",
        bad_backbone=fault == "wrong_backbone",
    )
    output = tmp_path / "run"
    assert launcher.main(arguments(fixture, output)) == 1
    assert not (output / "result.json").exists()


def test_existing_output_and_implicit_gpu_device_are_rejected(tmp_path, monkeypatch):
    fixture = asset_fixture(tmp_path)
    calls = fake_runner(monkeypatch, fixture)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(SystemExit):
        launcher.main(arguments(fixture, output))
    args = arguments(fixture, tmp_path / "run")
    args[-1] = "cuda"
    with pytest.raises(SystemExit):
        launcher.main(args)
    assert not calls


@pytest.mark.parametrize("fault", ["malformed", "nan", "wrong_shape", "one_step"])
def test_unmountable_or_short_training_artifact_cannot_report_success(
    tmp_path, monkeypatch, fault
):
    fixture = asset_fixture(tmp_path)
    fake_runner(monkeypatch, fixture, adapter_fault=fault)
    output = tmp_path / "run"
    assert launcher.main(arguments(fixture, output)) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["stages"][-1]["name"] == "verify_adapter"
    assert status["stages"][-1]["status"] == "failed"
    assert not (output / "result.json").exists()
