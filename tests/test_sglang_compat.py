import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheslide_sglang import compat


def valid_args():
    return SimpleNamespace(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        nnodes=1,
        max_running_requests=1,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        disable_prefill_cuda_graph=True,
        disable_decode_cuda_graph=True,
        attention_backend="cacheslide",
        skip_tokenizer_init=True,
        prefill_attention_backend="cacheslide",
        decode_attention_backend="cacheslide",
        device="cuda",
        dtype="float32",
        kv_cache_dtype="auto",
        num_continuous_decode_steps=1,
        json_model_override_args='{"architectures":["CacheSlideLlamaForCausalLM"]}',
    )


def test_valid_scope_and_manifest():
    compat.validate_engine_config(valid_args())
    manifest = compat.compatibility_manifest()
    assert manifest["sglang_version"] == "0.5.19"
    assert len(manifest["sha256"]) == 30
    assert manifest["native_gpu_validated"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("tp_size", 2),
        ("tp_size", True),
        ("max_running_requests", 2),
        ("disable_radix_cache", False),
        ("disable_overlap_schedule", False),
        ("disable_prefill_cuda_graph", False),
        ("disable_decode_cuda_graph", False),
        ("skip_tokenizer_init", False),
        ("attention_backend", "torch_native"),
        ("dtype", "auto"),
        ("kv_cache_dtype", "fp8"),
        ("chunked_prefill_size", 128),
        ("enable_unified_memory", True),
        ("enable_hierarchical_cache", True),
        ("enable_lora", True),
        ("speculative_algorithm", "EAGLE"),
        ("quantization", "fp8"),
        ("disaggregation_mode", "decode"),
        ("json_model_override_args", "{}"),
    ],
)
def test_reject_unsupported_modes(key, value):
    args = valid_args()
    setattr(args, key, value)
    with pytest.raises(compat.CompatibilityError):
        compat.validate_engine_config(args)


def test_source_identity_version_and_mutation(tmp_path, monkeypatch):
    path = tmp_path / "sglang" / "a.py"
    path.parent.mkdir()
    path.write_text("audited\n")
    manifest = {
        "sglang_version": "0.5.19",
        "sha256": {"sglang/a.py": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    monkeypatch.setattr(compat, "compatibility_manifest", lambda: manifest)
    assert compat.verify_sglang_sources(tmp_path, version="0.5.19") == manifest
    with pytest.raises(compat.CompatibilityError, match="required"):
        compat.verify_sglang_sources(tmp_path, version="0.5.20.dev0")
    path.write_text("changed\n")
    with pytest.raises(compat.CompatibilityError, match="modified"):
        compat.verify_sglang_sources(tmp_path, version="0.5.19")
    path.unlink()
    with pytest.raises(compat.CompatibilityError, match="missing"):
        compat.verify_sglang_sources(tmp_path, version="0.5.19")


def test_import_shadow_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compat.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(
            version="0.5.19",
            locate_file=lambda _: tmp_path / "installed/sglang/__init__.py",
        ),
    )
    monkeypatch.setattr(
        compat.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(tmp_path / "shadow/sglang/__init__.py")),
    )
    with pytest.raises(compat.CompatibilityError, match="shadowed"):
        compat.verify_installed_sglang()


def test_actual_audited_source_if_available():
    root = os.environ.get("CACHESLIDE_SGLANG_SOURCE")
    if not root:
        pytest.skip("set CACHESLIDE_SGLANG_SOURCE to validate official source snapshot")
    assert Path(root).is_dir()
    assert compat.verify_compatibility(root)["sglang_git_commit"] == (
        "0bcd822377da7b5718e674eaf9c870d349424dd1"
    )
