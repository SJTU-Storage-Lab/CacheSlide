"""Fail-closed source and configuration boundary for SGLang 0.5.19."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

from cacheslide_core.model_validation import validate_native_model_config

__all__ = [
    "CompatibilityError",
    "compatibility_manifest",
    "verify_sglang_sources",
    "verify_installed_sglang",
    "verify_compatibility",
    "validate_engine_config",
    "validate_native_model_config",
]


class CompatibilityError(RuntimeError):
    """The native engine differs from the explicitly audited contract."""


def compatibility_manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("compatibility.json").read_text())


def verify_sglang_sources(source_root: str | Path, *, version: str) -> dict:
    manifest = compatibility_manifest()
    if version != manifest["sglang_version"]:
        raise CompatibilityError(f"SGLang 0.5.19 required; found {version}")
    root = Path(source_root).resolve()
    mismatches = []
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file():
            mismatches.append(relative + " (missing)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative + " (modified)")
    if mismatches:
        raise CompatibilityError(
            "Audited SGLang source mismatch: " + ", ".join(mismatches)
        )
    return manifest


def verify_installed_sglang() -> dict:
    """Inspect package files without importing SGLang or accepting source overrides."""
    try:
        distribution = importlib.metadata.distribution("sglang")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CompatibilityError("Install the pinned SGLang 0.5.19 extra") from exc
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise CompatibilityError("Cannot locate the SGLang package")
    origin = Path(spec.origin).resolve()
    installed = Path(distribution.locate_file("sglang/__init__.py")).resolve()
    if installed != origin:
        raise CompatibilityError("SGLang import is shadowed by another source tree")
    return verify_sglang_sources(origin.parent.parent, version=distribution.version)


def verify_compatibility(source_root: str | Path | None = None) -> dict:
    """Read-only CLI/source audit; native launch always verifies the installation."""
    if source_root is None:
        return verify_installed_sglang()
    root = Path(source_root)
    if (root / "python" / "sglang").is_dir():
        root = root / "python"
    return verify_sglang_sources(
        root, version=compatibility_manifest()["sglang_version"]
    )


def validate_engine_config(server_args: Any) -> None:
    """Only the audited single-request, single-GPU eager configuration is accepted."""
    required = {
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "nnodes": 1,
        "max_running_requests": 1,
        "chunked_prefill_size": -1,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "disable_prefill_cuda_graph": True,
        "disable_decode_cuda_graph": True,
        "skip_tokenizer_init": True,
        "attention_backend": "cacheslide",
        "prefill_attention_backend": "cacheslide",
        "decode_attention_backend": "cacheslide",
        "device": "cuda",
        "kv_cache_dtype": "auto",
        "num_continuous_decode_steps": 1,
    }
    errors = []
    for name, expected in required.items():
        actual = getattr(server_args, name, None)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{name} must be {expected!r}")
    for name in (
        "enable_hierarchical_cache",
        "enable_hisparse",
        "enable_unified_memory",
        "enable_page_major_kv_layout",
        "enable_torch_compile",
        "enable_lora",
        "enable_dp_attention",
        "enable_two_batch_overlap",
        "enable_dynamic_chunking",
        "enable_pdmux",
        "enable_multi_layer_eagle",
        "enable_deterministic_inference",
    ):
        if getattr(server_args, name, False):
            errors.append(name + " is unsupported")
    for name in ("quantization", "speculative_algorithm", "dllm_algorithm"):
        if getattr(server_args, name, None) is not None:
            errors.append(name + " is unsupported")
    if getattr(server_args, "disaggregation_mode", "null") != "null":
        errors.append("disaggregation_mode must be null")
    if getattr(server_args, "dtype", None) not in {"float32", "float16", "bfloat16"}:
        errors.append("dtype must be explicit float32, float16 or bfloat16")
    try:
        overrides = json.loads(server_args.json_model_override_args)
    except (AttributeError, TypeError, ValueError):
        overrides = {}
    if overrides != {"architectures": ["CacheSlideLlamaForCausalLM"]}:
        errors.append(
            "only the CacheSlideLlamaForCausalLM architecture override is allowed"
        )
    if errors:
        raise CompatibilityError(
            "Unsupported CacheSlide SGLang configuration: " + "; ".join(errors)
        )
