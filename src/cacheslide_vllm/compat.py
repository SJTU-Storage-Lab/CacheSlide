"""Fail-closed compatibility boundary for the audited vLLM 0.29.0 sources."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any


class CompatibilityError(RuntimeError):
    pass


def compatibility_manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("compatibility.json").read_text())


def verify_vllm_sources(source_root: str | Path, *, version: str) -> dict[str, Any]:
    """Check a source tree or site-packages root without importing native CUDA code."""
    manifest = compatibility_manifest()
    if version != manifest["vllm_version"]:
        raise CompatibilityError(
            f"CacheSlide requires vLLM {manifest['vllm_version']}; got {version}"
        )
    root = Path(source_root).resolve()
    mismatches = []
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file():
            mismatches.append(f"{relative} (missing)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(f"{relative} (modified)")
    if mismatches:
        raise CompatibilityError(
            "CacheSlide native source contract differs from audited commit "
            f"{manifest['vllm_git_commit']}: " + ", ".join(mismatches)
        )
    return manifest


def verify_installed_vllm() -> dict[str, Any]:
    """Verify the installed distribution and imported package, never a test override."""
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CompatibilityError("Install the pinned vLLM 0.29.0 engine extra") from exc
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise CompatibilityError("Cannot locate the installed vLLM package")
    return verify_vllm_sources(Path(spec.origin).parent.parent, version=version)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def validate_native_model_config(hf: Any, trained: dict[str, Any]) -> None:
    """Bind native HF overrides to the geometry and numerics used for training."""
    if getattr(hf, "model_type", trained.get("model_type", "llama")) not in {
        "llama",
        "mistral",
    }:
        raise ValueError("native model_type must be llama or full-attention mistral")
    if getattr(hf, "hidden_act", "silu") != "silu":
        raise ValueError("native hidden_act must be silu, as in the trained backbone")
    for name in ("attention_bias", "mlp_bias", "qkv_bias", "bias"):
        if getattr(hf, name, False):
            raise ValueError(f"native {name} is unsupported by the trained backbone")
    native_tied = getattr(hf, "tie_word_embeddings", False)
    trained_tied = trained.get("tie_word_embeddings", False)
    if (
        type(native_tied) is not bool
        or type(trained_tied) is not bool
        or native_tied != trained_tied
    ):
        raise ValueError(
            "native model and trained artifact disagree: tie_word_embeddings"
        )
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    ):
        if getattr(hf, name, None) != trained[name]:
            raise ValueError(f"native model and trained artifact disagree: {name}")
    native_head_dim = getattr(hf, "head_dim", None) or (
        hf.hidden_size // hf.num_attention_heads
    )
    if native_head_dim != trained["head_dim"]:
        raise ValueError("native model and trained artifact disagree: head_dim")
    for name, default in (("rms_norm_eps", 1e-6), ("logit_scale", 1.0)):
        native = getattr(hf, name, default)
        if type(native) not in (int, float) or native != trained.get(name, default):
            raise ValueError(f"native model and trained artifact disagree: {name}")


def validate_engine_config(config: Any) -> None:
    """Validate modes only; the model validates its trained CoPE artifact separately."""
    errors: list[str] = []
    model = config.model_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    compilation = config.compilation_config
    attention = config.attention_config
    hf_config = model.hf_config

    if type(getattr(config, "use_v2_model_runner", None)) is not bool:
        errors.append("native model runner selection must be an explicit boolean")
    if getattr(hf_config, "architectures", None) != ["CacheSlideLlamaForCausalLM"]:
        errors.append("select only the CacheSlideLlamaForCausalLM architecture")
    if scheduler.max_num_seqs != 1:
        errors.append("max_num_seqs must be 1")
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "nnodes",
    ):
        if getattr(parallel, field, 1) != 1:
            errors.append(f"{field} must be 1")
    for field in ("enable_dbo", "use_ubatching", "enable_expert_parallel"):
        if getattr(parallel, field, False):
            errors.append(f"{field} is unsupported")
    if scheduler.async_scheduling is not False:
        errors.append("async_scheduling must be explicitly disabled")
    if scheduler.enable_chunked_prefill:
        errors.append("chunked prefill is unsupported")
    if getattr(scheduler, "scheduler_cls", None) is not None:
        errors.append("custom schedulers are unsupported")
    if cache.enable_prefix_caching:
        errors.append("automatic prefix caching is unsupported")
    if getattr(cache, "kv_sharing_fast_prefill", False):
        errors.append("KV sharing fast prefill is unsupported")
    if cache.cache_dtype not in ("auto", "float16", "bfloat16"):
        errors.append("quantized KV caches are unsupported")
    if cache.cache_dtype != "auto" and str(model.dtype) != f"torch.{cache.cache_dtype}":
        errors.append("explicit KV dtype must match the model dtype")
    if not model.enforce_eager:
        errors.append("enforce_eager must be true")
    if _enum_name(compilation.mode) not in ("NONE", "0"):
        errors.append("compilation mode must be NONE/0")
    if _enum_name(compilation.cudagraph_mode) not in ("NONE", "0"):
        errors.append("CUDA graph mode must be NONE")
    if _enum_name(attention.backend) != "FLASH_ATTN":
        errors.append("attention backend must be explicitly FLASH_ATTN")
    if getattr(attention, "backend_per_kind", None):
        errors.append("per-kind attention backend overrides are unsupported")
    if str(model.dtype) not in ("torch.float16", "torch.bfloat16"):
        errors.append("model dtype must be float16 or bfloat16")
    if getattr(model, "quantization", None) is not None:
        errors.append("quantized model weights are unsupported")
    if getattr(hf_config, "quantization_config", None):
        errors.append("quantized Hugging Face artifacts are unsupported")
    if getattr(hf_config, "is_causal", True) is not True:
        errors.append("bidirectional models are unsupported")
    if getattr(hf_config, "sliding_window", None) is not None:
        errors.append("sliding-window model attention is unsupported")
    if getattr(hf_config, "layer_types", None) not in (None, []):
        errors.append("heterogeneous attention layers are unsupported")
    if getattr(model, "runner_type", "generate") != "generate":
        errors.append("only the generation runner is supported")
    for field in (
        "speculative_config",
        "lora_config",
        "quant_config",
        "kv_transfer_config",
        "ec_transfer_config",
        "weight_transfer_config",
    ):
        if getattr(config, field, None) is not None:
            errors.append(f"{field} is unsupported")
    offload = getattr(config, "offload_config", None)
    if offload is not None:
        if getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0):
            errors.append("CPU weight offloading is unsupported")
        if getattr(getattr(offload, "prefetch", None), "offload_group_size", 0):
            errors.append("prefetch weight offloading is unsupported")
    if getattr(model, "enable_sleep_mode", False):
        errors.append("worker sleep mode is unsupported")
    if errors:
        raise CompatibilityError(
            "Unsupported CacheSlide engine configuration: " + "; ".join(errors)
        )
