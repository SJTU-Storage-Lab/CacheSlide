"""Shared trained-model geometry and numerics contract; no engine imports."""

from __future__ import annotations

from typing import Any


def validate_native_model_config(hf: Any, trained: dict[str, Any]) -> None:
    """Bind native HF overrides to the geometry and numerics used for training."""
    if getattr(hf, "model_type", trained.get("model_type", "llama")) not in {
        "llama",
        "mistral",
    }:
        raise ValueError("native model_type must be llama or full-attention mistral")
    if getattr(hf, "hidden_act", "silu") != "silu":
        raise ValueError("native hidden_act must be silu, as in the trained backbone")
    for name in ("final_logit_softcapping", "attn_logit_softcapping"):
        if getattr(hf, name, None) not in (None, 0):
            raise ValueError(f"native {name} is absent from the trained reference")
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
