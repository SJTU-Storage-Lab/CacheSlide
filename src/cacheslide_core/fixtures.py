"""Deterministic random tiny fixtures, never model-quality/performance data."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .contracts import RequestPlan


def tiny_checkpoint(path: Path) -> None:
    """Write a local six-layer GQA safetensors checkpoint, with no network access."""
    path.mkdir()
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
        "hidden_act": "silu",
    }
    generator = torch.Generator(device="cpu").manual_seed(92)

    def random(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    weights = {
        "model.embed_tokens.weight": random(16, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": random(16, 8),
    }
    for layer in range(6):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = torch.ones(8)
        for name, shape in {
            "self_attn.q_proj": (8, 8),
            "self_attn.k_proj": (4, 8),
            "self_attn.v_proj": (4, 8),
            "self_attn.o_proj": (8, 8),
            "mlp.gate_proj": (12, 8),
            "mlp.up_proj": (12, 8),
            "mlp.down_proj": (8, 12),
        }.items():
            weights[prefix + name + ".weight"] = random(*shape)
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(path / "model.safetensors"))


def tiny_plan(dynamic=(3,)) -> RequestPlan:
    tokens = (1, 2, *dynamic, 5, 6, 7, 8, 9)
    split = 2 + len(dynamic)
    return RequestPlan.parse(
        {
            "version": 1,
            "operation": "recompute",
            "namespace": "cpu-smoke",
            "task_id": "synthetic-only",
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
