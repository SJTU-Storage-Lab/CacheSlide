"""Safe, versioned CoPE/LoRA artifacts tied to the exact local backbone files."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from .contracts import digest
from .position import CoPE


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, max_bytes: int = 8 * 1024 * 1024) -> dict:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"metadata file is too large: {path.name}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"metadata must be an object: {path.name}")
    return data


def backbone_files(model_dir: Path) -> tuple[Path, ...]:
    """Only local safetensors checkpoints; never execute remote modeling code."""
    model_dir = model_dir.resolve(strict=True)
    config = model_dir / "config.json"
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weights = read_json(index).get("weight_map", {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError("invalid safetensors index")
        if any(not isinstance(name, str) for name in weights.values()):
            raise ValueError("checkpoint shard names must be strings")
        names = sorted(set(weights.values()))
    elif (model_dir / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        raise ValueError("a local unquantized safetensors checkpoint is required")
    files = [config] + ([index] if index.exists() else [])
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("checkpoint shard names must be simple relative filenames")
        path = model_dir / name
        if not path.is_file() or not path.resolve().is_relative_to(model_dir):
            raise ValueError("missing checkpoint shard or path outside model directory")
        files.append(path)
    return tuple(files)


def backbone_manifest(model_dir: str | Path) -> dict[str, str]:
    return {p.name: file_sha256(p) for p in backbone_files(Path(model_dir))}


def validate_llama_config(config: dict) -> dict:
    """Initial native family: plain Llama or full-attention Mistral, no quantization."""
    if not isinstance(config.get("model_type"), str) or config["model_type"] not in {
        "llama",
        "mistral",
    }:
        raise ValueError("this adapter supports Llama and full-attention Mistral only")
    if config.get("quantization_config") or config.get("sliding_window"):
        raise ValueError(
            "quantized or sliding-window checkpoints need a separate adapter"
        )
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("only SwiGLU/SILU Llama MLPs are supported")
    if any(
        config.get(k, False) for k in ("attention_bias", "mlp_bias", "qkv_bias", "bias")
    ):
        raise ValueError("biased projections are not supported by this adapter")
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    ):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"invalid model field {key}")
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    if "head_dim" not in config and config["hidden_size"] % heads:
        raise ValueError("hidden_size must divide evenly into default attention heads")
    head_dim = config.get("head_dim", config["hidden_size"] // heads)
    if type(kv_heads) is not int or kv_heads < 1 or heads % kv_heads:
        raise ValueError("invalid grouped-query head count")
    if type(head_dim) is not int or head_dim < 1:
        raise ValueError("invalid head dimension")
    epsilon = config.get("rms_norm_eps", 1e-6)
    if type(epsilon) not in {int, float}:
        raise ValueError("rms_norm_eps must be finite and positive")
    try:
        valid_epsilon = math.isfinite(epsilon) and epsilon > 0
    except OverflowError:
        valid_epsilon = False
    if not valid_epsilon:
        raise ValueError("rms_norm_eps must be finite and positive")
    return dict(config, num_key_value_heads=kv_heads, head_dim=head_dim)


def _adapter_shapes(
    config: dict, rank: int, max_positions: int
) -> dict[str, tuple[int, int]]:
    """Validate adapter geometry without allocating tensors or consuming RNG."""
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if type(max_positions) is not int or max_positions < 2:
        raise ValueError("CoPE max_positions must be an integer of at least two")
    hidden, heads, kv, dim = (
        config[key]
        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    return {
        "qkv_a": (rank, hidden),
        "qkv_b": ((heads + 2 * kv) * dim, rank),
        "out_a": (rank, heads * dim),
        "out_b": (hidden, rank),
        "cope.position_embeddings": (dim, max_positions),
    }


def _validate_tensors(
    weights: dict[str, torch.Tensor], shapes: dict[str, tuple[int, int]]
) -> None:
    if set(weights) != set(shapes):
        raise ValueError("adapter tensor keys do not match model layers")
    for name, expected in shapes.items():
        value = weights[name]
        if tuple(value.shape) != expected:
            raise ValueError(f"adapter tensor shape mismatch: {name}")
        if not value.is_floating_point():
            raise ValueError(f"adapter tensor must use a floating dtype: {name}")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"adapter contains a nonfinite tensor: {name}")


class AttentionAdapter(nn.Module):
    """CoPE plus low-rank QKV/output updates; all backbone weights stay frozen."""

    def __init__(self, config: dict, rank: int, max_positions: int):
        super().__init__()
        config = validate_llama_config(config)
        _adapter_shapes(config, rank, max_positions)
        hidden, heads, kv, dim = (
            config[k]
            for k in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            )
        )
        self.rank = rank
        self.qkv_a = nn.Parameter(torch.empty(rank, hidden))
        self.qkv_b = nn.Parameter(torch.zeros((heads + 2 * kv) * dim, rank))
        self.out_a = nn.Parameter(torch.empty(rank, heads * dim))
        self.out_b = nn.Parameter(torch.zeros(hidden, rank))
        nn.init.kaiming_uniform_(self.qkv_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_a, a=math.sqrt(5))
        self.cope = CoPE(dim, max_positions)

    def qkv_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(hidden, self.qkv_a), self.qkv_b)

    def output_delta(self, attention: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(attention, self.out_a), self.out_b)


class AdapterBundle:
    """Checksum and backbone verification occurs before replacing native RoPE."""

    def __init__(self, path: str | Path, model_dir: str | Path):
        self.path = Path(path).resolve(strict=True)
        meta = read_json(self.path / "manifest.json")
        if meta.get("format") != "cacheslide-adapter-v1":
            raise ValueError("unsupported CacheSlide adapter format")
        if (
            meta.get("trained") is not True
            or type(meta.get("training_steps")) is not int
        ):
            raise ValueError("a trained CoPE/LoRA adapter is required")
        if meta["training_steps"] < 1:
            raise ValueError(
                "zero-step/untrained adapters cannot enable CoPE inference"
            )
        if meta.get("base_files") != backbone_manifest(model_dir):
            raise ValueError("adapter backbone SHA-256 manifest mismatch")
        if meta.get("adapter_sha256") != file_sha256(self.path / "adapter.safetensors"):
            raise ValueError("adapter weight checksum mismatch")
        self.config = validate_llama_config(read_json(Path(model_dir) / "config.json"))
        if meta.get("model_config") != self.config:
            raise ValueError("adapter model geometry differs from the backbone")
        self.metadata = meta
        self.identity = digest(meta)
        self._weights = load_file(str(self.path / "adapter.safetensors"), device="cpu")
        shapes = _adapter_shapes(
            self.config, meta.get("rank"), meta.get("max_positions")
        )
        expected = {
            f"layers.{layer}.{key}": shape
            for layer in range(self.config["num_hidden_layers"])
            for key, shape in shapes.items()
        }
        _validate_tensors(self._weights, expected)

    def layer(self, index: int, *, device=None, dtype=None) -> AttentionAdapter:
        if type(index) is not int or not 0 <= index < self.config["num_hidden_layers"]:
            raise ValueError("adapter layer index is outside the model")
        module = AttentionAdapter(
            self.config, self.metadata["rank"], self.metadata["max_positions"]
        )
        prefix = f"layers.{index}."
        module.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in self._weights.items()
                if k.startswith(prefix)
            },
            strict=True,
        )
        module.requires_grad_(False)
        return module.to(device=device, dtype=dtype)


def save_adapter(
    output: str | Path,
    model_dir: str | Path,
    layers: list[AttentionAdapter],
    *,
    training_steps: int,
    training_tokens: int,
    losses: list[float],
) -> Path:
    """Write a new artifact directory; existing files are never overwritten."""
    if (
        type(training_steps) is not int
        or training_steps < 1
        or type(training_tokens) is not int
        or training_tokens < 1
        or not losses
    ):
        raise ValueError("adapter export requires actual training updates")
    if not all(type(x) in {int, float} and math.isfinite(x) for x in losses):
        raise ValueError("training produced nonfinite loss")
    config = validate_llama_config(read_json(Path(model_dir) / "config.json"))
    if len(layers) != config["num_hidden_layers"]:
        raise ValueError("adapter layer count mismatch")
    if not all(isinstance(layer, AttentionAdapter) for layer in layers):
        raise ValueError("all exported layers must be attention adapters")
    rank, max_positions = layers[0].rank, layers[0].cope.max_positions
    shapes = _adapter_shapes(config, rank, max_positions)
    for layer in layers:
        if layer.rank != rank or layer.cope.max_positions != max_positions:
            raise ValueError("adapter layers must share rank and position geometry")
        _validate_tensors(layer.state_dict(), shapes)
    # Validate every artifact input before creating the output directory.
    base_files = backbone_manifest(model_dir)
    weights = {
        f"layers.{i}.{key}": value.detach().to("cpu", copy=True).contiguous()
        for i, layer in enumerate(layers)
        for key, value in layer.state_dict().items()
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_file(weights, str(output / "adapter.safetensors"))
    metadata = {
        "format": "cacheslide-adapter-v1",
        "trained": True,
        "training_steps": training_steps,
        "training_tokens": training_tokens,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "rank": rank,
        "max_positions": max_positions,
        "model_config": config,
        "base_files": base_files,
        "adapter_sha256": file_sha256(output / "adapter.safetensors"),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output
