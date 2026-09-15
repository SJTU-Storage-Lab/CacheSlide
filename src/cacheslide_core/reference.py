"""Portable frozen-backbone Llama/Mistral reference for CoPE adapter training.

Loads local, unquantized safetensors only. No Transformers, pickle checkpoints,
remote model code, native RoPE, quantization or sliding-window attention is used.
The linear/norm/layer interfaces intentionally match the relevant vLLM Llama
interfaces so a CPU reference can exercise the same selective execution code.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from .artifacts import (
    AdapterBundle,
    AttentionAdapter,
    backbone_files,
    read_json,
    validate_llama_config,
)
from .position import CoPE, cope_attention

QKObserver = Callable[[int, Tensor, Tensor, CoPE], None]
AttentionHandler = Callable[
    [int, Tensor, Tensor, Tensor, Tensor, AttentionAdapter], Tensor
]


class FrozenLinear(nn.Module):
    """Bias-free linear with the native ``(output, bias)`` return convention."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, None]:
        return F.linear(hidden, self.weight), None


class FrozenRMSNorm(nn.Module):
    def __init__(self, weight: Tensor, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.variance_epsilon = epsilon

    def forward(
        self, hidden: Tensor, residual: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        combined = hidden if residual is None else hidden + residual
        dtype = torch.float64 if combined.dtype == torch.float64 else torch.float32
        normalized = combined.to(dtype)
        inverse = torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        normalized = (normalized * inverse).to(combined.dtype) * self.weight
        return normalized if residual is None else (normalized, combined)


class SiluAndMul(nn.Module):
    def forward(self, hidden: Tensor) -> Tensor:
        gate, up = hidden.chunk(2, dim=-1)
        return F.silu(gate) * up


class ReferenceMLP(nn.Module):
    def __init__(self, gate: Tensor, up: Tensor, down: Tensor) -> None:
        super().__init__()
        self.gate_up_proj = FrozenLinear(torch.cat((gate, up), dim=0))
        self.down_proj = FrozenLinear(down)
        self.act_fn = SiluAndMul()

    def forward(self, hidden: Tensor) -> Tensor:
        gate_up, _ = self.gate_up_proj(hidden)
        output, _ = self.down_proj(self.act_fn(gate_up))
        return output


class ReferenceAttention(nn.Module):
    def __init__(
        self,
        config: dict,
        layer_index: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        adapter: AttentionAdapter,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.qkv_proj = FrozenLinear(torch.cat((q, k, v), dim=0))
        self.o_proj = FrozenLinear(output)
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.layer_index = layer_index
        self.query_chunk_size = query_chunk_size
        self.attention_handler: AttentionHandler | None = None
        self.bind_adapter(adapter)

    def bind_adapter(self, adapter: AttentionAdapter) -> None:
        # ReferenceLlama.adaptors owns the module and performs device/dtype moves.
        # A weak reference avoids registering a second copy in the backbone tree.
        self._adapter_ref = weakref.ref(adapter)

    @property
    def adapter(self) -> AttentionAdapter:
        adapter = self._adapter_ref()
        if adapter is None:
            raise RuntimeError("the owning reference model's adapter was released")
        return adapter

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        *,
        observer: QKObserver | None = None,
    ) -> Tensor:
        adapter = self.adapter
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv + adapter.qkv_delta(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if observer is not None:
            observer(self.layer_index, q, k, adapter.cope)
        if self.attention_handler is None:
            attention = cope_attention(
                q,
                k,
                v,
                adapter.cope,
                positions,
                key_positions=positions,
                query_chunk_size=self.query_chunk_size,
            )
        else:
            attention = self.attention_handler(
                self.layer_index, positions, q, k, v, adapter
            )
        if attention.shape != q.shape:
            raise ValueError(
                "attention handler must return [token,query_head,head_dim]"
            )
        attention = attention.reshape(hidden_states.shape[0], self.q_size)
        output, _ = self.o_proj(attention)
        return output + adapter.output_delta(attention)


class ReferenceDecoderLayer(nn.Module):
    def __init__(
        self,
        attention: ReferenceAttention,
        mlp: ReferenceMLP,
        input_norm: Tensor,
        post_norm: Tensor,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = FrozenRMSNorm(input_norm, epsilon)
        self.post_attention_layernorm = FrozenRMSNorm(post_norm, epsilon)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        residual: Tensor | None = None,
        *,
        observer: QKObserver | None = None,
    ) -> tuple[Tensor, Tensor]:
        """vLLM-style fused residual representation: output is MLP delta + residual."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, observer=observer)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class ReferenceLlama(nn.Module):
    def __init__(
        self,
        config: dict,
        weights: dict[str, Tensor],
        *,
        rank: int,
        max_positions: int,
        query_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.config = validate_llama_config(config)
        if type(query_chunk_size) is not int or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        config = self.config
        self.logit_scale = config.get("logit_scale", 1.0)
        hidden = config["hidden_size"]
        head_dim = config["head_dim"]
        heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        intermediate, vocab = config["intermediate_size"], config["vocab_size"]
        epsilon = config.get("rms_norm_eps", 1e-6)
        if (
            not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("rms_norm_eps must be finite and positive")
        used = set()

        def take(name: str, shape: tuple[int, ...]) -> Tensor:
            if name not in weights:
                raise ValueError(f"checkpoint is missing required weight {name}")
            value = weights[name]
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"checkpoint weight has invalid shape/dtype: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"checkpoint weight is nonfinite: {name}")
            used.add(name)
            return value

        embedding = take("model.embed_tokens.weight", (vocab, hidden))
        self.embed_tokens = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.adapters = nn.ModuleList(
            AttentionAdapter(config, rank, max_positions)
            for _ in range(config["num_hidden_layers"])
        )
        layers = []
        for index in range(config["num_hidden_layers"]):
            prefix = f"model.layers.{index}."
            attention = ReferenceAttention(
                config,
                index,
                take(prefix + "self_attn.q_proj.weight", (heads * head_dim, hidden)),
                take(prefix + "self_attn.k_proj.weight", (kv_heads * head_dim, hidden)),
                take(prefix + "self_attn.v_proj.weight", (kv_heads * head_dim, hidden)),
                take(prefix + "self_attn.o_proj.weight", (hidden, heads * head_dim)),
                self.adapters[index],
                query_chunk_size,
            )
            mlp = ReferenceMLP(
                take(prefix + "mlp.gate_proj.weight", (intermediate, hidden)),
                take(prefix + "mlp.up_proj.weight", (intermediate, hidden)),
                take(prefix + "mlp.down_proj.weight", (hidden, intermediate)),
            )
            layers.append(
                ReferenceDecoderLayer(
                    attention,
                    mlp,
                    take(prefix + "input_layernorm.weight", (hidden,)),
                    take(prefix + "post_attention_layernorm.weight", (hidden,)),
                    epsilon,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = FrozenRMSNorm(take("model.norm.weight", (hidden,)), epsilon)
        if config.get("tie_word_embeddings", False):
            if "lm_head.weight" in weights:
                lm_weight = take("lm_head.weight", (vocab, hidden))
                if not torch.equal(lm_weight, embedding):
                    raise ValueError("tied lm_head differs from token embeddings")
            lm_weight = embedding
        else:
            lm_weight = take("lm_head.weight", (vocab, hidden))
        self.lm_head = FrozenLinear(lm_weight)
        # Old HF checkpoints can persist inverse RoPE frequencies. They are
        # intentionally unused when training/inferencing this explicit CoPE model.
        allowed_unused = {"model.rotary_emb.inv_freq"} | {
            f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
            for i in range(config["num_hidden_layers"])
        }
        unexpected = set(weights) - used - allowed_unused
        if unexpected:
            raise ValueError(
                f"unsupported checkpoint weight keys: {sorted(unexpected)[:4]}"
            )

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        rank: int = 8,
        max_positions: int = 256,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        query_chunk_size: int = 128,
    ) -> ReferenceLlama:
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("reference model dtype must be floating point")
        model_dir = Path(model_dir).resolve(strict=True)
        files = backbone_files(model_dir)
        config = validate_llama_config(read_json(model_dir / "config.json"))
        weights = {}
        for path in files:
            if path.suffix != ".safetensors":
                continue
            shard = load_file(str(path), device="cpu")
            if set(weights) & set(shard):
                raise ValueError("checkpoint shards contain duplicate tensor names")
            weights.update(shard)
        model = cls(
            config,
            weights,
            rank=rank,
            max_positions=max_positions,
            query_chunk_size=query_chunk_size,
        )
        return model.to(device=device, dtype=dtype)

    def load_adapters(self, bundle: AdapterBundle) -> None:
        """Install verified trained layers; all loaded adapter parameters are frozen."""
        if bundle.config != self.config:
            raise ValueError("adapter model configuration differs from reference model")
        weight = self.embed_tokens.weight
        self.adapters = nn.ModuleList(
            bundle.layer(i, device=weight.device, dtype=weight.dtype)
            for i in range(len(self.layers))
        )
        for layer, adapter in zip(self.layers, self.adapters, strict=True):
            layer.self_attn.bind_adapter(adapter)

    def get_input_embeddings(self, token_ids: Tensor) -> Tensor:
        return self.embed_tokens(token_ids)

    def forward(
        self, token_ids: Tensor, *, observer: QKObserver | None = None
    ) -> Tensor:
        """One unpadded causal sequence -> [sequence,vocabulary] next-token logits."""
        if (
            token_ids.ndim != 1
            or token_ids.dtype != torch.long
            or token_ids.numel() < 1
            or (token_ids < 0).any()
            or (token_ids >= self.config["vocab_size"]).any()
        ):
            raise ValueError("token_ids must be a nonempty in-vocabulary int64 vector")
        if token_ids.device != self.embed_tokens.weight.device:
            raise ValueError("tokens and reference model must share a device")
        positions = torch.arange(token_ids.numel(), device=token_ids.device)
        hidden_states, residual = self.embed_tokens(token_ids), None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, observer=observer
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        logits, _ = self.lm_head(hidden_states)
        # Match native LlamaForCausalLM/LogitsProcessor: scale after the head,
        # before the training loss or inference sampling consumes the logits.
        if self.logit_scale != 1.0:
            logits = logits * self.logit_scale
        return logits
