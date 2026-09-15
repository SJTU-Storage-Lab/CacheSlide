"""Opt-in native vLLM model; importing this module requires the pinned engine.

No native class is monkey-patched. Native projections, RMSNorm, SwiGLU,
checkpoint loader, sampler and KV block ownership are retained. Only this
explicit architecture uses trained CoPE and the selective layer loop.
"""

from __future__ import annotations

import weakref

import torch
from torch import nn
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from vllm.model_executor.models.utils import extract_layer_index

from cacheslide_core.artifacts import AdapterBundle
from cacheslide_core.config import CacheSlideSettings
from cacheslide_core.runtime import CacheSlideRuntime

from .compat import (
    validate_engine_config,
    validate_native_model_config,
    verify_installed_vllm,
)
from .integration import current_step
from .paged import NativePagedKV


class CacheSlideAttention(LlamaAttention):
    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)
        self.layer_index = extract_layer_index(prefix)
        self.layer_name = f"{prefix}.attn"
        self._runtime_ref = None
        self._adapter_ref = None

    def bind(self, runtime, adapter):
        self._runtime_ref = weakref.ref(runtime)
        self._adapter_ref = weakref.ref(adapter)

    def forward(self, positions, hidden_states):
        runtime, adapter = self._runtime_ref(), self._adapter_ref()
        if runtime is None or adapter is None:
            raise RuntimeError("CacheSlide runtime or trained adapter was released")
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv + adapter.qkv_delta(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), -1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # CoPE is an attention-score bias, not a rotary transform of cached K.
        attention = runtime.attention(self.layer_index, positions, q, k, v, adapter)
        attention = attention.reshape(hidden_states.shape[0], self.q_size)
        output, _ = self.o_proj(attention)
        return output + adapter.output_delta(attention)


class CacheSlideDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config, prefix="", **kwargs):
        super().__init__(
            vllm_config, prefix=prefix, attn_layer_type=CacheSlideAttention, **kwargs
        )


class CacheSlideModel(LlamaModel):
    def __init__(self, *, vllm_config, prefix="", layer_type=None):
        verify_installed_vllm()
        validate_engine_config(vllm_config)
        settings = CacheSlideSettings.from_mapping(
            vllm_config.additional_config.get("cacheslide", {})
        )
        model_path = vllm_config.model_config.model
        # Verify weights before native CUDA model allocation/positional replacement.
        bundle = AdapterBundle(settings.artifact_path, model_path)
        hf = vllm_config.model_config.hf_config
        validate_native_model_config(hf, bundle.config)
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=CacheSlideDecoderLayer
        )
        if self.start_layer != 0 or self.end_layer != len(self.layers):
            raise ValueError("CacheSlide does not support pipeline partitions")
        parameter = self.embed_tokens.weight
        self.cacheslide_adapters = nn.ModuleList(
            [
                bundle.layer(index, device=parameter.device, dtype=parameter.dtype)
                for index in range(len(self.layers))
            ]
        )
        self.cacheslide_runtime = CacheSlideRuntime(
            settings,
            bundle,
            arena_factory=self._arena,
            model_dtype=parameter.dtype,
            execution_identity="vllm:0.29.0:98dff2a81d747d1dba01a47f939f48c3526d4206",
        )
        for layer, adapter in zip(self.layers, self.cacheslide_adapters, strict=True):
            layer.self_attn.bind(self.cacheslide_runtime, adapter)

    def _arena(self, layer_index, prompt_length, max_selected, *, existing=None):
        layer = self.layers[layer_index].self_attn
        metadata, owner, cache, slot_mapping = get_attention_context(layer.layer_name)
        if metadata is None or owner is not layer.attn:
            raise ValueError("CacheSlide requires native per-layer attention metadata")
        if metadata.block_table.ndim != 2 or metadata.block_table.shape[0] != 1:
            raise ValueError("CacheSlide requires one native block table row")
        row = metadata.block_table[0]
        if existing is not None:
            if existing.cache.data_ptr() != cache.data_ptr():
                raise ValueError("native KV backing changed without full prefill")
            existing.update_block_table(row)
        step = current_step()
        if step is None:
            raise ValueError("native KV arenas require a real request context")
        arena = existing or NativePagedKV(
            cache,
            row,
            prompt_length,
            max_selected,
            request_id=step.request_id,
            layer_index=layer_index,
        )
        # Cross-check manager metadata against the actual kernel block geometry.
        count = len(step.positions)
        if slot_mapping is None or len(slot_mapping) < count:
            raise ValueError("native slot mapping does not cover the scheduled rows")
        expected = torch.tensor(
            [arena.canonical_slot(i) for i in step.positions],
            device=slot_mapping.device,
            dtype=slot_mapping.dtype,
        )
        if not torch.equal(slot_mapping[:count], expected):
            raise ValueError("native slot mapping and kernel block table disagree")
        return arena

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        step = current_step()
        if step is None:
            # Native memory profiling still exercises trained attention but must
            # not create persistent request state or dereference empty KV pages.
            # Dummy runner positions can all be zero; there is no real request
            # layout, so use a synthetic causal sequence of the same allocation.
            positions = torch.arange(len(positions), device=positions.device)
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        if intermediate_tensors is not None or inputs_embeds is not None or kwargs:
            raise ValueError("unsupported CacheSlide forward inputs")
        hidden = self.embed_input_ids(input_ids)
        return self.cacheslide_runtime.run(self, hidden, positions, step)


class CacheSlideLlamaForCausalLM(LlamaForCausalLM):
    def _init_model(self, vllm_config, prefix="", layer_type=None):
        return CacheSlideModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # These parameters were independently verified and loaded from the trained
        # artifact, and deliberately do not occur in the frozen HF checkpoint.
        loaded.update(
            name
            for name, _ in self.named_parameters()
            if name.startswith("model.cacheslide_adapters.")
        )
        return loaded

    def cacheslide_release_request(self, request_id):
        self.model.cacheslide_runtime.release(request_id)
