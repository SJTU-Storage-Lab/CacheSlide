"""Opt-in SGLang Llama/full-attention Mistral with native frozen projections.

The native outer LM, checkpoint loading and logits processor are inherited.
Existing model modules are moved into the wrapper without a second allocation.
Only the attention implementation and selective layer dispatch are replaced.
"""

from __future__ import annotations

import weakref

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.models.llama import LlamaAttention, LlamaForCausalLM, LlamaModel
from torch import nn

from cacheslide_core.artifacts import AdapterBundle
from cacheslide_core.context import current_step
from cacheslide_core.model_validation import validate_native_model_config
from cacheslide_core.runtime import CacheSlideRuntime
from cacheslide_core.storage import StaleCompletionError
from cacheslide_sglang import integration
from cacheslide_sglang.compat import verify_installed_sglang
from cacheslide_sglang.pool import SGLangTokenKV


class CacheSlideAttention(LlamaAttention):
    def __init__(self, native, layer_index, runtime, adapter):
        nn.Module.__init__(self)
        for name in ("q_size", "kv_size", "num_heads", "num_kv_heads", "head_dim"):
            setattr(self, name, getattr(native, name))
        self.qkv_proj, self.o_proj, self.attn = (
            native.qkv_proj,
            native.o_proj,
            native.attn,
        )
        self.layer_index = layer_index
        self._runtime_ref, self._adapter_ref = (
            weakref.ref(runtime),
            weakref.ref(adapter),
        )

    def forward(self, positions, hidden_states, forward_batch):
        if forward_batch is not integration.current_forward_batch():
            raise ValueError(
                "attention requires the currently bound native ForwardBatch"
            )
        runtime, adapter = self._runtime_ref(), self._adapter_ref()
        if runtime is None or adapter is None:
            raise RuntimeError("CacheSlide runtime or trained adapter was released")
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv + adapter.qkv_delta(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        contextual = runtime.attention(self.layer_index, positions, q, k, v, adapter)
        contextual = contextual.reshape(hidden_states.shape[0], self.q_size)
        output, _ = self.o_proj(contextual)
        return output + adapter.output_delta(contextual)


class _BoundLayer:
    """A non-module view preserves native checkpoint parameter paths."""

    def __init__(self, layer, forward_batch):
        self.layer, self.forward_batch = layer, forward_batch

    def __call__(self, positions, hidden, residual):
        return self.layer(positions, hidden, self.forward_batch, residual)


class _BoundModel:
    def __init__(self, model, forward_batch):
        self.layers = tuple(_BoundLayer(layer, forward_batch) for layer in model.layers)
        self.norm = model.norm


class CacheSlideModel(LlamaModel):
    def __init__(self, native, bundle, settings):
        nn.Module.__init__(self)
        for name in (
            "config",
            "padding_idx",
            "vocab_size",
            "pp_group",
            "start_layer",
            "end_layer",
            "layers_to_capture",
        ):
            setattr(self, name, getattr(native, name))
        self.embed_tokens, self.layers, self.norm = (
            native.embed_tokens,
            native.layers,
            native.norm,
        )
        if self.start_layer != 0 or self.end_layer != len(self.layers):
            raise ValueError("CacheSlide does not support pipeline partitions")
        parameter = self.embed_tokens.weight
        self.cacheslide_adapters = nn.ModuleList(
            bundle.layer(i, device=parameter.device, dtype=parameter.dtype)
            for i in range(len(self.layers))
        )
        self.cacheslide_runtime = CacheSlideRuntime(
            settings,
            bundle,
            arena_factory=self._arena,
            model_dtype=parameter.dtype,
            execution_identity="sglang:0.5.19:0bcd822377da7b5718e674eaf9c870d349424dd1",
        )
        for index, (layer, adapter) in enumerate(
            zip(self.layers, self.cacheslide_adapters, strict=True)
        ):
            layer.self_attn = CacheSlideAttention(
                layer.self_attn, index, self.cacheslide_runtime, adapter
            )

    def _arena(self, layer_index, prompt_length, max_selected, *, existing=None):
        request, batch = (
            integration.current_request(),
            integration.current_forward_batch(),
        )
        step = current_step()
        if request is None or batch is None or step is None:
            raise ValueError("native KV requires a bound real request")
        if request.request_id != step.request_id:
            raise StaleCompletionError("native request identity disagrees with runtime")
        pool, request_pool = get_token_to_kv_pool(), get_req_to_token_pool()
        if (
            pool is not request.runner.token_to_kv_pool
            or request_pool is not request.runner.req_to_token_pool
        ):
            raise StaleCompletionError("forward context native pools changed")
        if type(pool) is not MHATokenToKVPool:
            raise ValueError("CacheSlide requires the plain MHATokenToKVPool")
        if (
            pool.kv_cache_layout != "nhd"
            or pool.use_hnd
            or pool.is_quantized_kv_cache
            or pool.store_dtype != pool.dtype
        ):
            raise ValueError("CacheSlide requires unquantized native NHD KV storage")
        key, value = (
            pool.get_key_buffer(layer_index),
            pool.get_value_buffer(layer_index),
        )
        req_index, generation = request.req_pool_index, request.req_generation
        if req_index <= 0 or batch.req_pool_indices.tolist() != [req_index]:
            raise ValueError("native request row does not match request context")
        length = int(batch.seq_lens[0])
        if length <= max(step.positions):
            raise ValueError("native sequence length does not cover scheduled tokens")
        canonical = request_pool.req_to_token[req_index, :length]
        scheduled = torch.tensor(step.positions, device=canonical.device)
        actual = batch.out_cache_loc
        if (
            actual.ndim != 1
            or actual.numel() != len(step.positions)
            or not torch.equal(actual.to(canonical.dtype), canonical[scheduled])
        ):
            raise ValueError("native output KV locations disagree with canonical row")
        key_identity = (key.data_ptr(), key.shape, key.stride(), key.dtype)
        value_identity = (value.data_ptr(), value.shape, value.stride(), value.dtype)

        def owner_check():
            if int(request_pool.req_generation[req_index]) != generation:
                raise StaleCompletionError("native request row generation changed")
            if req_index in request_pool.free_slots:
                raise StaleCompletionError("native request row was released")
            current_k = pool.get_key_buffer(layer_index)
            current_v = pool.get_value_buffer(layer_index)
            if (
                current_k.data_ptr(),
                current_k.shape,
                current_k.stride(),
                current_k.dtype,
            ) != key_identity or (
                current_v.data_ptr(),
                current_v.shape,
                current_v.stride(),
                current_v.dtype,
            ) != value_identity:
                raise StaleCompletionError("native KV backing changed")

        owner_check()
        if existing is not None:
            if (
                existing.request_id != request.request_id
                or existing.key_buffer.data_ptr() != key.data_ptr()
                or existing.value_buffer.data_ptr() != value.data_ptr()
            ):
                raise StaleCompletionError(
                    "decode arena no longer belongs to this request"
                )
            existing.update_slot_mapping(canonical)
            return existing
        return SGLangTokenKV(
            key,
            value,
            canonical,
            prompt_length,
            max_selected,
            page_size=pool.page_size,
            request_id=request.request_id,
            layer_index=layer_index,
            ownership_check=owner_check,
        )

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        if forward_batch is not integration.current_forward_batch():
            raise ValueError("model requires the currently bound native ForwardBatch")
        step = current_step()
        if step is None:
            # Explicit profiling/warmup scope: trained CoPE, no persistent cache.
            positions = torch.arange(len(positions), device=positions.device)
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        if (
            input_embeds is not None
            or pp_proxy_tensors is not None
            or self.layers_to_capture
        ):
            raise ValueError("unsupported CacheSlide embedding/pipeline/hidden capture")
        hidden = self.embed_tokens(input_ids)
        return self.cacheslide_runtime.run(
            _BoundModel(self, forward_batch), hidden, positions, step
        )


class CacheSlideLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # SGLang's native Llama omits this optional processor argument, while
        # the verified training/reference stack honors the checkpoint field.
        self.logits_processor.logit_scale = float(getattr(config, "logit_scale", 1.0))

    def _init_model(self, config, quant_config=None, prefix=""):
        verify_installed_sglang()
        if quant_config is not None:
            raise ValueError("CacheSlide requires an unquantized backbone")
        launch = integration.launch_config()
        bundle = AdapterBundle(launch.settings.artifact_path, launch.model_path)
        validate_native_model_config(config, bundle.config)
        native = LlamaModel(config, quant_config=quant_config, prefix=prefix)
        return CacheSlideModel(native, bundle, launch.settings)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if get_embedding or self.capture_aux_hidden_states:
            raise ValueError(
                "CacheSlide serves generation without auxiliary hidden outputs"
            )
        with integration.bind_model_forward(input_ids, positions, forward_batch):
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )

    def cacheslide_release_request(self, request_id):
        self.model.cacheslide_runtime.release(request_id)


EntryClass = [CacheSlideLlamaForCausalLM]
