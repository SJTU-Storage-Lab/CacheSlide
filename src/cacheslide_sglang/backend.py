"""Metadata-only native backend for the explicit CacheSlide architecture.

Actual attention is the shared CoPE/WCA runtime, not RadixAttention. Reaching a
normal Radix kernel would mix incompatible positional/KV semantics and is fatal.
"""

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class CacheSlideAttentionBackend(AttentionBackend):
    extend_dummy_seqs_capped_by_req_pool = True

    def __init__(self, model_runner):
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_index_translator = getattr(model_runner, "kv_index_translator", None)
        self.forward_metadata = None

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        if in_capture:
            raise ValueError("CacheSlide requires eager execution without CUDA graphs")
        if forward_batch.batch_size not in (0, 1):
            raise ValueError("CacheSlide requires one request per forward")
        self.forward_metadata = None

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "CacheSlide requires its model attention wrapper; native RadixAttention "
            "must not execute against contextual KV"
        )

    forward_decode = forward
    forward_extend = forward
    forward_mixed = forward


def _create_backend(runner):
    return CacheSlideAttentionBackend(runner)


def register_backend():
    from sglang.srt.layers.attention.attention_registry import (
        ATTENTION_BACKENDS,
        register_attention_backend,
    )
    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )

    existing = ATTENTION_BACKENDS.get("cacheslide")
    if existing is not None and existing is not _create_backend:
        raise RuntimeError("another plugin already owns the cacheslide backend")
    register_attention_backend("cacheslide")(_create_backend)
    if "cacheslide" not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices(["cacheslide"])
