"""CPU execution of the real runtime and packed native KV adapter, without vLLM."""

from concurrent.futures import Future
from dataclasses import replace

import pytest
import torch

from cacheslide_core.runtime import CacheSlideRuntime
from cacheslide_vllm.paged import NativePagedKV
from tests.test_runtime import build_runtime, decode, plan, prefill


def bind(model, runtime):
    for layer in model.layers:
        layer.self_attn.attention_handler = runtime.attention


class CPUBlockPool:
    """Caller-owned noncontiguous native tensors and growing, permuted block rows."""

    def __init__(self, runtime):
        self.runtime = runtime
        config = runtime.bundle.config
        self.block_size = 2
        self.block_order = (11, 3, 14, 1, 8, 6, 0, 13, 4, 12, 9, 2, 15, 7, 5, 10)
        self.backing = [
            torch.full(
                (16, config["num_key_value_heads"], 2, 4 * config["head_dim"]), -999.0
            )
            for _ in range(config["num_hidden_layers"])
        ]
        self.created = []
        self.block_table_updates = 0

    def __call__(self, layer, prompt_length, max_selected, *, existing=None):
        tokens = prompt_length
        if existing is not None:
            tokens += self.runtime.active.metrics["decode_tokens"] + 1
        count = (tokens + self.block_size - 1) // self.block_size
        assert count <= len(self.block_order)
        row = self.block_order[:count] + (-1,) * (len(self.block_order) - count)
        cache = self.backing[layer][..., ::2]
        assert not cache.is_contiguous()
        if existing is not None:
            assert existing.cache.data_ptr() == cache.data_ptr()
            existing.update_block_table(row)
            self.block_table_updates += 1
            return existing
        arena = NativePagedKV(
            cache,
            row,
            prompt_length,
            max_selected,
            request_id="paired",
            layer_index=layer,
        )
        self.created.append(arena)
        return arena


@pytest.mark.parametrize("promote_at_gate", [False, True])
@pytest.mark.parametrize("load_ready", [False, True])
def test_dense_and_native_paged_runtime_match_through_shift_reuse_and_decode(
    tmp_path,
    promote_at_gate,
    load_ready,
    monkeypatch,
):
    """Actual native-hole consumption must preserve every logical KV and logit."""
    overrides = (
        {"correction_fraction": 0.17, "convergence_threshold": 2.0}
        if promote_at_gate
        else {}
    )
    model, dense = build_runtime(tmp_path, **overrides)
    native = CacheSlideRuntime(
        replace(dense.settings, cache_root=str(tmp_path / "native-cache")),
        dense.bundle,
        model_dtype=torch.float32,
    )
    pool = CPUBlockPool(native)
    native.arena_factory = pool
    original_load = native.store.submit_load

    class DeferredLoad(Future):
        # Deterministically keep the host completion unobserved until result().
        def __init__(self, loaded):
            super().__init__()
            self.loaded = loaded

        def result(self, timeout=None):
            if not self.done():
                self.set_result(self.loaded.result(timeout=timeout))
            return super().result(timeout=timeout)

    def controlled_load(key):
        loaded = original_load(key)
        if not load_ready:
            return DeferredLoad(loaded)
        ready = Future()
        ready.set_result(loaded.result())
        return ready

    monkeypatch.setattr(native.store, "submit_load", controlled_load)
    try:
        assert dense.identity == native.identity
        requests = (
            plan(operation="populate"),
            plan(operation="reuse"),
            plan((4, 10), operation="reuse"),
        )
        for request in requests:
            bind(model, dense)
            expected = prefill(model, dense, request, request_id="paired")
            bind(model, native)
            actual = prefill(model, native, request, request_id="paired")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert native.last_metrics["layer_rows"] == dense.last_metrics["layer_rows"]
            assert not native.last_metrics["fallback"]
            if request.operation == "reuse":
                assert native.last_metrics["cache_hit"]
                assert (
                    native.last_metrics["computed_token_layers"]
                    < native.last_metrics["dense_token_layers"]
                )
            arenas = native.requests["paired"].arenas
            assert len(arenas) == len(model.layers) == 6
            for layer, arena in arenas.items():
                key, value = arena.gather(range(len(request.token_ids)))
                expected_key, expected_value = dense.requests["paired"].dense_kv[layer]
                torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
                torch.testing.assert_close(value, expected_value, rtol=0, atol=0)

        if promote_at_gate:
            assert native.last_metrics["restored_rows"] > 0
            assert (
                native.last_metrics["restored_rows"]
                == dense.last_metrics["restored_rows"]
            )
        capacities = {
            layer: arena.stats()["sidecar_allocated_slots"]
            for layer, arena in arenas.items()
        }
        assert (sum(capacities.values()) == 0) is load_ready
        decode_steps = max(capacities.values()) + 3
        for index in range(decode_steps):
            token = int(expected[-1].argmax())
            position = len(request.token_ids) + index
            bind(model, dense)
            expected = decode(
                model, dense, request, token, position, request_id="paired"
            )
            bind(model, native)
            actual = decode(
                model, native, request, token, position, request_id="paired"
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for layer, arena in arenas.items():
                key, value = arena.gather(range(position + 1))
                expected_key, expected_value = dense.requests["paired"].dense_kv[layer]
                torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
                torch.testing.assert_close(value, expected_value, rtol=0, atol=0)

        for layer, arena in arenas.items():
            stats = arena.stats()
            # Every safe old selected native slot is reused, then native allocation
            # supplied by the block table handles the remaining decode tokens.
            assert stats["physical_hole_reuse"] == capacities[layer]
            assert stats["decode_native_bind"] == decode_steps - capacities[layer]
            assert stats["decode_native_bind"] >= 3
            assert stats["native_pool_capacity_slots"] == 32
            assert stats["native_pool_capacity_delta"] == 0
            assert stats["native_pool_blocks_released"] == 0
            assert stats["sidecar_allocated_slots"] == capacities[layer]
            assert torch.all(pool.backing[layer][..., 1::2] == -999)
        assert pool.block_table_updates == 6 * decode_steps
        assert len(pool.created) == 6 * len(requests)
    finally:
        dense.close()
        native.close()
