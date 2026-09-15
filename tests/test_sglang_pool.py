import threading
from concurrent.futures import Future
from dataclasses import replace

import pytest
import torch

import cacheslide_sglang.pool as pool_module
from cacheslide_core.storage import CacheCapacityError, StaleCompletionError
from cacheslide_sglang.pool import SGLangTokenKV


def arena_fixture(capacity=2, owner=None):
    backing_k, backing_v = torch.full((20, 2, 6), -99.0), torch.full((20, 2, 6), -77.0)
    keys, values = backing_k[..., ::2], backing_v[..., ::2]
    slots = torch.tensor([8, 9, 2, 3, 12, 13, 6, 7], dtype=torch.int32)
    arena = SGLangTokenKV(
        keys, values, slots, 5, capacity, page_size=2, ownership_check=owner
    )
    k = torch.arange(30, dtype=keys.dtype).reshape(5, 2, 3)
    v = k + 100
    arena.write_prefill(k, v)
    return arena, k, v, slots, backing_k, backing_v


@pytest.mark.parametrize("ready", [True, False])
def test_native_separate_buffers_ready_pending_and_decode_holes(ready):
    arena, k, v, slots, backing_k, backing_v = arena_fixture()
    original_slots = slots.clone()
    if ready:
        arena.write_selected_ready([1, 3], k[[1, 3]] + 1, v[[1, 3]] + 1)
        assert arena.stats()["sidecar_allocated_slots"] == 0
    else:
        loading = Future()
        result = arena.promote_selected(
            [1, 3], k[[1, 3]] + 1, v[[1, 3]] + 1, load_future=loading
        )
        assert result.result()[1] == 20
        assert arena.stats()["sidecar_allocated_slots"] == 2
        loading.set_result(None)
    arena.update_selected([1, 3], k[[1, 3]] + 2, v[[1, 3]] + 3)
    actual_k, actual_v = arena.gather([3, 0, 1])
    expected_k, expected_v = k.clone(), v.clone()
    expected_k[[1, 3]] += 2
    expected_v[[1, 3]] += 3
    torch.testing.assert_close(actual_k, expected_k[[3, 0, 1]])
    torch.testing.assert_close(actual_v, expected_v[[3, 0, 1]])
    assert arena.decode(5, k[0] + 4, v[0] + 5) == (13 if ready else 3)
    decoded_k, decoded_v = arena.gather([5])
    torch.testing.assert_close(decoded_k[0], k[0] + 4)
    torch.testing.assert_close(decoded_v[0], v[0] + 5)
    torch.testing.assert_close(arena.key_buffer[13], k[0] + 4)
    torch.testing.assert_close(arena.value_buffer[13], v[0] + 5)
    torch.testing.assert_close(slots, original_slots)
    assert torch.all(backing_k[..., 1::2] == -99)
    assert torch.all(backing_v[..., 1::2] == -77)
    assert arena.stats()["native_pool_blocks_released"] == 0
    assert arena.stats()["native_pool_capacity_delta"] == 0


def test_mapping_growth_is_metadata_only_and_movement_is_rejected():
    arena, k, v, slots, *_ = arena_fixture(0)
    arena.update_slot_mapping(slots[:5])
    with pytest.raises(ValueError, match="caller-owned"):
        arena.decode(5, k[0], v[0])
    arena.update_slot_mapping(slots)
    arena.decode(5, k[0], v[0])
    moved = slots.clone()
    moved[1] = 11
    with pytest.raises(StaleCompletionError, match="moved"):
        arena.update_slot_mapping(moved)
    with pytest.raises(CacheCapacityError):
        another, ak, av, *_ = arena_fixture(0)
        another.promote_selected([1], ak[1:2], av[1:2])


def test_pins_preserve_old_mirrors_and_reject_current_writes():
    arena, k, v, *_ = arena_fixture(1)
    with arena.slot_map.pinned_mapping([1]):
        arena.promote_selected([1], k[1:2] + 1, v[1:2]).result()
        arena.update_selected([1], k[1:2] + 2, v[1:2])
        torch.testing.assert_close(arena.key_buffer[9], k[1])
        assert arena.stats()["canonical_mirror_writes"] == 0
        assert arena.decode(5, k[0], v[0]) == 13
    second, k, v, *_ = arena_fixture()
    with second.slot_map.pinned_mapping([1]):
        with pytest.raises(RuntimeError, match="pinned"):
            second.write_selected_ready([1], k[1:2], v[1:2])
        with pytest.raises(RuntimeError, match="pinned"):
            second.update_selected([1], k[1:2], v[1:2])


@pytest.mark.parametrize("operation", ["ready", "update", "decode"])
@pytest.mark.parametrize("outcome", ["success", "failed", "recycled"])
def test_async_writes_reject_pins_until_completion_and_owner_validation(
    monkeypatch, operation, outcome
):
    generation = [1]

    def owner():
        if generation[0] != 1:
            raise StaleCompletionError("row generation recycled")

    arena, k, v, *_ = arena_fixture(owner=owner)
    device_done, writer_done = Future(), Future()
    enqueued = threading.Event()

    def completion(_):
        enqueued.set()
        return device_done

    def write():
        try:
            if operation == "ready":
                arena.write_selected_ready([1], k[1:2] + 1, v[1:2])
            elif operation == "update":
                arena.update_selected([1], k[1:2] + 1, v[1:2])
            else:
                arena.decode(5, k[0] + 1, v[0])
        except BaseException as error:
            writer_done.set_exception(error)
        else:
            writer_done.set_result(None)

    with monkeypatch.context() as patch:
        patch.setattr(pool_module, "_completion_future", completion)
        thread = threading.Thread(target=write, daemon=True)
        thread.start()
        try:
            assert enqueued.wait(3)
            assert not writer_done.done()
            with pytest.raises(KeyError if operation == "decode" else RuntimeError):
                with arena.slot_map.pinned_mapping([5 if operation == "decode" else 1]):
                    pass
            if outcome == "recycled":
                generation[0] = 2
        finally:
            if outcome == "failed":
                device_done.set_exception(OSError("device write failed"))
            else:
                device_done.set_result(None)
            thread.join(3)
        assert not thread.is_alive()
        assert arena.slot_map.stats()["inflight_slots"] == 0
        if outcome == "success":
            writer_done.result()
        else:
            with pytest.raises(
                OSError if outcome == "failed" else StaleCompletionError
            ):
                writer_done.result()
            with pytest.raises(StaleCompletionError):
                arena.snapshot()


def test_retirement_drains_device_work_without_waiting_for_host_load(monkeypatch):
    arena, k, v, *_ = arena_fixture(1)
    device_done, host_load, retired = Future(), Future(), threading.Event()
    monkeypatch.setattr(pool_module, "_completion_future", lambda _: device_done)
    promotion = arena.promote_selected([1], k[1:2], v[1:2], load_future=host_load)
    thread = threading.Thread(
        target=lambda: (arena.invalidate(), retired.set()), daemon=True
    )
    thread.start()
    assert not retired.wait(0.05)
    device_done.set_result(None)
    thread.join(3)
    assert retired.is_set() and not host_load.done()
    with pytest.raises(StaleCompletionError):
        promotion.result()
    host_load.set_result(None)
    assert arena.slot_map.stats()["inflight_slots"] == 0


@pytest.mark.parametrize("slots", [[0, 1], [1, 1], [1, 20], [-1, 1]])
def test_invalid_native_slot_coordinates(slots):
    with pytest.raises(ValueError):
        SGLangTokenKV(torch.zeros(20, 1, 2), torch.zeros(20, 1, 2), slots, 2, 0)


def test_native_paged_allocator_padding_page_is_not_request_owned():
    with pytest.raises(ValueError, match="padding page"):
        SGLangTokenKV(
            torch.zeros(20, 1, 2), torch.zeros(20, 1, 2), [1, 2], 2, 0, page_size=2
        )


def test_generation_rejection_happens_before_mutating_native_buffers():
    stale = [False]

    def owner():
        if stale[0]:
            raise StaleCompletionError("released owner")

    arena, k, v, *_ = arena_fixture(owner=owner)
    before = arena.key_buffer.clone()
    stale[0] = True
    with pytest.raises(StaleCompletionError):
        arena.write_prefill(k + 1, v)
    torch.testing.assert_close(arena.key_buffer, before)


@pytest.mark.parametrize("load_ready", [False, True])
@pytest.mark.parametrize("promote_at_gate", [False, True])
def test_shared_runtime_matches_dense_with_native_split_pool_and_wca(
    tmp_path, monkeypatch, load_ready, promote_at_gate
):
    from cacheslide_core.runtime import CacheSlideRuntime
    from tests.test_runtime import build_runtime, decode, plan, prefill

    overrides = (
        {"correction_fraction": 0.17, "convergence_threshold": 2.0}
        if promote_at_gate
        else {}
    )
    model, dense = build_runtime(tmp_path, **overrides)
    native = CacheSlideRuntime(
        replace(dense.settings, cache_root=str(tmp_path / "split-native-cache")),
        dense.bundle,
        model_dtype=torch.float32,
    )
    config = dense.bundle.config
    slots = torch.tensor(
        [11, 3, 14, 1, 8, 6, 2, 13, 4, 12, 9, 5, 15, 7, 10, *range(16, 64)],
        dtype=torch.int32,
    )
    buffers = [
        (
            torch.zeros(64, config["num_key_value_heads"], config["head_dim"]),
            torch.zeros(64, config["num_key_value_heads"], config["head_dim"]),
        )
        for _ in model.layers
    ]

    def arena_factory(layer, prompt_length, max_selected, *, existing=None):
        length = prompt_length
        if existing is not None:
            length += native.active.metrics["decode_tokens"] + 1
            existing.update_slot_mapping(slots[:length])
            return existing
        return SGLangTokenKV(
            *buffers[layer],
            slots[:length],
            prompt_length,
            max_selected,
            request_id="paired",
            layer_index=layer,
        )

    native.arena_factory = arena_factory
    original_load = native.store.submit_load

    class DeferredLoad(Future):
        def __init__(self, actual):
            super().__init__()
            self.actual = actual

        def result(self, timeout=None):
            if not self.done():
                self.set_result(self.actual.result(timeout))
            return super().result(timeout)

    def controlled_load(key):
        actual = original_load(key)
        if not load_ready:
            return DeferredLoad(actual)
        ready = Future()
        ready.set_result(actual.result())
        return ready

    monkeypatch.setattr(native.store, "submit_load", controlled_load)

    def bind(runtime):
        for layer in model.layers:
            layer.self_attn.attention_handler = runtime.attention

    try:
        for request in (
            plan(operation="populate"),
            plan(operation="reuse"),
            plan((4, 10), operation="reuse"),
        ):
            bind(dense)
            expected = prefill(model, dense, request, request_id="paired")
            bind(native)
            actual = prefill(model, native, request, request_id="paired")
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert native.last_metrics["layer_rows"] == dense.last_metrics["layer_rows"]
            assert not native.last_metrics["fallback"]
            if request.operation == "reuse":
                assert native.last_metrics["cache_hit"]
        arenas = native.requests["paired"].arenas
        capacity = max(a.stats()["sidecar_allocated_slots"] for a in arenas.values())
        assert bool(capacity) is not load_ready
        if promote_at_gate:
            assert native.last_metrics["restored_rows"] > 0
        for offset in range(capacity + 2):
            token, position = (
                int(expected[-1].argmax()),
                len(request.token_ids) + offset,
            )
            bind(dense)
            expected = decode(
                model, dense, request, token, position, request_id="paired"
            )
            bind(native)
            actual = decode(
                model, native, request, token, position, request_id="paired"
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for layer, arena in arenas.items():
                k, v = arena.gather(range(position + 1))
                expected_k, expected_v = dense.requests["paired"].dense_kv[layer]
                torch.testing.assert_close(k, expected_k, atol=0, rtol=0)
                torch.testing.assert_close(v, expected_v, atol=0, rtol=0)
        assert all(
            a.stats()["physical_hole_reuse"] == a.stats()["sidecar_allocated_slots"]
            for a in arenas.values()
        )
    finally:
        dense.close()
        native.close()
