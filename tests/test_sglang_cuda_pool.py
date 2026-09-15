"""Opt-in real-CUDA token-arena tests, NOT a native SGLang engine validation.

Run only in an explicitly assigned, otherwise idle single GPU UUID namespace:
  CUDA_VISIBLE_DEVICES=GPU-<full-UUID> CACHESLIDE_RUN_CUDA_TESTS=1 \
    python -m pytest tests/test_sglang_cuda_pool.py -q

The default skip occurs before importing Torch or any CacheSlide module. These
tests allocate tiny tensors. A test-only post-write CUDA dependency delays the
completion boundary without replacing its real CUDA Event-backed Future. The
separate unresolved host-load Future models the slot guard; it is not SSD I/O.
"""

import os
import re
import threading
from concurrent.futures import Future

import pytest

if os.environ.get("CACHESLIDE_RUN_CUDA_TESTS") != "1":
    pytest.skip("real CUDA pool tests require explicit opt-in", allow_module_level=True)

if not re.fullmatch(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
    os.environ.get("CUDA_VISIBLE_DEVICES", ""),
):
    raise pytest.UsageError("CUDA tests require exactly one full GPU UUID visibility")

import torch  # noqa: E402

import cacheslide_sglang.pool as pool_module  # noqa: E402
from cacheslide_core.storage import StaleCompletionError  # noqa: E402
from cacheslide_sglang.pool import SGLangTokenKV  # noqa: E402


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        pytest.fail("opted-in CUDA tests require exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        pytest.fail("opted-in CUDA tests require native bfloat16 support")
    with torch.cuda.device(0):
        yield torch.device("cuda:0")
        torch.cuda.synchronize(0)


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
def arena_case(device, request):
    dtype = request.param
    backing_k = torch.full((20, 2, 6), -99, dtype=dtype, device=device)
    backing_v = torch.full((20, 2, 6), -77, dtype=dtype, device=device)
    slots = (8, 9, 2, 3, 12, 13, 6, 7)
    arena = SGLangTokenKV(
        backing_k[..., ::2],
        backing_v[..., ::2],
        slots,
        5,
        2,
        page_size=2,
        request_id="cuda-arena-only",
    )
    key = torch.arange(30, dtype=dtype, device=device).reshape(5, 2, 3)
    value = key + 40
    arena.write_prefill(key, value)
    # Warm relevant indexed copies before the explicitly gated race below.
    arena.gather(range(5))
    yield arena, key, value, slots, backing_k, backing_v
    arena.invalidate()
    torch.cuda.synchronize(device)


def assert_kv(arena, key, value):
    actual_k, actual_v = arena.gather(range(len(key)))
    torch.testing.assert_close(actual_k, key, atol=0, rtol=0)
    torch.testing.assert_close(actual_v, value, atol=0, rtol=0)


def completion_gate(monkeypatch):
    """Delay once AFTER index/H2D copies, then record the production event.

    A pre-write gate can be consumed by implicit synchronization constructing
    index tensors. This instrumentation instead inserts a real device dependency
    at the completion boundary, delegating to the untouched completion function.
    """
    original = pool_module._completion_future
    holder = {}

    def after_writes(device):
        if not holder:
            blocker = torch.cuda.Stream(device=device)
            event = torch.cuda.Event()
            with torch.cuda.stream(blocker):
                torch.cuda._sleep(1_000_000_000)
                event.record(blocker)
            torch.cuda.current_stream(device).wait_event(event)
            holder.update(blocker=blocker, event=event)
        return original(device)

    monkeypatch.setattr(pool_module, "_completion_future", after_writes)
    return holder


def test_cuda_ready_writes_pin_release_and_nondefault_stream_equivalence(
    arena_case, device
):
    arena, key, value, slots, backing_k, backing_v = arena_case
    expected_k, expected_v = key.clone(), value.clone()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with arena.slot_map.pinned_mapping([1]):
        with pytest.raises(RuntimeError, match="pinned"):
            arena.write_selected_ready([1], key[1:2] + 10, value[1:2] + 20)
    assert arena.stats()["pinned_slots"] == 0
    with torch.cuda.stream(stream):
        arena.write_selected_ready([1, 3], key[[1, 3]] + 10, value[[1, 3]] + 20)
        arena.update_selected([1, 3], key[[1, 3]] + 12, value[[1, 3]] + 22)
        expected_k[[1, 3]] += 12
        expected_v[[1, 3]] += 22
        assert_kv(arena, expected_k, expected_v)
        assert arena.decode(5, key[0] + 60, value[0] + 60) == slots[5]
        expected_k = torch.cat((expected_k, key[0:1] + 60))
        expected_v = torch.cat((expected_v, value[0:1] + 60))
        assert_kv(arena, expected_k, expected_v)
    assert arena.stats()["sidecar_allocated_slots"] == 0
    assert arena.stats()["selected_ready_inplace_writes"] == 2
    assert arena.stats()["inflight_slots"] == 0
    assert arena.stats()["pinned_slots"] == 0
    torch.testing.assert_close(
        backing_k[..., 1::2], torch.full_like(backing_k[..., 1::2], -99)
    )
    torch.testing.assert_close(
        backing_v[..., 1::2], torch.full_like(backing_v[..., 1::2], -77)
    )


def test_cuda_pending_promotion_then_only_unpinned_loaded_holes_are_reclaimed(
    arena_case, device, monkeypatch
):
    arena, key, value, slots, *_ = arena_case
    loading = Future()
    expected_k, expected_v = key.clone(), value.clone()
    selected_k, selected_v = key[[1, 3]] + 10, value[[1, 3]] + 20
    stream = torch.cuda.Stream(device=device)
    # Ensure source tensors made on the current stream precede the worker stream.
    stream.wait_stream(torch.cuda.current_stream(device))
    gate_state = completion_gate(monkeypatch)
    with arena.slot_map.pinned_mapping([1]) as pinned:
        assert pinned[1] == slots[1]
        with torch.cuda.stream(stream):
            promotion = arena.promote_selected(
                [1, 3], selected_k, selected_v, load_future=loading
            )
        gate = gate_state["event"]
        assert not gate.query(), "CUDA gate expired before the race was observed"
        assert not promotion.done()
        assert arena.slot_map.snapshot()[1] == slots[1]
        mapping = promotion.result(timeout=10)
        assert gate.query() and mapping[1] >= arena.native_total_slots
        assert mapping[3] >= arena.native_total_slots
        assert not loading.done()
        expected_k[[1, 3]], expected_v[[1, 3]] = selected_k, selected_v
        assert_kv(arena, expected_k, expected_v)
        # Pending source I/O guards both original slots, even after publication.
        assert arena.decode(5, key[0] + 60, value[0] + 60) == slots[5]
        loading.set_result(None)
        # Slot 9 remains pinned; only vacated slot 3 can now be reused.
        assert arena.decode(6, key[0] + 80, value[0] + 80) == slots[3]
        torch.testing.assert_close(arena.key_buffer[slots[1]], key[1], atol=0, rtol=0)
    assert arena.stats()["pinned_slots"] == 0
    # Releasing the old mapping pin finally makes its original slot reclaimable.
    assert arena.decode(7, key[0] + 100, value[0] + 100) == slots[1]
    expected_k = torch.cat([expected_k, *(key[0:1] + n for n in (60, 80, 100))])
    expected_v = torch.cat([expected_v, *(value[0:1] + n for n in (60, 80, 100))])
    assert_kv(arena, expected_k, expected_v)
    for token in (5, 6, 7):
        torch.testing.assert_close(
            arena.key_buffer[slots[token]], expected_k[token], atol=0, rtol=0
        )
        torch.testing.assert_close(
            arena.value_buffer[slots[token]], expected_v[token], atol=0, rtol=0
        )
    assert arena.stats()["physical_hole_reuse"] == 2
    assert arena.stats()["native_pool_capacity_delta"] == 0
    assert arena.stats()["native_pool_blocks_released"] == 0
    assert arena.stats()["inflight_slots"] == 0
    gate_state["blocker"].synchronize()


def test_cuda_retirement_drains_real_event_and_rejects_late_publication(
    arena_case, device, monkeypatch
):
    arena, key, value, *_ = arena_case
    loading, retired, finished = Future(), threading.Event(), threading.Event()
    errors = []
    original_retire = arena.slot_map.invalidate

    def retire_map():
        original_retire()
        retired.set()

    monkeypatch.setattr(arena.slot_map, "invalidate", retire_map)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    gate_state = completion_gate(monkeypatch)
    with torch.cuda.stream(stream):
        promotion = arena.promote_selected(
            [1], key[1:2], value[1:2], load_future=loading
        )
    gate = gate_state["event"]

    def close():
        try:
            arena.invalidate()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    try:
        assert retired.wait(2)
        assert not gate.query(), "CUDA gate expired before retirement was observed"
        assert not finished.is_set()
        with pytest.raises(StaleCompletionError):
            arena.slot_map.snapshot()
        assert finished.wait(10)
        assert gate.query() and not errors
        with pytest.raises(StaleCompletionError):
            promotion.result(timeout=10)
        assert not loading.done()  # Native retirement must not wait on host I/O.
    finally:
        loading.set_result(None)
        thread.join(10)
        gate_state["blocker"].synchronize()
    assert not thread.is_alive()
    assert arena.stats()["inflight_slots"] == 0
