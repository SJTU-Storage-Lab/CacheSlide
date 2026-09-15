import threading
from concurrent.futures import Future

import pytest
import torch

import cacheslide_vllm.paged as paged
from cacheslide_vllm.paged import NativePagedKV
from cacheslide_vllm.position import CoPE, cope_attention
from cacheslide_vllm.storage import CacheCapacityError, StaleCompletionError


def fixture_arena(max_selected=2):
    backing = torch.full((8, 2, 2, 12), -999.0)
    cache = backing[..., ::2]
    assert not cache.is_contiguous()
    arena = NativePagedKV(
        cache, [4, 1, 6, 3], 5, max_selected, request_id="request-a", layer_index=2
    )
    key = torch.arange(30, dtype=cache.dtype).reshape(5, 2, 3) / 30
    value = key + 10
    arena.write_prefill(key, value)
    return arena, key, value, backing


def native_row(arena, token):
    block, offset = divmod(arena.canonical_slot(token), arena.block_size)
    return (
        arena.cache[block, :, offset, : arena.head_dim],
        arena.cache[block, :, offset, arena.head_dim :],
    )


def test_native_noncontiguous_prefill_staged_wca_decode_and_gqa():
    arena, key, value, backing = fixture_arena()
    assert arena.snapshot() == {0: 8, 1: 9, 2: 2, 3: 3, 4: 12}
    loading = Future()
    selected = torch.tensor([3, 1])
    raw_key, raw_value = key[selected] + 1, value[selected] + 2
    mapping = arena.promote_selected(
        selected, raw_key, raw_value, load_future=loading
    ).result()
    assert mapping[3] == 16 and mapping[1] == 17
    assert arena.stats()["vacated_slots"] == 2
    # Selected writes use independent extra storage while source DMA is pending.
    for row in selected:
        original_k, original_v = native_row(arena, int(row))
        torch.testing.assert_close(original_k, key[row])
        torch.testing.assert_close(original_v, value[row])
    with pytest.raises(RuntimeError, match="load must complete"):
        arena.write_prefill(key, value)
    with pytest.raises(RuntimeError, match="load must complete"):
        arena.update_selected(selected, raw_key, raw_value)
    loading.set_result(None)
    arena.write_prefill(key, value)
    # Mandatory/dynamic rows already map to native slots; only fixed rows stage.
    updates = [1, 3, 4]
    final_key, final_value = key[updates] + 0.5, value[updates] + 0.75
    arena.update_selected(updates, final_key, final_value)
    key[updates], value[updates] = final_key, final_value
    for row in updates:
        original_k, original_v = native_row(arena, row)
        torch.testing.assert_close(original_k, key[row])
        torch.testing.assert_close(original_v, value[row])
    order = [4, 1, 0, 3, 2]
    gathered_k, gathered_v = arena.gather(order)
    torch.testing.assert_close(gathered_k, key[order])
    torch.testing.assert_close(gathered_v, value[order])

    assert arena.decode(5, key[0] + 2, value[0] + 2, 13) == 3
    assert arena.decode(6, key[0] + 3, value[0] + 3, 6) == 9
    assert arena.decode(7, key[0] + 4, value[0] + 4, 7) == 7
    expected_k = torch.cat((key, key[:1] + 2, key[:1] + 3, key[:1] + 4))
    expected_v = torch.cat((value, value[:1] + 2, value[:1] + 3, value[:1] + 4))
    gathered_k, gathered_v = arena.gather(range(8))
    torch.testing.assert_close(gathered_k, expected_k)
    torch.testing.assert_close(gathered_v, expected_v)
    for token in (5, 6, 7):
        canonical_k, canonical_v = native_row(arena, token)
        torch.testing.assert_close(canonical_k, expected_k[token])
        torch.testing.assert_close(canonical_v, expected_v[token])
    # Four query heads attend through two KV heads without flattening head strides.
    query = torch.linspace(-1, 1, 12).reshape(1, 4, 3)
    cope = CoPE(3, 16)
    actual = cope_attention(query, gathered_k, gathered_v, cope, torch.tensor([7]))
    expected = cope_attention(query, expected_k, expected_v, cope, torch.tensor([7]))
    torch.testing.assert_close(actual, expected)
    assert torch.all(arena.cache[0] == -999)  # block 0 was never caller-owned
    assert torch.all(backing[..., 1::2] == -999)  # noncontiguous backing untouched
    assert arena.stats()["physical_hole_reuse"] == 2
    assert arena.stats()["decode_native_bind"] == 1
    assert arena.stats()["native_pool_capacity_slots"] == 16
    assert arena.stats()["native_pool_capacity_delta"] == 0
    assert arena.stats()["native_pool_blocks_released"] == 0
    assert arena.stats()["sidecar_capacity_slots"] == 2
    with pytest.raises(RuntimeError, match="after decode"):
        arena.write_prefill(key, value)
    with pytest.raises(RuntimeError, match="precede decode"):
        arena.update_selected(updates, final_key, final_value)


def test_source_slots_not_reused_until_load_completes():
    arena, key, value, _ = fixture_arena()
    loading = Future()
    arena.stage_selected(
        [1, 3], key[[1, 3]] + 1, value[[1, 3]], load_future=loading
    ).result()
    assert arena.decode(5, key[0], value[0]) == 13
    assert arena.stats()["physical_hole_reuse"] == 0
    loading.set_result(None)
    assert arena.decode(6, key[0], value[0]) == 3
    assert arena.decode(7, key[0], value[0]) == 9


def test_pinned_original_cannot_be_reused_for_decode():
    arena, key, value, _ = fixture_arena(max_selected=1)
    with arena.slot_map.pin_slots([9]):
        arena.promote_selected([1], key[1:2] + 1, value[1:2]).result()
        assert arena.decode(5, key[0], value[0]) == 13
    assert arena.decode(6, key[0], value[0]) == 9


@pytest.mark.parametrize("pin_mapping", [False, True])
def test_selected_update_preserves_pinned_original_and_skips_only_its_mirror(
    pin_mapping,
):
    arena, key, value, _ = fixture_arena()
    pin = (
        arena.slot_map.pinned_mapping([1])
        if pin_mapping
        else arena.slot_map.pin_slots([9])
    )
    with pin:
        arena.promote_selected([1, 3], key[[1, 3]] + 1, value[[1, 3]] + 1).result()
        arena.update_selected([1, 3], key[[1, 3]] + 2, value[[1, 3]] + 2)
        original_key, original_value = native_row(arena, 1)
        torch.testing.assert_close(original_key, key[1])
        torch.testing.assert_close(original_value, value[1])
        unpinned_key, unpinned_value = native_row(arena, 3)
        torch.testing.assert_close(unpinned_key, key[3] + 2)
        torch.testing.assert_close(unpinned_value, value[3] + 2)
        gathered_key, gathered_value = arena.gather([1, 3])
        torch.testing.assert_close(gathered_key, key[[1, 3]] + 2)
        torch.testing.assert_close(gathered_value, value[[1, 3]] + 2)
        assert arena.stats()["canonical_mirror_writes"] == 1
    arena.update_selected([1], key[1:2] + 3, value[1:2] + 3)
    torch.testing.assert_close(native_row(arena, 1)[0], key[1] + 3)
    assert arena.stats()["canonical_mirror_writes"] == 2
    assert arena.decode(5, key[0], value[0]) == 3
    assert arena.decode(6, key[0], value[0]) == 9


@pytest.mark.parametrize("staged", [False, True])
def test_selected_update_rejects_pinned_destination_before_any_writes(staged):
    arena, key, value, _ = fixture_arena(max_selected=1)
    if staged:
        arena.promote_selected([1], key[1:2], value[1:2]).result()
    before = arena.snapshot()
    with arena.slot_map.pinned_mapping([1]):
        with pytest.raises(RuntimeError, match="pinned|in-flight"):
            arena.update_selected([4, 1], key[[4, 1]] + 2, value[[4, 1]] + 2)
        actual_key, actual_value = arena.gather([4, 1])
        torch.testing.assert_close(actual_key, key[[4, 1]])
        torch.testing.assert_close(actual_value, value[[4, 1]])
        assert arena.snapshot() == before
    arena.update_selected([4, 1], key[[4, 1]] + 2, value[[4, 1]] + 2)
    torch.testing.assert_close(arena.gather([4, 1])[0], key[[4, 1]] + 2)


@pytest.mark.parametrize("retire", [False, True])
def test_selected_write_guards_precede_copy_and_outlive_device_completion(
    monkeypatch, retire
):
    arena, key, value, _ = fixture_arena(max_selected=1)
    arena.promote_selected([1], key[1:2], value[1:2]).result()
    copy_started, release_copy, completion_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    device_done, update_done = Future(), Future()
    original_write = arena._write_slots
    original_completion = paged._completion_future

    class FakeEvent:
        def record(self, stream):
            assert stream == "test-stream"

        def synchronize(self):
            completion_started.set()
            device_done.result(timeout=3)

    def controlled_write(*args):
        copy_started.set()
        assert release_copy.wait(3)
        original_write(*args)

    def completion(_):
        # Exercise the actual event-to-Future adapter without a GPU launch.
        return original_completion(torch.device("cuda"))

    def update():
        try:
            arena.update_selected([1, 4], key[[1, 4]] + 2, value[[1, 4]] + 2)
        except BaseException as error:
            update_done.set_exception(error)
        else:
            update_done.set_result(None)

    with monkeypatch.context() as patch:
        patch.setattr(arena, "_write_slots", controlled_write)
        patch.setattr(paged, "_completion_future", completion)
        patch.setattr(torch.cuda, "Event", FakeEvent)
        patch.setattr(torch.cuda, "current_stream", lambda device: "test-stream")
        writer = threading.Thread(target=update, daemon=True)
        writer.start()
        try:
            assert copy_started.wait(3)
            # Reservations precede even the first copy, closing check/write races.
            for slot in (16, 12):
                with pytest.raises(RuntimeError, match="being written"):
                    with arena.slot_map.pin_slots([slot]):
                        pass
            with pytest.raises(RuntimeError, match="being written"):
                with arena.slot_map.pinned_mapping([1, 4]):
                    pass
            assert arena.slot_map.stats()["inflight_slots"] == 3
            release_copy.set()
            assert completion_started.wait(3)
            assert not update_done.done()
            # Device copies were enqueued, but their completion is still pending.
            with pytest.raises(RuntimeError, match="being written"):
                with arena.slot_map.pinned_mapping([1]):
                    pass
            # No map mutex is held across the completion Future wait.
            assert arena.slot_map.snapshot()[1] == 16
            if retire:
                arena.slot_map.invalidate()
        finally:
            release_copy.set()
            device_done.set_result(None)
            writer.join(timeout=3)
        assert not writer.is_alive()
        if retire:
            with pytest.raises(StaleCompletionError, match="retired"):
                update_done.result()
        else:
            update_done.result()
            with arena.slot_map.pinned_mapping([1, 4]):
                assert arena.slot_map.stats()["pinned_slots"] == 2
        assert arena.slot_map.stats()["inflight_slots"] == 0
    if not retire:
        torch.testing.assert_close(arena.gather([1, 4])[0], key[[1, 4]] + 2)


def test_selected_publication_waits_for_actual_write_completion(monkeypatch):
    arena, key, value, _ = fixture_arena(max_selected=1)
    written = Future()
    initial = arena.snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(paged, "_completion_future", lambda _: written)
        promotion = arena.promote_selected([1], key[1:2] + 1, value[1:2])
        assert not promotion.done()
        assert arena.snapshot() == initial
        with pytest.raises(RuntimeError, match="writes must complete"):
            arena.decode(5, key[0], value[0])
        written.set_result(None)
        assert promotion.result()[1] == 16
    gathered, _ = arena.gather([1])
    torch.testing.assert_close(gathered, key[1:2] + 1)


def test_failed_selected_device_write_preserves_original_mapping(monkeypatch):
    arena, key, value, _ = fixture_arena(max_selected=1)
    written = Future()
    initial = arena.snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(paged, "_completion_future", lambda _: written)
        promotion = arena.promote_selected([1], key[1:2] + 1, value[1:2])
        written.set_exception(OSError("device copy failed"))
        with pytest.raises(OSError, match="device copy"):
            promotion.result()
    assert arena.snapshot() == initial
    original, _ = arena.gather([1])
    torch.testing.assert_close(original, key[1:2])


def test_failed_source_load_blocks_decode_and_baseline():
    arena, key, value, _ = fixture_arena(max_selected=1)
    loading = Future()
    arena.promote_selected([1], key[1:2], value[1:2], load_future=loading).result()
    loading.set_exception(OSError("source load failed"))
    with pytest.raises(OSError, match="source load"):
        arena.decode(5, key[0], value[0])
    with pytest.raises(OSError, match="source load"):
        arena.write_prefill(key, value)
    assert 5 not in arena.snapshot()


def test_bounded_sidecar_never_claims_unowned_native_capacity():
    arena, key, value, _ = fixture_arena(max_selected=1)
    initial = arena.snapshot()
    with pytest.raises(CacheCapacityError):
        arena.promote_selected([1, 3], key[[1, 3]], value[[1, 3]])
    assert arena.snapshot() == initial
    assert torch.all(arena.cache[0] == -999)
    arena.promote_selected([1], key[1:2], value[1:2]).result()
    assert arena.snapshot()[1] == arena.native_total_slots
    with pytest.raises(ValueError, match="not owned"):
        arena.decode(5, key[0], value[0], canonical_slot=0)
    assert 5 not in arena.snapshot()


def test_zero_sidecar_can_bind_new_owned_native_slots():
    arena, key, value, _ = fixture_arena(max_selected=0)
    with pytest.raises(CacheCapacityError):
        arena.promote_selected([1], key[1:2], value[1:2])
    assert arena.decode(5, key[0], value[0]) == 13
    assert arena.stats()["sidecar_capacity_slots"] == 0


def test_caller_block_table_growth_and_existing_block_ownership():
    arena, key, value, _ = fixture_arena(max_selected=0)
    arena.update_block_table([4, 1, 6, -1])
    arena.decode(5, key[0], value[0])
    with pytest.raises(ValueError, match="no caller-owned"):
        arena.decode(6, key[0], value[0])
    with pytest.raises(ValueError, match="moved an existing"):
        arena.update_block_table([4, 2, 6, 3])
    arena.update_block_table(torch.tensor([4, 1, 6, 3], dtype=torch.int32))
    assert arena.decode(6, key[0], value[0]) == 6
    assert arena.decode(7, key[0], value[0]) == 7
    arena.update_block_table([4, 1, 6, 3, 0])
    assert arena.decode(8, key[0], value[0]) == 0
    assert arena.stats()["native_pool_blocks_released"] == 0
    assert arena.stats()["native_pool_capacity_slots"] == 16


def test_shared_prefix_slots_are_never_overwritten_or_reclaimed():
    original, key, value, _ = fixture_arena(max_selected=1)
    arena = NativePagedKV(original.cache, [4, 1, 6, 3], 5, 1, shared_slots={9})
    with pytest.raises(ValueError, match="shared"):
        arena.write_prefill(key, value)
    arena.promote_selected([1], key[1:2] + 1, value[1:2]).result()
    arena.update_selected([1], key[1:2] + 2, value[1:2] + 2)
    canonical_key, canonical_value = native_row(arena, 1)
    torch.testing.assert_close(canonical_key, key[1])
    torch.testing.assert_close(canonical_value, value[1])
    gathered_key, _ = arena.gather([1])
    torch.testing.assert_close(gathered_key, key[1:2] + 2)
    assert arena.decode(5, key[0], value[0]) == 13
    assert arena.stats()["physical_hole_reuse"] == 0


def test_request_retirement_cannot_publish_into_another_arena(monkeypatch):
    arena, key, value, _ = fixture_arena(max_selected=1)
    another = NativePagedKV(arena.cache, [2, 5], 2, 1, request_id="request-b")
    another.write_prefill(key[:2] + 5, value[:2] + 5)
    written = Future()
    with monkeypatch.context() as patch:
        patch.setattr(paged, "_completion_future", lambda _: written)
        promotion = arena.promote_selected([1], key[1:2] + 1, value[1:2])
        arena.invalidate()
        written.set_result(None)
        with pytest.raises(StaleCompletionError):
            promotion.result()
    with pytest.raises(StaleCompletionError):
        arena.gather([1])
    assert another.snapshot() == {0: 4, 1: 5}
    actual_key, actual_value = another.gather([0, 1])
    torch.testing.assert_close(actual_key, key[:2] + 5)
    torch.testing.assert_close(actual_value, value[:2] + 5)


@pytest.mark.parametrize("failure", [False, True])
def test_cuda_future_waits_for_event_without_launching_gpu(monkeypatch, failure):
    started, released = threading.Event(), threading.Event()
    recorded = []

    class FakeEvent:
        def record(self, stream):
            recorded.append(stream)

        def synchronize(self):
            started.set()
            assert released.wait(2)
            if failure:
                raise OSError("injected CUDA event failure")

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "test-stream")
    completed = paged._completion_future(torch.device("cuda"))
    assert started.wait(1)
    assert recorded == ["test-stream"]
    assert not completed.done()
    assert not completed.cancel()
    released.set()
    if failure:
        with pytest.raises(OSError, match="event failure"):
            completed.result(timeout=2)
    else:
        assert completed.result(timeout=2) is None


def test_decode_copy_failure_retires_partial_mapping(monkeypatch):
    arena, key, value, _ = fixture_arena(max_selected=0)
    failed = Future()
    failed.set_exception(OSError("copy failed"))
    monkeypatch.setattr(paged, "_completion_future", lambda _: failed)
    with pytest.raises(OSError, match="copy failed"):
        arena.decode(5, key[0], value[0])
    with pytest.raises(StaleCompletionError):
        arena.snapshot()


@pytest.mark.parametrize("operation", ["baseline", "fused"])
def test_nontransactional_copy_failure_retires_arena(monkeypatch, operation):
    arena, key, value, _ = fixture_arena(max_selected=1)
    arena.promote_selected([1], key[1:2], value[1:2]).result()
    failed = Future()
    failed.set_exception(OSError("copy failed"))
    monkeypatch.setattr(paged, "_completion_future", lambda _: failed)
    with pytest.raises(OSError, match="copy failed"):
        if operation == "baseline":
            arena.write_prefill(key, value)
        else:
            arena.update_selected([1], key[1:2], value[1:2])
    with pytest.raises(StaleCompletionError):
        arena.snapshot()
    assert arena.slot_map.stats()["inflight_slots"] == 0


def test_layout_and_record_validation():
    cache = torch.zeros(4, 2, 2, 6)
    for row in ([4], [-2], [True]):
        with pytest.raises(ValueError, match="block_table"):
            NativePagedKV(cache, row, 1, 1)
    with pytest.raises(ValueError, match="distinct"):
        NativePagedKV(cache, [1, 1], 4, 1)
    with pytest.raises(ValueError, match="floating"):
        NativePagedKV(torch.zeros(4, 2, 2, 5), [1], 1, 1)
    arena = NativePagedKV(cache, [1, 2], 2, 1)
    good = torch.zeros(2, 2, 3)
    with pytest.raises(ValueError, match="dtype"):
        arena.write_prefill(good.double(), good)
    with pytest.raises(ValueError, match="distinct"):
        arena.write([0, 0], good, good)
    empty_key, empty_value = arena.gather([])
    assert empty_key.shape == empty_value.shape == (0, 2, 3)
