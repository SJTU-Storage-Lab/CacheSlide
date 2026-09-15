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


@pytest.mark.parametrize("max_selected", [0, 2])
def test_load_first_selected_writes_stay_native_without_sidecar_or_holes(max_selected):
    arena, key, value, _ = fixture_arena(max_selected=max_selected)
    original = arena.snapshot()
    assert arena.stats()["sidecar_allocated_slots"] == 0
    assert (
        arena.write_selected_ready([1, 3], key[[1, 3]] + 1, value[[1, 3]] + 2)
        == original
    )
    key[[1, 3]] += 1
    value[[1, 3]] += 2
    gathered_key, gathered_value = arena.gather(range(5))
    torch.testing.assert_close(gathered_key, key)
    torch.testing.assert_close(gathered_value, value)
    assert arena.stats()["selected_ready_inplace_writes"] == 2
    assert arena.stats()["sidecar_allocated_slots"] == 0
    assert arena.stats()["vacated_slots"] == 0
    assert arena.stats()["canonical_mirror_writes"] == 0
    assert arena.decode(5, key[0], value[0]) == 13
    assert arena.stats()["physical_hole_reuse"] == 0
    assert arena.stats()["native_pool_blocks_released"] == 0


def test_ready_selected_requires_all_actual_baseline_writes_not_host_future():
    original, key, value, _ = fixture_arena()
    arena = NativePagedKV(original.cache, [4, 1, 6, 3], 5, 2)
    host_ready = Future()
    host_ready.set_result(b"host bytes loaded")
    # Completing a host load and even a selected sidecar write is insufficient.
    arena.promote_selected([3], key[3:4], value[3:4], load_future=host_ready).result()
    with pytest.raises(RuntimeError, match="baseline device writes"):
        arena.write_selected_ready([1], key[1:2], value[1:2])
    arena.write([0, 1], key[:2], value[:2])
    with pytest.raises(RuntimeError, match="baseline device writes"):
        arena.write_selected_ready([1], key[1:2], value[1:2])
    arena.write([2, 3, 4], key[2:], value[2:])
    assert arena.write_selected_ready([1], key[1:2] + 1, value[1:2])[1] == 9


def test_pending_host_load_keeps_relocation_then_reuses_original_decode_slot():
    original, key, value, _ = fixture_arena()
    arena = NativePagedKV(original.cache, [4, 1, 6, 3], 5, 1)
    loading = Future()
    assert (
        arena.promote_selected(
            [1], key[1:2] + 1, value[1:2] + 1, load_future=loading
        ).result()[1]
        == 16
    )
    assert arena.stats()["sidecar_allocated_slots"] == 1
    assert arena.stats()["selected_ready_inplace_writes"] == 0
    torch.testing.assert_close(native_row(arena, 1)[0], key[1])
    loading.set_result(None)
    arena.write_prefill(key, value)
    arena.update_selected([1], key[1:2] + 2, value[1:2] + 2)
    assert arena.decode(5, key[0] + 3, value[0] + 3) == 9
    gathered_key, _ = arena.gather([1, 5])
    torch.testing.assert_close(gathered_key[0], key[1] + 2)
    torch.testing.assert_close(gathered_key[1], key[0] + 3)


@pytest.mark.parametrize("guard_kind", ["pin", "writer"])
def test_ready_selected_rejects_guarded_destination_before_any_copy(guard_kind):
    arena, key, value, _ = fixture_arena()
    guard = (
        arena.slot_map.pin_slots([9])
        if guard_kind == "pin"
        else arena.slot_map.write_guard([9])
    )
    with guard:
        with pytest.raises(RuntimeError, match="pinned or in-flight"):
            arena.write_selected_ready([4, 1], key[[4, 1]] + 1, value[[4, 1]] + 1)
        torch.testing.assert_close(native_row(arena, 4)[0], key[4])
        torch.testing.assert_close(native_row(arena, 1)[0], key[1])
    arena.write_selected_ready([4, 1], key[[4, 1]] + 1, value[[4, 1]] + 1)
    assert arena.stats()["sidecar_allocated_slots"] == 0


def test_ready_selected_rejects_repeated_or_relocated_tokens_and_decode():
    arena, key, value, _ = fixture_arena()
    arena.write_selected_ready([1], key[1:2], value[1:2])
    with pytest.raises(ValueError, match="update_selected"):
        arena.write_selected_ready([1], key[1:2], value[1:2])
    with pytest.raises(ValueError, match="update_selected"):
        arena.promote_selected([1], key[1:2], value[1:2])
    arena.promote_selected([3], key[3:4], value[3:4]).result()
    with pytest.raises(ValueError, match="update_selected"):
        arena.write_selected_ready([3], key[3:4], value[3:4])
    arena.update_selected([1, 3], key[[1, 3]] + 1, value[[1, 3]] + 1)
    torch.testing.assert_close(arena.gather([1, 3])[0], key[[1, 3]] + 1)
    arena.decode(5, key[0], value[0])
    with pytest.raises(RuntimeError, match="precede decode"):
        arena.write_selected_ready([0], key[:1], value[:1])


@pytest.mark.parametrize("operation", ["baseline", "ready"])
@pytest.mark.parametrize("outcome", ["success", "failure", "retired"])
def test_baseline_and_ready_guards_outlive_async_device_write(
    monkeypatch, operation, outcome
):
    arena, key, value, _ = fixture_arena()
    device_done, write_done = Future(), Future()
    enqueued = threading.Event()

    def completion(_):
        enqueued.set()
        return device_done

    def write():
        try:
            if operation == "baseline":
                arena.write_prefill(key + 1, value + 1)
            else:
                arena.write_selected_ready([1], key[1:2] + 1, value[1:2] + 1)
        except BaseException as error:
            write_done.set_exception(error)
        else:
            write_done.set_result(None)

    with monkeypatch.context() as patch:
        patch.setattr(paged, "_completion_future", completion)
        thread = threading.Thread(target=write, daemon=True)
        thread.start()
        try:
            assert enqueued.wait(3)
            assert not write_done.done()
            with pytest.raises(RuntimeError, match="being written"):
                with arena.slot_map.pinned_mapping([1]):
                    pass
            with pytest.raises(RuntimeError, match="in-flight"):
                with arena.slot_map.write_guard([9]):
                    pass
            if outcome == "retired":
                arena.slot_map.invalidate()
        finally:
            if outcome == "failure":
                device_done.set_exception(OSError("device write failed"))
            else:
                device_done.set_result(None)
            thread.join(timeout=3)
        assert not thread.is_alive()
        if outcome == "success":
            write_done.result()
            with arena.slot_map.pin_slots([9]):
                pass
        else:
            with pytest.raises(
                OSError if outcome == "failure" else StaleCompletionError
            ):
                write_done.result()
            with pytest.raises(StaleCompletionError):
                arena.snapshot()
        assert arena.slot_map.stats()["inflight_slots"] == 0


def test_baseline_write_does_not_overwrite_external_pins():
    arena, key, value, _ = fixture_arena()
    with arena.slot_map.pin_slots([9]):
        with pytest.raises(RuntimeError, match="pinned or in-flight"):
            arena.write_prefill(key + 1, value + 1)
        torch.testing.assert_close(arena.gather(range(5))[0], key)


def test_physical_page_selected_counts_follow_relocation_and_pin_guards():
    arena, key, value, _ = fixture_arena(max_selected=3)
    # Two selected rows coalesce into sidecar page 8, one into page 9.
    arena.promote_selected([0, 1, 3], key[[0, 1, 3]], value[[0, 1, 3]]).result()
    metadata = {page["page_id"]: page for page in arena.page_metadata()}
    assert {page: info["selected_count"] for page, info in metadata.items()} == {
        1: 0,
        6: 0,
        8: 2,
        9: 1,
    }
    assert arena.spill_order() == (1, 6, 8, 9)
    with arena.slot_map.pinned_mapping([0]):
        assert arena.spill_order() == (1, 6, 9)
        assert next(page for page in arena.page_metadata() if page["page_id"] == 8)[
            "pinned"
        ]
    # Metadata does not spill records, allocate native capacity, or drop logical KV.
    torch.testing.assert_close(arena.gather(range(5))[0], key)
    assert arena.stats()["native_pool_blocks_released"] == 0


def test_load_first_page_policy_counts_native_selected_rows_without_relocation():
    arena, key, value, _ = fixture_arena(max_selected=0)
    arena.write_selected_ready([0, 1, 3], key[[0, 1, 3]], value[[0, 1, 3]])
    assert arena.spill_order() == (6, 4, 1)
    assert {
        page["page_id"]: page["selected_count"] for page in arena.page_metadata()
    } == {
        1: 1,
        4: 2,
        6: 0,
    }
    assert arena.stats()["sidecar_allocated_slots"] == 0


def test_page_policy_excludes_shared_and_loading_sources():
    original, key, value, _ = fixture_arena()
    arena = NativePagedKV(original.cache, [4, 1, 6, 3], 5, 1, shared_slots={9})
    loading = Future()
    arena.promote_selected([3], key[3:4], value[3:4], load_future=loading).result()
    assert arena.spill_order() == (6, 8)
    metadata = {page["page_id"]: page for page in arena.page_metadata()}
    assert metadata[4]["shared"] and metadata[1]["inflight"]
    loading.set_result(None)
    assert arena.spill_order() == (1, 6, 8)


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


@pytest.mark.parametrize("reuse_hole", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "retired"])
def test_decode_publication_waits_for_copy_and_actual_device_completion(
    monkeypatch, reuse_hole, outcome
):
    arena, key, value, _ = fixture_arena(max_selected=1)
    if reuse_hole:
        arena.promote_selected([1], key[1:2], value[1:2]).result()
    initial = arena.snapshot()
    expected_slot, canonical_slot = (9 if reuse_hole else 13), 13
    copy_started, release_copy, event_waiting = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    device_done, decode_done = Future(), Future()
    original_write, original_completion = arena._write_slots, paged._completion_future

    class FakeEvent:
        def record(self, stream):
            assert stream == "decode-test-stream"

        def synchronize(self):
            event_waiting.set()
            device_done.result(timeout=3)

    def controlled_write(*args):
        copy_started.set()
        assert release_copy.wait(3)
        original_write(*args)

    def decode():
        try:
            result = arena.decode(5, key[0] + 2, value[0] + 3)
        except BaseException as error:
            decode_done.set_exception(error)
        else:
            decode_done.set_result(result)

    def assert_unpublished_and_guarded():
        assert arena.slot_map.snapshot() == initial
        with pytest.raises(KeyError):
            with arena.slot_map.pinned_mapping([5]):
                pass
        for slot in {expected_slot, canonical_slot}:
            with pytest.raises(ValueError, match="currently mapped"):
                with arena.slot_map.pin_slots([slot]):
                    pass
            with pytest.raises(RuntimeError, match="in-flight"):
                with arena.slot_map.write_guard([slot]):
                    pass
        with arena.slot_map.pinned_mapping([0, 1]):
            pass

    with monkeypatch.context() as patch:
        patch.setattr(arena, "_write_slots", controlled_write)
        patch.setattr(
            paged,
            "_completion_future",
            lambda _: original_completion(torch.device("cuda")),
        )
        patch.setattr(torch.cuda, "Event", FakeEvent)
        patch.setattr(torch.cuda, "current_stream", lambda _: "decode-test-stream")
        writer = threading.Thread(target=decode, daemon=True)
        writer.start()
        try:
            assert copy_started.wait(3)
            assert_unpublished_and_guarded()  # Before even the first physical copy.
            release_copy.set()
            assert event_waiting.wait(3)
            assert not decode_done.done()
            assert_unpublished_and_guarded()  # Enqueued is not device-complete.
            if outcome == "retired":
                arena.slot_map.invalidate()  # Must not deadlock behind a held map lock.
        finally:
            release_copy.set()
            if outcome == "failure":
                device_done.set_exception(OSError("decode device write failed"))
            else:
                device_done.set_result(None)
            writer.join(timeout=3)
        assert not writer.is_alive()
        assert arena.slot_map.stats()["inflight_slots"] == 0
        if outcome == "success":
            assert decode_done.result() == expected_slot
            with arena.slot_map.pinned_mapping([5]) as published:
                assert published == {5: expected_slot}
        else:
            with pytest.raises(
                OSError if outcome == "failure" else StaleCompletionError
            ):
                decode_done.result()
            assert arena.slot_map.stats()["mapped_tokens"] == len(initial)
            assert 5 not in arena._canonical
            with pytest.raises(StaleCompletionError):
                with arena.slot_map.pinned_mapping([5]):
                    pass
    if outcome == "success":
        gathered_key, gathered_value = arena.gather([5])
        torch.testing.assert_close(gathered_key[0], key[0] + 2)
        torch.testing.assert_close(gathered_value[0], value[0] + 3)
        torch.testing.assert_close(native_row(arena, 5)[0], key[0] + 2)
        torch.testing.assert_close(native_row(arena, 5)[1], value[0] + 3)


@pytest.mark.parametrize("reuse_hole", [False, True])
def test_decode_partial_host_copy_failure_never_publishes_new_token(
    monkeypatch, reuse_hole
):
    arena, key, value, _ = fixture_arena(max_selected=1)
    if reuse_hole:
        arena.promote_selected([1], key[1:2], value[1:2]).result()
    original_write = arena._write_slots

    def partially_failed_write(*args):
        original_write(*args)
        raise OSError("decode copy interrupted")

    monkeypatch.setattr(arena, "_write_slots", partially_failed_write)
    with pytest.raises(OSError, match="copy interrupted"):
        arena.decode(5, key[0], value[0])
    assert arena.slot_map.stats()["mapped_tokens"] == 5
    assert arena.slot_map.stats()["inflight_slots"] == 0
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
