from concurrent.futures import Future

import pytest

from cacheslide_vllm.slide import LayerSlotMap, SlotWrite, coalesce_slot_writes
from cacheslide_vllm.storage import CacheCapacityError, StaleCompletionError


def test_coalesce_adjacent_slot_records():
    assert coalesce_slot_writes({4: b"dd", 2: b"bb", 1: b"aa"}) == [
        SlotWrite(1, 2, b"aabb"),
        SlotWrite(4, 1, b"dd"),
    ]
    with pytest.raises(ValueError):
        coalesce_slot_writes({0: b"a", 1: b"bb"})


def test_pending_load_redirects_only_after_write_and_reclaims_after_load():
    layer = LayerSlotMap({0: 0, 1: 1, 2: 2}, max_slots=6)
    loading, writing = Future(), Future()
    observed = []
    completion = layer.promote_selected(
        {1: b"new"}, lambda runs: observed.extend(runs) or writing, load_future=loading
    )
    assert observed == [SlotWrite(3, 1, b"new")]
    assert layer.snapshot() == {0: 0, 1: 1, 2: 2}
    # No source or destination in flight is available to decode.
    assert layer.allocate_decode(3) == 4
    writing.set_result(None)
    assert completion.result()[1] == 3
    assert layer.allocate_decode(4) == 5
    with pytest.raises(CacheCapacityError):
        layer.allocate_decode(5)
    loading.set_result(None)
    assert layer.allocate_decode(5) == 1
    assert layer.snapshot()[0] == 0 and layer.snapshot()[2] == 2


def test_ready_exclusive_atomic_writer_can_overwrite_original():
    loaded = Future()
    loaded.set_result(None)
    layer = LayerSlotMap({0: 0, 1: 1}, max_slots=2)
    observed = []
    result = layer.promote_selected(
        {1: b"x"},
        lambda runs: observed.extend(runs),
        load_future=loaded,
        allow_inplace=True,
    )
    assert result.result() == {0: 0, 1: 1}
    assert observed == [SlotWrite(1, 1, b"x")]
    assert layer.stats()["vacated_slots"] == 0


def test_shared_slots_never_overwritten_or_reclaimed():
    layer = LayerSlotMap({0: 0, 1: 1}, shared_slots={1}, max_slots=4)
    observed = []
    result = layer.promote_selected(
        {1: b"x"}, lambda runs: observed.extend(runs), allow_inplace=True
    )
    assert result.result()[1] == 2
    assert observed[0].start_slot == 2
    assert layer.allocate_decode(2) == 3
    with pytest.raises(CacheCapacityError):
        layer.allocate_decode(3)


def test_failure_preserves_original_mapping_and_data():
    layer = LayerSlotMap({0: 0, 1: 1}, max_slots=3)
    arena = {0: b"a", 1: b"b"}
    writing = Future()

    def partially_failed_writer(runs):
        arena[runs[0].start_slot] = runs[0].payload
        return writing

    completion = layer.promote_selected({1: b"X"}, partially_failed_writer)
    writing.set_exception(OSError("injected partial write failure"))
    with pytest.raises(OSError):
        completion.result()
    assert layer.snapshot() == {0: 0, 1: 1}
    assert arena[1] == b"b"
    # A failed extra destination can be reused after its writer has quiesced.
    assert layer.allocate_decode(2) == 2


def test_pins_delay_reclamation_and_force_copy_on_write():
    layer = LayerSlotMap({0: 0, 1: 1}, max_slots=3)
    with layer.pin_slots([1]):
        assert (
            layer.promote_selected(
                {1: b"x"}, lambda _: None, allow_inplace=True
            ).result()[1]
            == 2
        )
        with pytest.raises(CacheCapacityError):
            layer.allocate_decode(2)
    assert layer.allocate_decode(2) == 1


def test_stale_completion_cannot_publish_or_reopen_retired_mapping():
    layer = LayerSlotMap({0: 0})
    writing = Future()
    result = layer.promote_selected({0: b"x"}, lambda _: writing)
    layer.invalidate()
    writing.set_result(None)
    with pytest.raises(StaleCompletionError):
        result.result()
    with pytest.raises(StaleCompletionError):
        layer.snapshot()
    assert layer.stats()["inflight_slots"] == 0


def test_duplicate_pending_write_and_failed_load_are_rejected():
    layer = LayerSlotMap({0: 0})
    writing = Future()
    first = layer.promote_selected({0: b"x"}, lambda _: writing)
    with pytest.raises(RuntimeError):
        layer.promote_selected({0: b"y"}, lambda _: None)
    writing.set_result(None)
    first.result()
    loading = Future()
    loading.set_exception(OSError("load failed"))
    with pytest.raises(OSError):
        layer.promote_selected({0: b"y"}, lambda _: None, load_future=loading)


def test_failed_multi_slot_reservation_does_not_leak_capacity():
    layer = LayerSlotMap({0: 0, 1: 1}, max_slots=3)
    with pytest.raises(CacheCapacityError):
        layer.promote_selected({0: b"a", 1: b"b"}, lambda _: None)
    assert layer.snapshot() == {0: 0, 1: 1}
    assert layer.allocate_decode(2) == 2


def test_load_ready_race_preserves_independent_destination():
    layer = LayerSlotMap({0: 0}, max_slots=2)
    loading = Future()
    writing = Future()
    captured = []

    def writer(runs):
        captured.extend(runs)
        loading.set_result(None)
        return writing

    result = layer.promote_selected(
        {0: b"x"}, writer, load_future=loading, allow_inplace=True
    )
    assert captured[0].start_slot == 1
    writing.set_result(None)
    result.result()
    assert layer.allocate_decode(1) == 0


def test_sidecar_allocation_floor_prevents_native_pool_collisions():
    layer = LayerSlotMap({0: 2, 1: 3}, allocation_floor=100, max_slots=102)
    runs = []
    layer.promote_selected(
        {0: b"a", 1: b"b"}, lambda writes: runs.extend(writes)
    ).result()
    assert runs == [SlotWrite(100, 2, b"ab")]
    assert layer.snapshot() == {0: 100, 1: 101}
    # A canonical native decode slot does not consume fresh sidecar capacity.
    assert layer.bind_decode(2, 20) == 20
    with pytest.raises(ValueError):
        layer.bind_decode(3, 102)


def test_bind_decode_reclaims_only_safe_native_holes():
    layer = LayerSlotMap({0: 0, 1: 1}, allocation_floor=8, max_slots=10)
    loading, writing = Future(), Future()
    completion = layer.promote_selected(
        {1: b"x"}, lambda _: writing, load_future=loading
    )
    writing.set_result(None)
    completion.result()
    # The vacated source is still receiving its old layer load.
    assert layer.bind_decode(2, 2, reuse_vacated=True) == 2
    loading.set_result(None)
    assert layer.bind_decode(3, 3, reuse_vacated=True) == 1
    assert layer.snapshot()[3] == 1
    # No remaining native hole: bind the supplied canonical slot, never virtual.
    assert layer.bind_decode(4, 4, reuse_vacated=True) == 4


def test_native_binding_rejects_valid_shared_pinned_and_inflight_slots():
    layer = LayerSlotMap({0: 0, 1: 1}, shared_slots={0}, allocation_floor=10)
    with pytest.raises(CacheCapacityError):
        layer.bind_decode(2, 0)
    with layer.pinned_mapping([1]) as snapshot:
        layer.promote_selected({1: b"x"}, lambda _: None).result()
        assert snapshot == {1: 1}
        with pytest.raises(CacheCapacityError):
            layer.bind_decode(2, 1)
    assert layer.bind_decode(2, 1) == 1
    layer.promote_selected({0: b"x"}, lambda _: None).result()
    with pytest.raises(CacheCapacityError):
        layer.bind_decode(3, 0)
    layer.invalidate()
    with pytest.raises(StaleCompletionError):
        layer.bind_decode(3, 3)


def test_atomic_pinned_mapping_preserves_old_slots_across_publication():
    layer = LayerSlotMap({0: 0, 1: 1}, allocation_floor=8)
    with layer.pinned_mapping([0, 1]) as snapshot:
        layer.promote_selected({1: b"x"}, lambda _: None).result()
        assert snapshot == {0: 0, 1: 1}
        assert layer.snapshot()[1] == 8
        assert layer.bind_decode(2, 2, reuse_vacated=True) == 2
    assert layer.bind_decode(3, 3, reuse_vacated=True) == 1


@pytest.mark.parametrize("floor", [-1, True, 1.0])
def test_allocation_floor_requires_nonnegative_integer(floor):
    with pytest.raises(ValueError, match="allocation_floor"):
        LayerSlotMap({}, allocation_floor=floor)
