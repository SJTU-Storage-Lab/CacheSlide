from concurrent.futures import Future

import pytest

from cacheslide_core.slide import LayerSlotMap, SlotWrite, coalesce_slot_writes
from cacheslide_core.storage import CacheCapacityError, StaleCompletionError


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


@pytest.mark.parametrize("outcome", ["success", "failure", "retired"])
def test_async_inplace_promotion_rejects_new_pins_and_concurrent_writers(outcome):
    layer = LayerSlotMap({0: 0, 1: 1}, max_slots=2)
    written = Future()
    completion = layer.promote_selected(
        {0: b"new"}, lambda _: written, allow_inplace=True
    )
    assert not completion.done()
    assert layer.snapshot() == {0: 0, 1: 1}
    for pin in (layer.pin_slots([0]), layer.pinned_mapping([0])):
        with pytest.raises(RuntimeError, match="being written"):
            with pin:
                pass
    with pytest.raises(RuntimeError, match="in-flight"):
        with layer.write_guard([0]):
            pass
    with layer.pinned_mapping([1]):
        assert layer.stats()["pinned_slots"] == 1
    if outcome == "retired":
        layer.invalidate()
    if outcome == "failure":
        written.set_exception(OSError("transaction rolled back"))
        with pytest.raises(OSError, match="rolled back"):
            completion.result()
    else:
        written.set_result(None)
        if outcome == "retired":
            with pytest.raises(StaleCompletionError):
                completion.result()
        else:
            assert completion.result() == {0: 0, 1: 1}
    assert layer.stats()["inflight_slots"] == 0
    if outcome != "retired":
        with layer.pin_slots([0]):
            pass
        with layer.write_guard([0]):
            pass


def test_existing_write_guard_forces_promotion_to_independent_destination():
    layer = LayerSlotMap({0: 0}, max_slots=2)
    written = Future()
    with layer.write_guard([0]):
        completion = layer.promote_selected(
            {0: b"new"}, lambda _: written, allow_inplace=True
        )
        written.set_result(None)
        assert completion.result() == {0: 1}
        with pytest.raises(CacheCapacityError):
            layer.allocate_decode(1)
    assert layer.allocate_decode(1) == 0


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


@pytest.mark.parametrize("reuse_hole", [False, True])
def test_decode_write_reserves_both_destinations_before_mapping_publication(reuse_hole):
    layer = LayerSlotMap({0: 0, 1: 1}, allocation_floor=8, max_slots=9)
    if reuse_hole:
        layer.promote_selected({1: b"selected"}, lambda _: None).result()
    initial = layer.snapshot()
    expected = 1 if reuse_hole else 2
    with layer.decode_write_guard(2, 2, reuse_vacated=True) as chosen:
        assert chosen == expected
        assert layer.snapshot() == initial
        assert layer.stats()["inflight_slots"] == (2 if reuse_hole else 1)
        with pytest.raises(KeyError):
            with layer.pinned_mapping([2]):
                pass
        for destination in {chosen, 2}:
            with pytest.raises(ValueError, match="currently mapped"):
                with layer.pin_slots([destination]):
                    pass
            with pytest.raises(RuntimeError, match="in-flight"):
                with layer.write_guard([destination]):
                    pass
            with pytest.raises(CacheCapacityError, match="guarded"):
                layer.bind_decode(3, destination)
        with pytest.raises(ValueError, match="new nonnegative"):
            layer.bind_decode(2, 3)
        with pytest.raises(ValueError, match="new nonnegative"):
            layer.allocate_decode(2)
        with layer.pinned_mapping([0]):
            pass  # The reservation does not block unrelated readers.
    assert layer.snapshot() == {**initial, 2: expected}
    assert layer.stats()["inflight_slots"] == 0
    with layer.pinned_mapping([2]) as published:
        assert published == {2: expected}


@pytest.mark.parametrize("outcome", ["failure", "retired"])
def test_failed_decode_write_retires_without_publishing_or_leaking_guards(outcome):
    layer = LayerSlotMap({0: 0}, allocation_floor=8)
    expected = OSError if outcome == "failure" else StaleCompletionError
    with pytest.raises(expected):
        with layer.decode_write_guard(1, 1):
            if outcome == "failure":
                raise OSError("partially copied KV")
            layer.invalidate()
    assert layer.stats()["mapped_tokens"] == 1
    assert layer.stats()["inflight_slots"] == 0
    with pytest.raises(StaleCompletionError):
        layer.snapshot()
    with pytest.raises(StaleCompletionError):
        with layer.pinned_mapping([1]):
            pass


def test_decode_write_precondition_failure_keeps_existing_map_usable():
    layer = LayerSlotMap({0: 0}, allocation_floor=8)
    with pytest.raises(CacheCapacityError):
        with layer.decode_write_guard(1, 0):
            pytest.fail("invalid destination must fail before the write")
    assert layer.snapshot() == {0: 0}
    assert layer.stats()["inflight_slots"] == 0
    with layer.decode_write_guard(1, 1):
        pass
    assert layer.snapshot() == {0: 0, 1: 1}


def test_atomic_pinned_mapping_preserves_old_slots_across_publication():
    layer = LayerSlotMap({0: 0, 1: 1}, allocation_floor=8)
    with layer.pinned_mapping([0, 1]) as snapshot:
        layer.promote_selected({1: b"x"}, lambda _: None).result()
        assert snapshot == {0: 0, 1: 1}
        assert layer.snapshot()[1] == 8
        assert layer.bind_decode(2, 2, reuse_vacated=True) == 2
    assert layer.bind_decode(3, 3, reuse_vacated=True) == 1


def test_write_guard_filters_optional_slots_and_blocks_hole_reclamation():
    layer = LayerSlotMap(
        {0: 0, 1: 1, 2: 2}, shared_slots={0}, allocation_floor=8, max_slots=11
    )
    loading = Future()
    with layer.pin_slots([1]):
        layer.promote_selected({1: b"x", 2: b"y"}, lambda _: None).result()
        with layer.write_guard([8, 9], optional_slots=[0, 1, 2]) as mirrors:
            assert mirrors == {2}  # Skip shared 0 and externally pinned 1 only.
            assert layer.stats()["inflight_slots"] == 3
            assert layer.bind_decode(3, 3, reuse_vacated=True) == 3
            with pytest.raises(RuntimeError, match="being written"):
                with layer.pin_slots([8]):
                    pass
            with pytest.raises(RuntimeError, match="being written"):
                with layer.pinned_mapping([1]):
                    pass
        assert layer.bind_decode(4, 4, reuse_vacated=True) == 2
    assert layer.bind_decode(5, 5, reuse_vacated=True) == 1
    layer.promote_selected({3: b"z"}, lambda _: None, load_future=loading).result()
    with layer.write_guard([], optional_slots=[3]) as mirrors:
        assert mirrors == set()  # A still-loading source cannot be mirrored.
        loading.set_result(None)  # Callback needs the map mutex; must not deadlock.
    with layer.write_guard([], optional_slots=[3]) as mirrors:
        assert mirrors == {3}


@pytest.mark.parametrize("guard_kind", ["pin", "write"])
def test_required_write_guard_rejection_is_atomic_and_recoverable(guard_kind):
    layer = LayerSlotMap({0: 0, 1: 1})
    guard = layer.pin_slots([1]) if guard_kind == "pin" else layer.write_guard([1])
    with guard:
        with pytest.raises(RuntimeError, match="pinned or in-flight"):
            with layer.write_guard([0, 1]):
                pytest.fail("must reject all writes before entering the body")
        # Failed multi-slot reservations must not retain a partial write guard.
        with layer.pin_slots([0]):
            pass
    with layer.write_guard([0, 1]):
        assert layer.stats()["inflight_slots"] == 2
    assert layer.stats()["inflight_slots"] == 0
    with layer.pin_slots([0, 1]):
        pass


def test_write_guard_releases_reservations_on_error_and_retirement():
    layer = LayerSlotMap({0: 0}, shared_slots=())
    with pytest.raises(OSError, match="copy failed"):
        with layer.write_guard([0]):
            raise OSError("copy failed")
    assert layer.stats()["inflight_slots"] == 0
    with layer.pin_slots([0]):
        pass
    with pytest.raises(StaleCompletionError, match="retired"):
        with layer.write_guard([0]):
            layer.invalidate()
    assert layer.stats()["inflight_slots"] == 0


def test_write_guard_validates_slots_and_shared_destinations():
    layer = LayerSlotMap({0: 0}, shared_slots={0})
    with pytest.raises(ValueError, match="shared"):
        with layer.write_guard([0]):
            pass
    for invalid in (-1, True, 1.0):
        with pytest.raises(ValueError, match="nonnegative integers"):
            with layer.write_guard([], optional_slots=[1, invalid]):
                pass
    assert layer.stats()["inflight_slots"] == 0


@pytest.mark.parametrize("floor", [-1, True, 1.0])
def test_allocation_floor_requires_nonnegative_integer(floor):
    with pytest.raises(ValueError, match="allocation_floor"):
        LayerSlotMap({}, allocation_floor=floor)
