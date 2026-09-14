"""Per-layer SLIDE slot redirection with completion-ordered publication.

The slots belong to a sidecar arena supplied by the caller, not vLLM's native
block pool. A writer must publish its Future only when all destination writes
are complete. Shared and loading source slots always use copy-on-write.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass

from .storage import CacheCapacityError, StaleCompletionError


@dataclass(frozen=True)
class SlotWrite:
    start_slot: int
    slot_count: int
    payload: bytes


def coalesce_slot_writes(writes: Mapping[int, bytes]) -> list[SlotWrite]:
    """Combine adjacent equal-sized token records into contiguous writes."""
    if not writes:
        return []
    sizes = {len(payload) for payload in writes.values()}
    if len(sizes) != 1 or next(iter(sizes)) == 0:
        raise ValueError("token records must have one positive byte size")
    if any(
        type(slot) is not int or slot < 0 or not isinstance(data, bytes)
        for slot, data in writes.items()
    ):
        raise ValueError("slots must be nonnegative integers and records bytes")
    result = []
    start = previous = -2
    parts: list[bytes] = []
    for slot, payload in sorted(writes.items()):
        if slot != previous + 1 and parts:
            result.append(SlotWrite(start, len(parts), b"".join(parts)))
            parts = []
        if not parts:
            start = slot
        parts.append(payload)
        previous = slot
    result.append(SlotWrite(start, len(parts), b"".join(parts)))
    return result


class LayerSlotMap:
    """Thread-safe token-to-slot map for one request and one model layer.

    Fresh slots must refer to unused space in the caller's sidecar arena.
    Reclaimed source slots become available for decode only after selected KV
    writes commit and all source loads/pins finish. ``invalidate`` retires this
    map permanently; late callbacks cannot publish mappings into another request.
    """

    def __init__(
        self,
        initial_slots: Mapping[int, int],
        *,
        shared_slots: Iterable[int] = (),
        max_slots: int | None = None,
        allocation_floor: int = 0,
    ):
        self._map = dict(initial_slots)
        if any(
            type(token) is not int or token < 0 or type(slot) is not int or slot < 0
            for token, slot in self._map.items()
        ):
            raise ValueError("tokens and slots must be nonnegative integers")
        self._shared = set(shared_slots)
        if not self._shared.issubset(set(self._map.values())):
            raise ValueError("shared slots must be present in the initial mapping")
        if type(allocation_floor) is not int or allocation_floor < 0:
            raise ValueError("allocation_floor must be a nonnegative integer")
        self.allocation_floor = allocation_floor
        self._next_slot = max(allocation_floor, max(self._map.values(), default=-1) + 1)
        if max_slots is not None and (
            type(max_slots) is not int or max_slots < self._next_slot or max_slots < 0
        ):
            raise ValueError("max_slots is smaller than the initial arena")
        self.max_slots = max_slots
        self._mutex = threading.RLock()
        self._pins: Counter[int] = Counter()
        self._inflight: Counter[int] = Counter()
        self._pending_tokens: set[int] = set()
        self._vacated: set[int] = set()
        self._unused: set[int] = set()
        self._generation = 0
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise StaleCompletionError("layer slot map is retired")

    def _fresh_slot(self) -> int:
        if self._unused:
            slot = min(self._unused)
            self._unused.remove(slot)
            return slot
        if self.max_slots is not None and self._next_slot >= self.max_slots:
            raise CacheCapacityError("sidecar slot arena is full")
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def snapshot(self) -> dict[int, int]:
        with self._mutex:
            self._check_open()
            return dict(self._map)

    @contextmanager
    def pin_slots(self, slots: Iterable[int]) -> Iterator[None]:
        unique = set(slots)
        with self._mutex:
            self._check_open()
            if not unique.issubset(set(self._map.values())):
                raise ValueError("only currently mapped slots can be pinned")
            self._pins.update(unique)
        try:
            yield
        finally:
            with self._mutex:
                self._pins.subtract(unique)

    @contextmanager
    def pinned_mapping(self, tokens: Iterable[int]) -> Iterator[dict[int, int]]:
        """Atomically snapshot and pin selected token mappings during a gather."""
        tokens = tuple(tokens)
        with self._mutex:
            self._check_open()
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("tokens must be nonnegative integer positions")
            snapshot = {token: self._map[token] for token in tokens}
            slots = set(snapshot.values())
            self._pins.update(slots)
        try:
            yield snapshot
        finally:
            with self._mutex:
                self._pins.subtract(slots)

    def promote_selected(
        self,
        updates: Mapping[int, bytes],
        writer: Callable[[list[SlotWrite]], Future | None],
        *,
        load_future: Future | None = None,
        allow_inplace: bool = False,
    ) -> Future[dict[int, int]]:
        """Write selected tokens and atomically publish their slot mappings.

        When the source load is pending, selected tokens go to extra slots.
        ``allow_inplace=True`` permits exclusive unpinned slots to be overwritten
        after a successful load. It requires a transactional writer: failure
        must preserve old bytes. The default uses independent destinations and
        therefore preserves original data even after a partially failed write.
        """
        result: Future[dict[int, int]] = Future()
        result.set_running_or_notify_cancel()
        updates = dict(updates)
        # Validate before acquiring any destination slots.
        coalesce_slot_writes({i: data for i, data in enumerate(updates.values())})
        with self._mutex:
            self._check_open()
            if not updates:
                result.set_result(dict(self._map))
                return result
            if not set(updates).issubset(self._map):
                raise KeyError("selected tokens must already have source slots")
            if self._pending_tokens.intersection(updates):
                raise RuntimeError("selected tokens already have an in-flight write")
            load_pending = load_future is not None and not load_future.done()
            if load_future is not None and not load_pending:
                # A failed/cancelled load cannot justify overwriting its slots.
                load_future.result()
            generation = self._generation
            source = {token: self._map[token] for token in updates}
            multiplicity = Counter(self._map.values())
            destinations: dict[int, int] = {}
            allocated: set[int] = set()
            try:
                for token, old_slot in source.items():
                    must_copy = (
                        not allow_inplace
                        or load_pending
                        or old_slot in self._shared
                        or self._pins[old_slot]
                        or self._inflight[old_slot]
                        or multiplicity[old_slot] > 1
                    )
                    destination = self._fresh_slot() if must_copy else old_slot
                    destinations[token] = destination
                    if destination != old_slot:
                        allocated.add(destination)
            except BaseException:
                self._unused.update(allocated)
                raise
            self._pending_tokens.update(updates)
            guarded = set(source.values()) | set(destinations.values())
            self._inflight.update(guarded)
            if load_pending:
                # These guards outlive selected-write completion if loading continues.
                loading_slots = set(source.values())
                self._inflight.update(loading_slots)

                def load_finished(_: Future) -> None:
                    with self._mutex:
                        self._inflight.subtract(loading_slots)

                load_future.add_done_callback(load_finished)

        runs = coalesce_slot_writes(
            {destinations[token]: data for token, data in updates.items()}
        )

        def finish(
            write_future: Future | None = None, failure: BaseException | None = None
        ) -> None:
            completion_error: BaseException | None = None
            snapshot: dict[int, int] | None = None
            try:
                if failure is not None:
                    raise failure
                if write_future is not None:
                    write_future.result()
                with self._mutex:
                    if self._closed or generation != self._generation:
                        raise StaleCompletionError(
                            "selected write completed for a retired layer"
                        )
                    for token, old_slot in source.items():
                        if self._map[token] != old_slot:
                            raise StaleCompletionError(
                                "source mapping changed before publication"
                            )
                    self._map.update(destinations)
                    still_valid = set(self._map.values())
                    for old_slot in source.values():
                        if old_slot not in still_valid and old_slot not in self._shared:
                            self._vacated.add(old_slot)
                    snapshot = dict(self._map)
            except BaseException as exc:
                with self._mutex:
                    self._unused.update(allocated)
                completion_error = exc
            finally:
                with self._mutex:
                    self._inflight.subtract(guarded)
                    self._pending_tokens.difference_update(updates)
            if completion_error is not None:
                result.set_exception(completion_error)
            else:
                assert snapshot is not None
                result.set_result(snapshot)

        try:
            written = writer(runs)
            if written is None:
                finish()
            elif isinstance(written, Future):
                written.add_done_callback(finish)
            else:
                raise TypeError("writer must return a concurrent Future or None")
        except BaseException as exc:
            finish(failure=exc)
        return result

    def allocate_decode(self, token: int) -> int:
        with self._mutex:
            self._check_open()
            if type(token) is not int or token < 0 or token in self._map:
                raise ValueError("decode token must be a new nonnegative position")
            valid = set(self._map.values())
            eligible = sorted(
                slot
                for slot in self._vacated
                if slot not in valid
                and slot not in self._shared
                and not self._pins[slot]
                and not self._inflight[slot]
            )
            if eligible:
                slot = eligible[0]
                self._vacated.remove(slot)
            else:
                slot = self._fresh_slot()
            self._map[token] = slot
            return slot

    def bind_decode(
        self, token: int, physical_slot: int, *, reuse_vacated: bool = False
    ) -> int:
        """Bind a caller-owned native slot below the sidecar allocation floor.

        The caller must obtain exclusive ownership from its native allocator.
        This method neither allocates native slots nor changes native page tables.
        A currently valid/shared/pinned/in-flight slot cannot be rebound.
        ``reuse_vacated`` first chooses an eligible native hole. The supplied
        canonical slot is checked even then so the caller can mirror its write.
        Caller-side synchronization must cover actual writes and gathers.
        """
        with self._mutex:
            self._check_open()
            if type(token) is not int or token < 0 or token in self._map:
                raise ValueError("decode token must be a new nonnegative position")
            if (
                type(physical_slot) is not int
                or not 0 <= physical_slot < self.allocation_floor
            ):
                raise ValueError("native decode slot must be below allocation_floor")
            if (
                physical_slot in self._map.values()
                or physical_slot in self._shared
                or self._pins[physical_slot]
                or self._inflight[physical_slot]
            ):
                raise CacheCapacityError("native decode slot is valid or guarded")
            selected_slot = physical_slot
            if reuse_vacated:
                valid = set(self._map.values())
                eligible = sorted(
                    slot
                    for slot in self._vacated
                    if slot < self.allocation_floor
                    and slot not in valid
                    and slot not in self._shared
                    and not self._pins[slot]
                    and not self._inflight[slot]
                )
                if eligible:
                    selected_slot = eligible[0]
            self._vacated.discard(selected_slot)
            self._map[token] = selected_slot
            return selected_slot

    def invalidate(self) -> None:
        with self._mutex:
            self._generation += 1
            self._closed = True

    def stats(self) -> dict[str, int]:
        with self._mutex:
            return {
                "mapped_tokens": len(self._map),
                "vacated_slots": len(self._vacated),
                "shared_slots": len(self._shared),
                "pinned_slots": sum(count > 0 for count in self._pins.values()),
                "inflight_slots": sum(count > 0 for count in self._inflight.values()),
                "generation": self._generation,
            }
