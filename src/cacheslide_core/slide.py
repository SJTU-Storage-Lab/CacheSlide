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
        self._writes: Counter[int] = Counter()
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

    def page_metadata(
        self, page_size: int, selected_tokens: Iterable[int]
    ) -> tuple[dict[str, int | bool], ...]:
        """Snapshot physical selected-token counts and page-level guards.

        This is not an eviction reservation or backing-store dirty tracking.
        ``dirty`` uses the paper's selected-token-presence definition. A caller
        still needs coherent write-back/residency ownership before any eviction.
        """
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be a positive integer")
        selected = set(selected_tokens)
        with self._mutex:
            self._check_open()
            if any(type(token) is not int for token in selected) or not (
                selected.issubset(self._map)
            ):
                raise ValueError("selected tokens must have current logical mappings")
            mapped = Counter(slot // page_size for slot in self._map.values())
            counts = Counter(self._map[token] // page_size for token in selected)
            shared = {slot // page_size for slot in self._shared}
            pinned = {slot // page_size for slot, count in self._pins.items() if count}
            inflight = {
                slot // page_size for slot, count in self._inflight.items() if count
            }
            return tuple(
                {
                    "page_id": page,
                    "mapped_tokens": mapped[page],
                    "selected_count": counts[page],
                    "dirty": counts[page] > 0,
                    "shared": page in shared,
                    "pinned": page in pinned,
                    "inflight": page in inflight,
                    "spill_eligible": page not in shared | pinned | inflight,
                }
                for page in sorted(mapped)
            )

    @contextmanager
    def pin_slots(self, slots: Iterable[int]) -> Iterator[None]:
        unique = set(slots)
        with self._mutex:
            self._check_open()
            if not unique.issubset(set(self._map.values())):
                raise ValueError("only currently mapped slots can be pinned")
            if any(self._writes[slot] for slot in unique):
                raise RuntimeError("physical slots are being written")
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
            if any(self._writes[slot] for slot in slots):
                raise RuntimeError("physical slots are being written")
            self._pins.update(slots)
        try:
            yield snapshot
        finally:
            with self._mutex:
                self._pins.subtract(slots)

    @contextmanager
    def write_guard(
        self, slots: Iterable[int], *, optional_slots: Iterable[int] = ()
    ) -> Iterator[frozenset[int]]:
        """Reserve physical writes atomically against readers and reclamation.

        Required slots must be exclusive and unpinned; otherwise no reservation
        is made. Optional slots (e.g. canonical mirrors) are individually omitted
        if shared, pinned, or in flight. The caller must own these physical slots
        and keep this guard until actual device writes complete, not just enqueue.

        New pin attempts fail while a write is reserved. The map mutex is *not*
        held during the body, so completion callbacks and retirement can proceed.
        This does not publish token mappings or replace caller-side synchronization
        for other mapping/writing operations.
        """
        required_values, optional_values = tuple(slots), tuple(optional_slots)
        if any(
            type(slot) is not int or slot < 0
            for slot in required_values + optional_values
        ):
            raise ValueError("write slots must be nonnegative integers")
        required, optional = set(required_values), set(optional_values)
        with self._mutex:
            self._check_open()
            if required & self._shared:
                raise ValueError("required write slots cannot be shared")
            if any(self._pins[slot] or self._inflight[slot] for slot in required):
                raise RuntimeError("required write slots are pinned or in-flight")
            available = frozenset(
                slot
                for slot in optional
                if slot not in self._shared
                and not self._pins[slot]
                and not self._inflight[slot]
            )
            guarded = required | available
            generation = self._generation
            self._writes.update(guarded)
            self._inflight.update(guarded)
        try:
            yield available
            with self._mutex:
                if self._closed or generation != self._generation:
                    raise StaleCompletionError("write completed for a retired layer")
        finally:
            with self._mutex:
                self._writes.subtract(guarded)
                self._inflight.subtract(guarded)

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
            # Loading sources may still be read, but actual destinations must
            # reject new readers until the writer's completion (also in-place).
            writing_slots = set(destinations.values())
            self._writes.update(writing_slots)
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
                    self._writes.subtract(writing_slots)
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
            if (
                type(token) is not int
                or token < 0
                or token in self._map
                or token in self._pending_tokens
            ):
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

    def _decode_destination(
        self, token: int, physical_slot: int, *, reuse_vacated: bool
    ) -> int:
        """Validate and choose a native destination while the map mutex is held."""
        self._check_open()
        if (
            type(token) is not int
            or token < 0
            or token in self._map
            or token in self._pending_tokens
        ):
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
        return selected_slot

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
            selected_slot = self._decode_destination(
                token, physical_slot, reuse_vacated=reuse_vacated
            )
            self._vacated.discard(selected_slot)
            self._map[token] = selected_slot
            return selected_slot

    @contextmanager
    def decode_write_guard(
        self, token: int, physical_slot: int, *, reuse_vacated: bool = False
    ) -> Iterator[int]:
        """Reserve decode and mirror destinations, then publish completed KV.

        Selection and both write reservations are atomic. The new token remains
        absent from the public mapping until successful exit; readers cannot pin
        it while copies are pending. The caller must wait for actual device
        completion inside the body. No map mutex is held across that wait.

        These writes may be nontransactional, so an exception retires the entire
        map before releasing reservations. Unrelated existing mappings and pins
        remain usable during a successful pending write. ``bind_decode`` remains
        available for callers that already synchronized external writes.
        """
        with self._mutex:
            selected_slot = self._decode_destination(
                token, physical_slot, reuse_vacated=reuse_vacated
            )
            guarded = {selected_slot, physical_slot}
            generation = self._generation
            self._vacated.discard(selected_slot)
            self._pending_tokens.add(token)
            self._writes.update(guarded)
            self._inflight.update(guarded)
        try:
            yield selected_slot
            with self._mutex:
                if self._closed or generation != self._generation:
                    raise StaleCompletionError(
                        "decode write completed for a retired layer"
                    )
                self._map[token] = selected_slot
        except BaseException:
            with self._mutex:
                if not self._closed:
                    self.invalidate()
            raise
        finally:
            with self._mutex:
                self._writes.subtract(guarded)
                self._inflight.subtract(guarded)
                self._pending_tokens.discard(token)

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
