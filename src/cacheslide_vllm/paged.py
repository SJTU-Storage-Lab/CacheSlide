"""Request-local SLIDE indirection over vLLM's packed native KV cache.

The native cache is [block, KV head, block offset, 2 * head dimension].
Only slots named by the caller's block table are writable. Selected KV uses a
bounded device-local sidecar; native block ownership and pool capacity remain
with vLLM. Once an original slot is reused, attention must gather through this
map: mirroring new decode rows does not restore displaced old canonical rows.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import Future

import torch

from .slide import LayerSlotMap, SlotWrite


def _indices(value: Sequence[int] | torch.Tensor, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be an integer vector")
        # Only token/slot metadata crosses devices; KV records remain on device.
        value = value.detach().cpu().tolist()
    result = list(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{name} must contain nonnegative integer positions")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain distinct positions")
    return result


def _completion_future(device: torch.device) -> Future:
    """CUDA event completion, not host enqueue, makes a write publishable."""
    completed = Future()
    completed.set_running_or_notify_cancel()
    if device.type != "cuda":
        completed.set_result(None)
        return completed
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))

    def finish() -> None:
        try:
            event.synchronize()
        except BaseException as error:
            completed.set_exception(error)
        else:
            completed.set_result(None)

    threading.Thread(target=finish, name="cacheslide-kv-write", daemon=True).start()
    return completed


class NativePagedKV:
    """One request/layer's physical native slots plus selected-token sidecar.

    The block table may contain trailing -1 entries for unallocated blocks.
    Before decode crosses a block boundary, the caller must provide its freshly
    allocated table through ``update_block_table``. The adapter never allocates,
    frees, or claims capacity from vLLM's native block pool.
    """

    def __init__(
        self,
        cache: torch.Tensor,
        block_table: Sequence[int] | torch.Tensor,
        prompt_length: int,
        max_selected: int,
        *,
        request_id: str = "",
        layer_index: int = 0,
        shared_slots: Iterable[int] = (),
    ):
        if (
            not isinstance(cache, torch.Tensor)
            or cache.ndim != 4
            or min(cache.shape) < 1
            or cache.shape[-1] % 2
            or cache.dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or cache.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("cache must be floating [blocks,KV heads,block size,2*D]")
        if type(prompt_length) is not int or prompt_length < 1:
            raise ValueError("prompt_length must be a positive integer")
        if type(max_selected) is not int or max_selected < 0:
            raise ValueError("max_selected must be a nonnegative integer")
        if type(layer_index) is not int or layer_index < 0:
            raise ValueError("layer_index must be a nonnegative integer")
        self.cache = cache
        self.prompt_length = prompt_length
        self.request_id = request_id
        self.layer_index = layer_index
        self.block_size = cache.shape[2]
        self.kv_heads = cache.shape[1]
        self.head_dim = cache.shape[3] // 2
        self.native_total_slots = cache.shape[0] * self.block_size
        self.max_selected = max_selected
        self._mutex = threading.RLock()
        self._block_table = self._table(block_table)
        self._canonical = {
            token: self._canonical_slot(token) for token in range(prompt_length)
        }
        if len(set(self._canonical.values())) != prompt_length:
            raise ValueError("active logical blocks must name distinct native slots")
        self._shared = set(shared_slots)
        self.slot_map = LayerSlotMap(
            self._canonical,
            shared_slots=self._shared,
            allocation_floor=self.native_total_slots,
            max_slots=self.native_total_slots + max_selected,
        )
        # The ready-load branch needs no extra KV copy or device allocation.
        # Reserve only a logical capacity; allocate lazily on first relocation.
        self._side_key: torch.Tensor | None = None
        self._side_value: torch.Tensor | None = None
        self._baseline_ready: set[int] = set()
        self._loads: list[Future] = []
        self._promotions: list[Future] = []
        self._selected: set[int] = set()
        self._decoded = False
        self._hole_reuses = 0
        self._native_binds = 0
        self._canonical_mirrors = 0
        self._ready_inplace_writes = 0

    def _table(self, block_table: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(block_table, torch.Tensor):
            if block_table.ndim != 1 or block_table.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("block_table must be a one-dimensional integer row")
            block_table = block_table.detach().cpu().tolist()
        row = tuple(block_table)
        if not row or any(
            type(block) is not int or not -1 <= block < self.cache.shape[0]
            for block in row
        ):
            raise ValueError("block_table contains invalid native block IDs")
        return row

    def _canonical_slot(self, token: int, row: tuple[int, ...] | None = None) -> int:
        row = self._block_table if row is None else row
        block_index, offset = divmod(token, self.block_size)
        if block_index >= len(row) or row[block_index] < 0:
            raise ValueError("logical token has no caller-owned native block")
        return row[block_index] * self.block_size + offset

    def update_block_table(self, block_table: Sequence[int] | torch.Tensor) -> None:
        """Accept caller-owned new blocks without moving existing canonical slots."""
        row = self._table(block_table)
        with self._mutex:
            self.slot_map.snapshot()
            if any(
                self._canonical_slot(token, row) != slot
                for token, slot in self._canonical.items()
            ):
                raise ValueError("block-table update moved an existing native token")
            self._block_table = row

    def canonical_slot(self, token: int) -> int:
        """Resolve a logical token through the caller's current native block table."""
        if type(token) is not int or token < 0:
            raise ValueError("token must be a nonnegative integer")
        with self._mutex:
            self.slot_map.snapshot()
            return self._canonical_slot(token)

    def _records(
        self, positions: list[int], key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (len(positions), self.kv_heads, self.head_dim)
        for tensor in (key, value):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected
                or tensor.device != self.cache.device
                or tensor.dtype != self.cache.dtype
            ):
                raise ValueError(
                    "K/V must match cache device/dtype and [token,KV head,D]"
                )
        return key, value

    def _check_loads(self, *, allow_pending: bool = False) -> None:
        for loading in self._loads:
            if not loading.done():
                if allow_pending:
                    continue
                raise RuntimeError(
                    "source KV load must complete before baseline/mirror writes"
                )
            loading.result()

    def _check_promotions(self) -> None:
        for promotion in self._promotions:
            if not promotion.done():
                raise RuntimeError("selected writes must complete before updating them")
            promotion.result()

    def _write_slots(
        self, slots: list[int], key: torch.Tensor, value: torch.Tensor
    ) -> None:
        native_rows = [
            row for row, slot in enumerate(slots) if slot < self.native_total_slots
        ]
        extra_rows = [
            row for row, slot in enumerate(slots) if slot >= self.native_total_slots
        ]
        device = self.cache.device
        if native_rows:
            selected = torch.tensor(native_rows, device=device)
            blocks = torch.tensor(
                [slots[row] // self.block_size for row in native_rows], device=device
            )
            offsets = torch.tensor(
                [slots[row] % self.block_size for row in native_rows], device=device
            )
            # Index the actual strides; reshape would silently copy some native views.
            self.cache[blocks, :, offsets, : self.head_dim] = key[selected]
            self.cache[blocks, :, offsets, self.head_dim :] = value[selected]
        if extra_rows:
            if self._side_key is None:
                shape = (self.max_selected, self.kv_heads, self.head_dim)
                self._side_key, self._side_value = (
                    self.cache.new_empty(shape),
                    self.cache.new_empty(shape),
                )
            assert self._side_value is not None
            selected = torch.tensor(extra_rows, device=device)
            extra = torch.tensor(
                [slots[row] - self.native_total_slots for row in extra_rows],
                device=device,
            )
            self._side_key[extra] = key[selected]
            self._side_value[extra] = value[selected]

    def write(
        self,
        logical_positions: Sequence[int] | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Load baseline KV into original native slots, regardless of redirection."""
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        with self._mutex:
            self.slot_map.snapshot()
            self._check_loads()
            if self._decoded:
                raise RuntimeError(
                    "baseline writes cannot overwrite slots after decode"
                )
            if any(token not in self._canonical for token in positions):
                raise KeyError("baseline writes require existing logical tokens")
            slots = [self._canonical[token] for token in positions]
            if self._shared.intersection(slots):
                raise ValueError("baseline writes cannot mutate shared native slots")
            with self.slot_map.write_guard(slots):
                try:
                    self._write_slots(slots, key, value)
                    _completion_future(self.cache.device).result()
                except BaseException:
                    self.slot_map.invalidate()
                    raise
            # A completed host read is not evidence of native baseline readiness.
            # Only this completed physical device write can establish that fact.
            self._baseline_ready.update(positions)

    load_baseline = write

    def write_prefill(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.write(range(self.prompt_length), key, value)

    def promote_selected(
        self,
        logical_positions: Sequence[int] | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        load_future: Future | None = None,
    ) -> Future[dict[int, int]]:
        """Stage selected device KV while original baseline loading is outstanding."""
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        if load_future is not None and not isinstance(load_future, Future):
            raise TypeError("load_future must be a concurrent Future")
        with self._mutex:
            if self._decoded:
                raise RuntimeError("selected promotion must precede decode")
            if self._selected.intersection(positions):
                raise ValueError("already staged tokens must use update_selected")
            if load_future is not None and load_future.done():
                load_future.result()

            def writer(runs: list[SlotWrite]) -> Future:
                slots, rows = [], []
                for run in runs:
                    for offset in range(run.slot_count):
                        slots.append(run.start_slot + offset)
                        rows.append(
                            int.from_bytes(
                                run.payload[offset * 8 : (offset + 1) * 8], "little"
                            )
                        )
                row_indices = torch.tensor(rows, device=key.device, dtype=torch.long)
                self._write_slots(slots, key[row_indices], value[row_indices])
                return _completion_future(self.cache.device)

            # Byte payloads identify tensor rows; no KV serialization or CPU transfer.
            updates = {
                token: row.to_bytes(8, "little") for row, token in enumerate(positions)
            }
            promotion = self.slot_map.promote_selected(
                updates,
                writer,
                load_future=load_future,
                allow_inplace=False,
            )
            self._selected.update(positions)
            self._promotions.append(promotion)
            if load_future is not None and load_future not in self._loads:
                self._loads.append(load_future)
            return promotion

    stage_selected = promote_selected

    def write_selected_ready(
        self,
        logical_positions: Sequence[int] | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> dict[int, int]:
        """Load-first branch: update original slots without a sidecar or holes.

        The entire prompt baseline must have completed ``write``/``write_prefill``
        on this arena's device. A host Future becoming ready cannot establish
        this precondition. Required slots must remain original, exclusive, and
        unpinned. This method waits for actual device completion; a partial or
        failed nontransactional copy retires the arena for full-prefill fallback.
        Further fused updates use ``update_selected``; the pending-load branch
        continues to use ``promote_selected``.
        """
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        with self._mutex:
            mapping = self.slot_map.snapshot()
            self._check_loads()
            self._check_promotions()
            if self._decoded:
                raise RuntimeError("ready selected writes must precede decode")
            if len(self._baseline_ready) != self.prompt_length:
                raise RuntimeError("native baseline device writes must complete first")
            if not set(positions).issubset(self._canonical):
                raise ValueError("ready selected tokens require original native slots")
            if self._selected.intersection(positions) or any(
                mapping[token] != self._canonical[token] for token in positions
            ):
                raise ValueError("already selected tokens must use update_selected")
            slots = [self._canonical[token] for token in positions]
            with self.slot_map.write_guard(slots):
                try:
                    self._write_slots(slots, key, value)
                    _completion_future(self.cache.device).result()
                except BaseException:
                    self.slot_map.invalidate()
                    raise
            self._selected.update(positions)
            self._ready_inplace_writes += len(positions)
            return self.slot_map.snapshot()

    def update_selected(
        self,
        logical_positions: Sequence[int] | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write fused selected and fresh mandatory rows to their current slots.

        External readers pinning an old canonical source keep its original bytes;
        only that optional mirror is skipped. Pinned current destinations reject
        the update before any writes. Physical guards outlive device completion.
        """
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        with self._mutex:
            self._check_loads()
            self._check_promotions()
            if self._decoded:
                raise RuntimeError("selected updates must precede decode hole reuse")
            mapping = self.slot_map.snapshot()
            if not set(positions).issubset(mapping):
                raise ValueError("updated rows must already have logical mappings")
            if any(mapping[token] in self._shared for token in positions):
                raise ValueError("updated rows cannot mutate shared native slots")
            mirror_candidates = [
                row
                for row, token in enumerate(positions)
                if self._canonical[token] not in self._shared
                and self._canonical[token] != mapping[token]
            ]
            slots = [mapping[token] for token in positions]
            with self.slot_map.write_guard(
                slots,
                optional_slots=[
                    self._canonical[positions[row]] for row in mirror_candidates
                ],
            ) as writable_mirrors:
                mirror_rows = [
                    row
                    for row in mirror_candidates
                    if self._canonical[positions[row]] in writable_mirrors
                ]
                try:
                    self._write_slots(slots, key, value)
                    if mirror_rows:
                        rows = torch.tensor(mirror_rows, device=key.device)
                        self._write_slots(
                            [self._canonical[positions[row]] for row in mirror_rows],
                            key[rows],
                            value[rows],
                        )
                    _completion_future(self.cache.device).result()
                except BaseException:
                    self.slot_map.invalidate()
                    raise
                self._canonical_mirrors += len(mirror_rows)

    def gather(
        self, logical_positions: Sequence[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather logical KV in caller order with atomic mapping pins."""
        positions = _indices(logical_positions, "logical_positions")
        with self._mutex, self.slot_map.pinned_mapping(positions) as mapping:
            self._check_loads(allow_pending=True)
            shape = (len(positions), self.kv_heads, self.head_dim)
            key, value = self.cache.new_empty(shape), self.cache.new_empty(shape)
            slots = [mapping[token] for token in positions]
            native_rows = [
                row for row, slot in enumerate(slots) if slot < self.native_total_slots
            ]
            extra_rows = [
                row for row, slot in enumerate(slots) if slot >= self.native_total_slots
            ]
            device = self.cache.device
            if native_rows:
                rows = torch.tensor(native_rows, device=device)
                blocks = torch.tensor(
                    [slots[row] // self.block_size for row in native_rows],
                    device=device,
                )
                offsets = torch.tensor(
                    [slots[row] % self.block_size for row in native_rows], device=device
                )
                key[rows] = self.cache[blocks, :, offsets, : self.head_dim]
                value[rows] = self.cache[blocks, :, offsets, self.head_dim :]
            if extra_rows:
                assert self._side_key is not None and self._side_value is not None
                rows = torch.tensor(extra_rows, device=device)
                extra = torch.tensor(
                    [slots[row] - self.native_total_slots for row in extra_rows],
                    device=device,
                )
                key[rows] = self._side_key[extra]
                value[rows] = self._side_value[extra]
            # Do not release source pins before a gather's device reads finish.
            _completion_future(self.cache.device).result()
            return key, value

    def decode(
        self,
        token: int,
        key: torch.Tensor,
        value: torch.Tensor,
        canonical_slot: int | None = None,
    ) -> int:
        """Reuse an eligible native hole, otherwise bind the caller's native slot.

        A new decode row is also mirrored into its own canonical native slot.
        Old selected rows displaced by hole reuse remain available only through
        this map, so subsequent attention must continue using ``gather``.
        Neither the new token mapping nor its destinations are readable until
        all device writes complete. A failed partial copy retires this arena.
        """
        if type(token) is not int or token < 0:
            raise ValueError("decode token must be a nonnegative integer")
        if key.ndim == 2:
            key = key.unsqueeze(0)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        self._records([token], key, value)
        with self._mutex:
            self._check_loads(allow_pending=True)
            self._check_promotions()
            if token != len(self._canonical):
                raise ValueError("decode must append the next logical token")
            expected = self._canonical_slot(token)
            if canonical_slot is None:
                canonical_slot = expected
            if type(canonical_slot) is not int or canonical_slot != expected:
                raise ValueError(
                    "decode canonical slot is not owned by its block table"
                )
            if canonical_slot in self._canonical.values():
                raise ValueError(
                    "decode native slot aliases an existing canonical token"
                )
            with self.slot_map.decode_write_guard(
                token, canonical_slot, reuse_vacated=True
            ) as chosen:
                self._write_slots([chosen], key, value)
                if chosen != canonical_slot:
                    self._write_slots([canonical_slot], key, value)
                _completion_future(self.cache.device).result()
            self._hole_reuses += int(chosen != canonical_slot)
            self._canonical_mirrors += int(chosen != canonical_slot)
            self._native_binds += int(chosen == canonical_slot)
            self._canonical[token] = canonical_slot
            self._decoded = True
            return chosen

    def snapshot(self) -> dict[int, int]:
        with self._mutex:
            return self.slot_map.snapshot()

    def page_metadata(self) -> tuple[dict[str, int | bool], ...]:
        """Actual mapped physical pages; selected counts follow relocation.

        Counts describe this arena only, never immutable reusable snapshots.
        A dirty page contains selected tokens; this says nothing about whether
        an SSD copy exists or is current. No native block is released here.
        """
        with self._mutex:
            return self.slot_map.page_metadata(self.block_size, self._selected)

    def spill_order(self) -> tuple[int, ...]:
        """Advisory clean-first, descending-selected-count physical page order.

        Shared, pinned, and in-flight pages are excluded. This snapshot neither
        reserves pages nor performs spill/write coalescing/eviction. Native pool
        ownership remains with vLLM; a future residency owner must revalidate
        guards and complete backing writes before retiring any device records.
        """
        candidates = [page for page in self.page_metadata() if page["spill_eligible"]]
        return tuple(
            page["page_id"]
            for page in sorted(
                candidates,
                key=lambda page: (
                    page["dirty"],
                    -page["selected_count"],
                    page["page_id"],
                ),
            )
        )

    def stats(self) -> dict[str, int]:
        with self._mutex:
            return {
                **self.slot_map.stats(),
                "physical_hole_reuse": self._hole_reuses,
                "decode_native_bind": self._native_binds,
                "canonical_mirror_writes": self._canonical_mirrors,
                "sidecar_capacity_slots": self.max_selected,
                "sidecar_allocated_slots": (
                    self.max_selected if self._side_key is not None else 0
                ),
                "selected_ready_inplace_writes": self._ready_inplace_writes,
                "native_pool_capacity_slots": self.native_total_slots,
                "native_pool_capacity_blocks": self.cache.shape[0],
                "native_pool_capacity_delta": 0,
                "native_pool_blocks_released": 0,
            }

    def invalidate(self) -> None:
        with self._mutex:
            self.slot_map.invalidate()
