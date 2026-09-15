"""Per-layer SLIDE over SGLang's separate native NHD K and V buffers.

The shared ReqToTokenPool row remains canonical and is never modified. A bounded
sidecar and LayerSlotMap provide this layer's redirection; neither allocates nor
releases native token-pool capacity. Reads after hole reuse must use this arena.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import torch

from cacheslide_core.slide import LayerSlotMap, SlotWrite
from cacheslide_core.storage import StaleCompletionError


def _indices(values, name):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be an integer vector")
        values = values.detach().cpu().tolist()
    result = list(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain distinct positions")
    return result


def _completion_future(device: torch.device) -> Future:
    result = Future()
    result.set_running_or_notify_cancel()
    if device.type != "cuda":
        result.set_result(None)
        return result
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))

    def finish():
        try:
            event.synchronize()
        except BaseException as error:
            result.set_exception(error)
        else:
            result.set_result(None)

    threading.Thread(target=finish, name="cacheslide-sg-kv-write", daemon=True).start()
    return result


class SGLangTokenKV:
    """One request/layer on real native [slot,KV-head,D] K/V storage.

    ``ownership_check`` must reject a recycled request row, generation, or pool
    backing. It runs before operations and after actual device completion, not
    while holding the slot-map mutex. Integration must disable concurrent native
    request recycling and wait for arena retirement before releasing pool slots.
    """

    def __init__(
        self,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
        slot_mapping: Sequence[int] | torch.Tensor,
        prompt_length: int,
        max_selected: int,
        *,
        page_size: int = 1,
        request_id: str = "",
        layer_index: int = 0,
        ownership_check: Callable[[], None] | None = None,
    ):
        if (
            not isinstance(key_buffer, torch.Tensor)
            or not isinstance(value_buffer, torch.Tensor)
            or key_buffer.ndim != 3
            or value_buffer.shape != key_buffer.shape
            or min(key_buffer.shape) < 1
            or key_buffer.dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or value_buffer.dtype != key_buffer.dtype
            or value_buffer.device != key_buffer.device
            or key_buffer.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("native K/V must be matching floating [slot,KV-head,D]")
        if type(prompt_length) is not int or prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if type(max_selected) is not int or max_selected < 0:
            raise ValueError("max_selected must be nonnegative")
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be positive")
        self.key_buffer, self.value_buffer = key_buffer, value_buffer
        self.prompt_length, self.max_selected = prompt_length, max_selected
        self.page_size, self.request_id, self.layer_index = (
            page_size,
            request_id,
            layer_index,
        )
        self.native_total_slots, self.kv_heads, self.head_dim = key_buffer.shape
        self._ownership_check = ownership_check
        self._mutex = threading.RLock()
        self._slots = self._validate_slots(slot_mapping)
        if len(self._slots) < prompt_length:
            raise ValueError("canonical mapping does not cover prompt")
        self._canonical = dict(enumerate(self._slots[:prompt_length]))
        self.slot_map = LayerSlotMap(
            self._canonical,
            allocation_floor=self.native_total_slots,
            max_slots=self.native_total_slots + max_selected,
        )
        self._side_key = self._side_value = None
        self._baseline_ready: set[int] = set()
        self._selected: set[int] = set()
        self._loads: list[Future] = []
        self._promotions: list[Future] = []
        self._device_operations: list[Future] = []
        self._decoded = False
        self._hole_reuses = self._mirrors = self._native_binds = self._ready_writes = 0
        self._check_owner()

    def _validate_slots(self, slots):
        slots = _indices(slots, "native slot mapping")
        if any(
            slot < self.page_size or slot >= self.native_total_slots for slot in slots
        ):
            raise ValueError(
                "native slots must exclude the padding page and fit the pool"
            )
        return tuple(slots)

    def _check_owner(self):
        self.slot_map.snapshot()
        if self._ownership_check is not None:
            try:
                self._ownership_check()
            except BaseException:
                self.slot_map.invalidate()
                raise

    def _completion(self):
        completed, checked = _completion_future(self.key_buffer.device), Future()
        self._device_operations = [
            future for future in self._device_operations if not future.done()
        ]
        self._device_operations.append(completed)
        checked.set_running_or_notify_cancel()

        def finish(future):
            try:
                future.result()
                self._check_owner()
            except BaseException as error:
                self.slot_map.invalidate()
                checked.set_exception(error)
            else:
                checked.set_result(None)

        completed.add_done_callback(finish)
        return checked

    def update_slot_mapping(self, slot_mapping):
        slots = self._validate_slots(slot_mapping)
        with self._mutex:
            self._check_owner()
            if any(
                token >= len(slots) or slots[token] != slot
                for token, slot in self._canonical.items()
            ):
                raise StaleCompletionError(
                    "native canonical mapping moved existing tokens"
                )
            self._slots = slots

    def canonical_slot(self, token):
        with self._mutex:
            self._check_owner()
            if type(token) is not int or not 0 <= token < len(self._slots):
                raise ValueError("logical token has no caller-owned native slot")
            return self._slots[token]

    def _records(self, positions, key, value):
        for tensor in (key, value):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (len(positions), self.kv_heads, self.head_dim)
                or tensor.dtype != self.key_buffer.dtype
                or tensor.device != self.key_buffer.device
            ):
                raise ValueError("K/V must match native token shape, dtype and device")

    def _check_loads(self, *, allow_pending=False):
        for future in self._loads:
            if not future.done():
                if allow_pending:
                    continue
                raise RuntimeError("source KV load must complete before native writes")
            future.result()

    def _check_promotions(self):
        for future in self._promotions:
            if not future.done():
                raise RuntimeError("selected writes must complete first")
            future.result()

    def _write_slots(self, slots, key, value):
        device = self.key_buffer.device
        for sidecar in (False, True):
            rows = [
                i
                for i, slot in enumerate(slots)
                if (slot >= self.native_total_slots) == sidecar
            ]
            if not rows:
                continue
            if sidecar and self._side_key is None:
                shape = (self.max_selected, self.kv_heads, self.head_dim)
                self._side_key = self.key_buffer.new_empty(shape)
                self._side_value = self.value_buffer.new_empty(shape)
            destinations = torch.tensor(
                [slots[i] - (self.native_total_slots if sidecar else 0) for i in rows],
                device=device,
            )
            source = torch.tensor(rows, device=device)
            target_k = self._side_key if sidecar else self.key_buffer
            target_v = self._side_value if sidecar else self.value_buffer
            target_k.index_copy_(0, destinations, key[source])
            target_v.index_copy_(0, destinations, value[source])

    def write(self, logical_positions, key, value):
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        with self._mutex:
            self._check_owner()
            self._check_loads()
            if self._decoded:
                raise RuntimeError("baseline writes cannot follow decode")
            if not set(positions).issubset(self._canonical):
                raise ValueError("baseline rows require existing canonical tokens")
            with self.slot_map.write_guard([self._canonical[p] for p in positions]):
                try:
                    self._write_slots(
                        [self._canonical[p] for p in positions], key, value
                    )
                    self._completion().result()
                except BaseException:
                    self.slot_map.invalidate()
                    raise
            self._baseline_ready.update(positions)

    load_baseline = write

    def write_prefill(self, key, value):
        self.write(range(self.prompt_length), key, value)

    def promote_selected(self, logical_positions, key, value, *, load_future=None):
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        if load_future is not None and not isinstance(load_future, Future):
            raise TypeError("load_future must be a concurrent Future")
        with self._mutex:
            self._check_owner()
            if self._decoded or self._selected.intersection(positions):
                raise ValueError("promotion needs new selected tokens before decode")

            def writer(runs: list[SlotWrite]):
                slots, rows = [], []
                for run in runs:
                    for offset in range(run.slot_count):
                        slots.append(run.start_slot + offset)
                        rows.append(
                            int.from_bytes(
                                run.payload[offset * 8 : (offset + 1) * 8], "little"
                            )
                        )
                indices = torch.tensor(rows, device=key.device, dtype=torch.long)
                self._write_slots(slots, key[indices], value[indices])
                return self._completion()

            promotion = self.slot_map.promote_selected(
                {
                    token: row.to_bytes(8, "little")
                    for row, token in enumerate(positions)
                },
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

    def write_selected_ready(self, logical_positions, key, value):
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        with self._mutex:
            self._check_owner()
            self._check_loads()
            self._check_promotions()
            mapping = self.slot_map.snapshot()
            if self._decoded:
                raise RuntimeError("ready selected writes must precede decode")
            if len(self._baseline_ready) != self.prompt_length:
                raise RuntimeError("native baseline device writes must complete first")
            if not set(positions).issubset(self._canonical):
                raise ValueError("ready rows require canonical tokens")
            if self._selected.intersection(positions) or any(
                mapping[p] != self._canonical[p] for p in positions
            ):
                raise ValueError("already selected tokens must use update_selected")
            with self.slot_map.write_guard([mapping[p] for p in positions]):
                try:
                    self._write_slots([mapping[p] for p in positions], key, value)
                    self._completion().result()
                except BaseException:
                    self.slot_map.invalidate()
                    raise
            self._selected.update(positions)
            self._ready_writes += len(positions)
            return mapping

    def update_selected(self, logical_positions, key, value):
        positions = _indices(logical_positions, "logical_positions")
        self._records(positions, key, value)
        with self._mutex:
            self._check_owner()
            self._check_loads()
            self._check_promotions()
            if self._decoded:
                raise RuntimeError("selected updates must precede decode")
            mapping = self.slot_map.snapshot()
            if not set(positions).issubset(mapping):
                raise ValueError("updated rows require current token mappings")
            candidates = [
                i for i, p in enumerate(positions) if mapping[p] != self._canonical[p]
            ]
            with self.slot_map.write_guard(
                [mapping[p] for p in positions],
                optional_slots=[self._canonical[positions[i]] for i in candidates],
            ) as mirrors:
                try:
                    self._write_slots([mapping[p] for p in positions], key, value)
                    rows = [
                        i
                        for i in candidates
                        if self._canonical[positions[i]] in mirrors
                    ]
                    if rows:
                        indices = torch.tensor(rows, device=key.device)
                        self._write_slots(
                            [self._canonical[positions[i]] for i in rows],
                            key[indices],
                            value[indices],
                        )
                    self._completion().result()
                except BaseException:
                    self.slot_map.invalidate()
                    raise
                self._mirrors += len(rows)

    def gather(self, logical_positions):
        positions = _indices(logical_positions, "logical_positions")
        with self._mutex:
            self._check_owner()
            self._check_loads(allow_pending=True)
            with self.slot_map.pinned_mapping(positions) as mapping:
                shape = (len(positions), self.kv_heads, self.head_dim)
                key = self.key_buffer.new_empty(shape)
                value = self.value_buffer.new_empty(shape)
                for sidecar in (False, True):
                    rows = [
                        i
                        for i, p in enumerate(positions)
                        if (mapping[p] >= self.native_total_slots) == sidecar
                    ]
                    if not rows:
                        continue
                    source = torch.tensor(
                        [
                            mapping[positions[i]]
                            - (self.native_total_slots if sidecar else 0)
                            for i in rows
                        ],
                        device=key.device,
                    )
                    target = torch.tensor(rows, device=key.device)
                    source_k = self._side_key if sidecar else self.key_buffer
                    source_v = self._side_value if sidecar else self.value_buffer
                    key[target], value[target] = source_k[source], source_v[source]
                self._completion().result()
                return key, value

    def decode(self, token, key, value, canonical_slot=None):
        if key.ndim == 2:
            key = key.unsqueeze(0)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        self._records([token], key, value)
        with self._mutex:
            self._check_owner()
            self._check_loads(allow_pending=True)
            self._check_promotions()
            if type(token) is not int or token != len(self._canonical):
                raise ValueError("decode must append the next logical token")
            expected = self.canonical_slot(token)
            if canonical_slot is not None and (
                type(canonical_slot) is not int or canonical_slot != expected
            ):
                raise ValueError("decode slot is not owned by the canonical mapping")
            with self.slot_map.decode_write_guard(
                token, expected, reuse_vacated=True
            ) as slot:
                self._write_slots([slot], key, value)
                if slot != expected:
                    self._write_slots([expected], key, value)
                self._completion().result()
            self._canonical[token] = expected
            self._decoded = True
            self._hole_reuses += int(slot != expected)
            self._mirrors += int(slot != expected)
            self._native_binds += int(slot == expected)
            return slot

    def snapshot(self):
        with self._mutex:
            self._check_owner()
            return self.slot_map.snapshot()

    def page_metadata(self):
        with self._mutex:
            self._check_owner()
            return self.slot_map.page_metadata(self.page_size, self._selected)

    def spill_order(self):
        pages = [page for page in self.page_metadata() if page["spill_eligible"]]
        return tuple(
            page["page_id"]
            for page in sorted(
                pages, key=lambda p: (p["dirty"], -p["selected_count"], p["page_id"])
            )
        )

    def stats(self):
        return {
            **self.slot_map.stats(),
            "physical_hole_reuse": self._hole_reuses,
            "decode_native_bind": self._native_binds,
            "canonical_mirror_writes": self._mirrors,
            "sidecar_capacity_slots": self.max_selected,
            "sidecar_allocated_slots": self.max_selected
            if self._side_key is not None
            else 0,
            "selected_ready_inplace_writes": self._ready_writes,
            "native_pool_capacity_slots": self.native_total_slots,
            "native_pool_capacity_delta": 0,
            "native_pool_blocks_released": 0,
        }

    def invalidate(self):
        with self._mutex:
            self.slot_map.invalidate()
            pending = tuple(self._device_operations)
        # Retire before draining so no late selected publication can succeed.
        # Underlying device Futures do not depend on the arena mutex or a source
        # load callback; wait outside the mutex before native slots are released.
        for future in pending:
            try:
                future.result()
            except BaseException:
                pass
