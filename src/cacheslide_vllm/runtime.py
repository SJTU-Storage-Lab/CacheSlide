"""Selective, request-local CacheSlide execution shared by CPU and native vLLM.

The immutable host cache stores only fixed-token K/V and per-layer input states.
Hidden states are necessary for tokens promoted by the four-layer WCA gate.
All native output rows are preserved; only the last prompt row is sampled.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from torch import Tensor

from .artifacts import AdapterBundle
from .config import CacheSlideSettings
from .contracts import RequestPlan, digest
from .integration import StepContext
from .position import ChunkIdentity, cope_attention
from .profiles import ProfileBundle
from .storage import CacheCapacityError, CacheIntegrityError, TieredPageStore
from .wca import WCAState

logger = logging.getLogger(__name__)


class ReuseFailure(RuntimeError):
    """A reuse-specific failure that permits full-prefill retry before sampling."""


@dataclass
class RequestState:
    plan: RequestPlan
    reuse: bool
    profiles: dict[int, Any]
    reason: str
    prefill_tokens: int = 0
    wca: WCAState | None = None
    kv_future: Future | None = None
    kv_future_layer: int = -1
    dense_kv: dict[int, tuple[Tensor, Tensor]] = field(default_factory=dict)
    arenas: dict[int, Any] = field(default_factory=dict)
    created_keys: list[str] = field(default_factory=list)
    write_cache: bool = False
    selected_at_layer: Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class CacheSlideRuntime:
    """One single-request engine, with a bounded and restartable host page store.

    ``arena_factory(layer, prompt_length, max_selected)`` is optional for CPU
    reference tests. Native vLLM supplies real block-table-backed page arenas.
    """

    def __init__(
        self,
        settings: CacheSlideSettings,
        bundle: AdapterBundle,
        *,
        arena_factory=None,
        model_dtype: torch.dtype | None = None,
    ):
        if model_dtype is not None and not isinstance(model_dtype, torch.dtype):
            raise ValueError("model_dtype must be a torch dtype or None")
        self.settings = settings
        self.bundle = bundle
        self.model_dtype = model_dtype
        self.layers = bundle.config["num_hidden_layers"]
        if settings.calibration_layer >= self.layers:
            raise ValueError("calibration_layer is outside the model")
        self.profiles = (
            ProfileBundle(
                settings.profile_path, bundle.identity, settings.max_profile_elements
            )
            if settings.profile_path
            else None
        )
        self.identity = digest(
            {
                "adapter": bundle.identity,
                "profiles": self.profiles.identity if self.profiles else None,
                "format": "cacheslide-runtime-v1",
                "model_dtype": str(model_dtype) if model_dtype is not None else None,
            }
        )
        self.store = TieredPageStore(
            settings.cache_root,
            cpu_budget_bytes=settings.cpu_budget_bytes,
            disk_budget_bytes=settings.disk_budget_bytes,
        )
        self.arena_factory = arena_factory
        self.requests: dict[str, RequestState] = {}
        self.active: RequestState | None = None
        self.last_metrics: dict = {}

    def _key(self, plan: RequestPlan, layer: int, kind: str) -> str:
        return plan.cache_key(self.identity, layer) + "/" + kind

    def _commit_key(self, plan: RequestPlan) -> str:
        return self._key(plan, 0, "committed")

    def _preflight(self, plan: RequestPlan) -> tuple[bool, dict, str]:
        profiles = {}
        if self.profiles is not None:
            for layer in range(self.layers):
                profile = self.profiles.get(plan, layer)
                if profile is None:
                    return False, {}, "CCPE profile miss"
                profiles[layer] = profile
        else:
            return False, {}, "no calibrated CCPE profiles"
        if not plan.fixed_indices:
            return False, profiles, "no reusable tokens"
        keys = [self._commit_key(plan)] + [
            self._key(plan, layer, kind)
            for layer in range(self.layers)
            for kind in ("kv", "state")
        ]
        if not all(self.store.contains(key) for key in keys):
            return False, profiles, "immutable cache miss"
        return True, profiles, "hit"

    def _discard_uncommitted(self, state: RequestState) -> None:
        remaining = []
        for key in reversed(state.created_keys):
            try:
                if self.store.contains(key):
                    self.store.delete(key)
            except (OSError, RuntimeError) as exc:
                # A cache cleanup failure must not prevent full recomputation.
                logger.warning("CacheSlide deferred uncommitted page cleanup: %s", exc)
                remaining.append(key)
        state.created_keys[:] = reversed(remaining)

    def _recover_population(self, state: RequestState) -> None:
        """Remove only known layer pages of this exact uncommitted identity."""
        if self.store.contains(self._commit_key(state.plan)):
            raise ReuseFailure("incomplete committed population requires cache repair")
        removed = 0
        try:
            for layer in range(self.layers):
                for kind in ("state", "kv"):
                    key = self._key(state.plan, layer, kind)
                    if self.store.contains(key):
                        self.store.delete(key)
                        removed += 1
        except (OSError, RuntimeError) as exc:
            raise ReuseFailure(
                "uncommitted population recovery is unavailable"
            ) from exc
        state.metrics["recovered_orphan_pages"] = removed

    def release(self, request_id: str) -> None:
        state = self.requests.pop(request_id, None)
        if state is None:
            return
        if state.kv_future is not None:
            state.kv_future.cancel()
            # Drain before deleting pages, so late I/O cannot resurrect a page.
            if not state.kv_future.cancelled():
                try:
                    state.kv_future.result()
                except Exception:
                    logger.exception("CacheSlide pending load failed during cleanup")
        self._record_metrics(state)
        for arena in state.arenas.values():
            arena.invalidate()
        if state.created_keys:
            self._discard_uncommitted(state)
        self.last_metrics = dict(state.metrics)

    def _record_metrics(self, state: RequestState) -> None:
        state.metrics["paged_layers"] = {
            str(layer): arena.stats() for layer, arena in state.arenas.items()
        }
        self.last_metrics = dict(state.metrics)

    def close(self) -> None:
        for request_id in tuple(self.requests):
            self.release(request_id)
        self.store.close()

    def _read(
        self, state: RequestState, layer: int, kind: str, device, dtype
    ) -> dict[str, Tensor]:
        key = self._key(state.plan, layer, kind)
        try:
            if kind == "kv" and state.kv_future_layer == layer:
                future, state.kv_future = state.kv_future, None
                state.kv_future_layer = -1
                data = future.result()
            else:
                data = self.store.read(key)
            tensors = load_tensors(data)
            expected = {"k", "v"} if kind == "kv" else {"hidden", "residual"}
            if set(tensors) != expected:
                raise ValueError("snapshot fields differ from the runtime schema")
            count = len(state.plan.fixed_indices)
            for name, tensor in tensors.items():
                shape = (
                    (
                        count,
                        self.bundle.config["num_key_value_heads"],
                        self.bundle.config["head_dim"],
                    )
                    if kind == "kv"
                    else (count, self.bundle.config["hidden_size"])
                )
                if name == "residual" and layer == 0:
                    shape = (0,)
                if tuple(tensor.shape) != shape or not tensor.is_floating_point():
                    raise ValueError("snapshot tensor geometry is incompatible")
                if not torch.isfinite(tensor).all():
                    raise ValueError("snapshot tensor is nonfinite")
            return {
                name: tensor.to(device=device, dtype=dtype)
                for name, tensor in tensors.items()
            }
        except (ValueError, KeyError, OSError, RuntimeError, SafetensorError) as exc:
            raise ReuseFailure(f"invalid {kind} snapshot at layer {layer}") from exc

    def _put(
        self, state: RequestState, layer: int, kind: str, tensors: dict[str, Tensor]
    ) -> None:
        if not state.write_cache:
            return
        key = self._key(state.plan, layer, kind)
        try:
            if self.store.contains(key):
                raise ReuseFailure("population collides with immutable pages")
            data = save_tensors(
                {
                    name: tensor.detach().contiguous().cpu()
                    for name, tensor in tensors.items()
                }
            )
            self.store.put(key, data, selected_count=0)
            state.created_keys.append(key)
            # Persist each layer rather than accumulating the model in host RAM.
            self.store.submit_spill(key).result()
        except (ValueError, OSError, RuntimeError, SafetensorError) as exc:
            raise ReuseFailure(
                f"cannot persist {kind} snapshot at layer {layer}"
            ) from exc
        state.metrics["snapshot_bytes_written"] += len(data)

    def _commit_population(self, state: RequestState) -> None:
        key = self._commit_key(state.plan)
        try:
            self.store.put(key, b"cacheslide-runtime-v1", selected_count=0)
            state.created_keys.append(key)
            self.store.submit_spill(key).result()
        except (ValueError, OSError, RuntimeError) as exc:
            raise ReuseFailure("cannot persist population commit marker") from exc
        state.created_keys.clear()
        state.write_cache = False

    def _prefetch(self, state: RequestState, layer: int) -> None:
        if state.reuse and layer < self.layers:
            if state.kv_future is not None:
                raise RuntimeError("more than one outstanding KV prefetch")
            state.kv_future = self.store.submit_load(self._key(state.plan, layer, "kv"))
            state.kv_future_layer = layer

    @torch.inference_mode()
    def run(
        self, model, hidden: Tensor, positions: Tensor, step: StepContext
    ) -> Tensor:
        if self.active is not None:
            raise RuntimeError("CacheSlide does not allow concurrent model forwards")
        metadata = (step.extra_args or {}).get("cacheslide")
        if metadata is None:
            raise ValueError("each CacheSlide request requires extra_args.cacheslide")
        plan = RequestPlan.parse(metadata, step.prompt_token_ids)
        n = len(plan.token_ids)
        replay_ids = step.replay_token_ids
        prefill_tokens = len(replay_ids) if replay_ids is not None else n
        if prefill_tokens > self.settings.max_prompt_tokens:
            raise ValueError("prompt exceeds the configured CacheSlide token budget")
        if (
            len(positions) != len(step.positions)
            or tuple(positions.tolist()) != step.positions
        ):
            raise ValueError("native rows and CacheSlide positions disagree")
        prefill = step.positions == tuple(range(prefill_tokens))
        if prefill:
            # Preemption restarts from an uncached full prompt; discard old maps.
            self.release(step.request_id)
            hit, profiles, reason = self._preflight(plan)
            if plan.operation == "calibrate":
                raise ValueError("use the offline calibrate command, not generation")
            # After preemption vLLM replays prompt + already generated tokens.
            # Rebuild their exact causal state densely; do not advertise a cache
            # hit or publish a new approximate snapshot under the original key.
            replaying = replay_ids is not None and prefill_tokens > n
            reuse = plan.operation == "reuse" and hit and not replaying
            if replaying:
                reason = "preemption replay requires full recomputation"
            state = RequestState(
                plan, reuse, profiles, reason, prefill_tokens=prefill_tokens
            )
            state.write_cache = (
                plan.operation == "populate"
                and not hit
                and bool(profiles)
                and not replaying
            )
            state.metrics = {
                "request_id": step.request_id,
                "operation": plan.operation,
                "cache_hit": reuse,
                "reason": reason,
                "prompt_tokens": n,
                "layer_rows": [],
                "snapshot_bytes_written": 0,
                "fallback": replaying and plan.operation == "reuse",
                "decode_tokens": prefill_tokens - n,
                "replayed_tokens": prefill_tokens - n,
                "restored_rows": 0,
                "recovered_orphan_pages": 0,
            }
            self.requests[step.request_id] = state
        else:
            state = self.requests.get(step.request_id)
            if state is None or state.plan != plan:
                raise ValueError(
                    "decode requires the unchanged populated request state"
                )
            expected = n + state.metrics["decode_tokens"]
            if step.positions != (expected,):
                raise ValueError("decode tokens must arrive once in consecutive order")
        self.active = state
        try:
            if prefill:
                try:
                    output = self._prefill(model, hidden, positions, state)
                except (ReuseFailure, CacheCapacityError, CacheIntegrityError) as exc:
                    # No token has been sampled. Recompute every layer from input.
                    logger.warning("CacheSlide full-prefill fallback: %s", exc)
                    if state.kv_future is not None:
                        try:
                            state.kv_future.result()
                        except Exception:
                            pass
                        state.kv_future = None
                        state.kv_future_layer = -1
                    for arena in state.arenas.values():
                        arena.invalidate()
                    state.arenas.clear()
                    state.dense_kv.clear()
                    self._discard_uncommitted(state)
                    state.reuse = state.write_cache = False
                    state.wca = None
                    state.selected_at_layer = None
                    state.metrics.update(
                        cache_hit=False,
                        fallback=True,
                        reason=str(exc),
                        layer_rows=[],
                        restored_rows=0,
                    )
                    output = self._prefill(model, hidden, positions, state)
                self.last_metrics = dict(state.metrics)
                logger.info("CacheSlide prefill metrics: %s", state.metrics)
                return output
            residual = None
            for layer in model.layers:
                hidden, residual = layer(positions, hidden, residual)
            hidden, _ = model.norm(hidden, residual)
            state.metrics["decode_tokens"] += 1
            self._record_metrics(state)
            return hidden
        finally:
            self.active = None

    def _prefill(
        self, model, hidden: Tensor, positions: Tensor, state: RequestState
    ) -> Tensor:
        # Native fused RMSNorm mutates the residual; preserve retry embeddings.
        hidden = hidden.clone()
        if state.write_cache:
            self._recover_population(state)
        n, device = len(positions), hidden.device
        residual = None
        previous = torch.arange(n, device=device)
        fixed = torch.tensor(state.plan.fixed_indices, device=device, dtype=torch.long)
        self._prefetch(state, 0)
        for index, layer in enumerate(model.layers):
            active = state.wca.active_indices if state.wca is not None else previous
            if state.wca is not None:
                # Gather continuing rows; restore newly selected rows from this
                # layer's input snapshot, not a zero/stale previous-layer output.
                lookup = torch.full((n,), -1, device=device, dtype=torch.long)
                lookup[previous] = torch.arange(len(previous), device=device)
                rows = lookup[active]
                continuing = rows >= 0
                next_hidden = hidden.new_empty((len(active), hidden.shape[-1]))
                next_residual = torch.empty_like(next_hidden)
                next_hidden[continuing] = hidden[rows[continuing]]
                next_residual[continuing] = residual[rows[continuing]]
                if not continuing.all():
                    snapshot = self._read(state, index, "state", device, hidden.dtype)
                    ordinal = torch.full((n,), -1, device=device, dtype=torch.long)
                    ordinal[fixed] = torch.arange(len(fixed), device=device)
                    restored = ordinal[active[~continuing]]
                    if (restored < 0).any():
                        raise ReuseFailure("a mandatory dynamic row was lost")
                    next_hidden[~continuing] = snapshot["hidden"][restored]
                    next_residual[~continuing] = snapshot["residual"][restored]
                    state.metrics["restored_rows"] += int((~continuing).sum().item())
                hidden, residual = next_hidden, next_residual
            if state.write_cache:
                self._put(
                    state,
                    index,
                    "state",
                    {
                        "hidden": hidden[fixed],
                        "residual": (
                            residual[fixed]
                            if residual is not None
                            else hidden.new_empty((0,))
                        ),
                    },
                )
            state.metrics["layer_rows"].append(len(active))
            state.selected_at_layer = (
                state.wca.selected_indices.clone() if state.wca is not None else None
            )
            if state.selected_at_layer is not None:
                state.selected_at_layer = state.selected_at_layer[
                    ~state.wca.mandatory_mask[state.selected_at_layer]
                ]
            hidden, residual = layer(positions[active], hidden, residual)
            if not torch.isfinite(hidden).all() or not torch.isfinite(residual).all():
                if state.reuse:
                    raise ReuseFailure("WCA produced nonfinite layer output")
                raise ValueError("full CoPE recomputation produced nonfinite output")
            previous = active
        hidden, _ = model.norm(hidden, residual)
        # Native V1 sampling indexes the last row of the original scheduled batch.
        # Uncomputed rows are placeholders, never advertised as prompt logprobs.
        output = hidden.new_zeros((n, hidden.shape[-1]))
        output[previous] = hidden
        if state.write_cache:
            self._commit_population(state)
        state.metrics["computed_token_layers"] = sum(state.metrics["layer_rows"])
        state.metrics["dense_token_layers"] = n * self.layers
        state.metrics["storage"] = self.store.stats()
        self._record_metrics(state)
        return output

    def attention(
        self, layer: int, positions: Tensor, q: Tensor, k: Tensor, v: Tensor, adapter
    ) -> Tensor:
        state = self.active
        if state is None:
            # Native memory profiling: no request means no durable cache or slots.
            return cope_attention(
                q,
                k,
                v,
                adapter.cope,
                positions,
                key_positions=positions,
                query_chunk_size=self.settings.query_chunk_size,
            )
        n = state.prefill_tokens
        if int(positions[0]) >= n:
            if self.arena_factory is None:
                old_k, old_v = state.dense_kv[layer]
                full_k, full_v = torch.cat((old_k, k)), torch.cat((old_v, v))
                state.dense_kv[layer] = (full_k, full_v)
            else:
                arena = state.arenas[layer]
                # Native block tables may expand as the scheduler allocates decode.
                self.arena_factory(layer, n, 0, existing=arena)
                arena.decode(
                    int(positions[0]),
                    k[0],
                    v[0],
                    arena.canonical_slot(int(positions[0])),
                )
                full_k, full_v = arena.gather(
                    torch.arange(int(positions[0]) + 1, device=k.device)
                )
            return self._attend(state, layer, q, full_k, full_v, positions, adapter)
        fixed = torch.tensor(
            state.plan.fixed_indices, device=k.device, dtype=torch.long
        )
        sparse = state.reuse and state.wca is not None
        selected = positions[torch.isin(positions, fixed)] if sparse else positions[:0]
        arena = None
        if self.arena_factory is not None:
            arena = self.arena_factory(layer, n, len(selected))
            state.arenas[layer] = arena
            if sparse and len(selected):
                mask = torch.isin(positions, fixed)
                # Stage fresh selected rows while the baseline host read is pending.
                arena.promote_selected(
                    selected, k[mask], v[mask], load_future=state.kv_future
                ).result()
        cached_k = k.new_zeros((n, *k.shape[1:]))
        cached_v = torch.zeros_like(cached_k)
        if state.reuse:
            cache = self._read(state, layer, "kv", k.device, k.dtype)
            cached_k[fixed], cached_v[fixed] = cache["k"], cache["v"]
            # Start the next layer's read while current-layer attention executes.
            self._prefetch(state, layer + 1)
        if sparse:
            update = state.wca.update(
                layer + 1, cached_k, cached_v, k, v, computed_indices=positions
            )
            full_k, full_v = cached_k.clone(), cached_v.clone()
            full_k[positions], full_v[positions] = update.fused_k, update.fused_v
            if arena is not None:
                arena.write_prefill(cached_k, cached_v)
                arena.update_selected(positions, update.fused_k, update.fused_v)
        else:
            full_k, full_v = k, v
            if state.reuse and layer == self.settings.calibration_layer:
                reused = torch.zeros(n, device=k.device, dtype=torch.bool)
                mandatory = torch.zeros_like(reused)
                reused[fixed] = True
                mandatory[list(state.plan.mandatory_indices)] = True
                # Dynamic rows compare to themselves; only fixed error selects work.
                cached_k[~reused] = k[~reused]
                state.wca = WCAState.initialize(
                    cached_k, k, reused, mandatory, self.settings.wca_config()
                )
                state.wca.last_layer = layer + 1
            if arena is not None:
                arena.write_prefill(full_k, full_v)
        if state.write_cache:
            self._put(state, layer, "kv", {"k": full_k[fixed], "v": full_v[fixed]})
        if arena is None:
            state.dense_kv[layer] = (full_k, full_v)
        else:
            full_k, full_v = arena.gather(torch.arange(n, device=k.device))
        return self._attend(state, layer, q, full_k, full_v, positions, adapter)

    def _attend(self, state, layer, q, k, v, positions, adapter) -> Tensor:
        """Query-bounded CoPE with canonical fixed/fixed profile substitution.

        Causal visibility controls the contextual gates. WCA's updated-and-self
        policy is a later association mask: it intentionally does not change
        CCPE coordinates as the selected-token set changes between layers.
        """
        profile = state.profiles.get(layer)
        fixed = torch.tensor(
            state.plan.fixed_indices, device=q.device, dtype=torch.long
        )
        ordinal = {position: i for i, position in enumerate(state.plan.fixed_indices)}
        keys = torch.arange(len(k), device=q.device)
        output = []
        group = q.shape[1] // k.shape[1]
        expanded_k = k.repeat_interleave(group, 1).float()
        expanded_v = v.repeat_interleave(group, 1).float()
        chunks = tuple(ChunkIdentity(*entry) for entry in state.plan.fixed_layout)
        for start in range(0, len(q), self.settings.query_chunk_size):
            query = q[start : start + self.settings.query_chunk_size]
            pos = positions[start : start + len(query)]
            logits = torch.einsum("qhd,khd->hqk", query.float(), expanded_k)
            logits *= q.shape[-1] ** -0.5
            allowed = (keys[None, :] <= pos[:, None])[None].expand(q.shape[1], -1, -1)
            contextual = adapter.cope.contextual_positions(logits, allowed)
            rows = [i for i, p in enumerate(pos.tolist()) if p in ordinal]
            if profile is not None and rows:
                ordinals = torch.tensor([ordinal[int(pos[i])] for i in rows])
                canonical = profile.lookup(
                    chunks,
                    ordinals,
                    checkpoint_id=self.bundle.identity,
                    trained_profile_version=profile.trained_profile_version,
                    device=q.device,
                    dtype=contextual.dtype,
                )
                for index, row in enumerate(rows):
                    contextual[:, row, fixed] = canonical[:, index]
            logits = logits + adapter.cope.positional_bias(query, contextual)
            if (
                state.selected_at_layer is not None
                and self.settings.selected_attention == "updated_and_self"
            ):
                selected_query = torch.isin(pos, state.selected_at_layer)
                updated_key = ~torch.isin(keys, fixed)
                visibility = (
                    ~selected_query[:, None]
                    | updated_key[None, :]
                    | (pos[:, None] == keys[None, :])
                )
                allowed = allowed & visibility[None]
            if not torch.isfinite(logits).all():
                raise ReuseFailure("nonfinite contextual attention logits")
            probability = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
            output.append(
                torch.einsum("hqk,khd->qhd", probability, expanded_v).to(v.dtype)
            )
        return torch.cat(output)
