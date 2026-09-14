"""Portable contextual position encoding and explicit CCPE calibration.

CoPE follows the contextual gates and interpolated learned embeddings in
arXiv:2405.18719 section 4. It requires trained learned embeddings; replacing
RoPE with a newly initialized instance does not preserve a model's behavior.

CacheSlide does not fully specify its task-histogram calibration procedure.
The CCPE implementation here explicitly chooses a joint quantized-pattern
histogram over genuine contextual position traces. This is a reproducible
profile format, not a claim of access to the authors' trained task profiles.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _rows(name: str, tensor: Tensor) -> None:
    if (
        tensor.ndim != 3
        or not tensor.is_floating_point()
        or min(tensor.shape[1:]) < 1
        or not torch.isfinite(tensor).all()
    ):
        raise ValueError(f"{name} must be finite floating [token, head, dimension]")


def _positions(name: str, tensor: Tensor, n: int, device: torch.device) -> None:
    if (
        tensor.ndim != 1
        or tensor.shape[0] != n
        or tensor.dtype != torch.long
        or tensor.device != device
        or (tensor < 0).any()
    ):
        raise ValueError(f"{name} must be nonnegative int64 [{n}] on {device}")
    if torch.unique(tensor).numel() != n:
        raise ValueError(f"{name} must identify distinct token positions")


def _qk_layout(query: Tensor, key: Tensor) -> None:
    _rows("query", query)
    _rows("key", key)
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("Q/K dtype and device must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K head dimensions must match")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be a multiple of KV heads for GQA")
    if key.shape[0] == 0:
        raise ValueError("attention requires at least one key")


def _accumulation_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _logits(query: Tensor, key: Tensor) -> Tensor:
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    dtype = _accumulation_dtype(query)
    logits = torch.einsum("qhd,khd->hqk", query.to(dtype), key.to(dtype))
    logits = logits / math.sqrt(query.shape[-1])
    if not torch.isfinite(logits).all():
        raise ValueError("attention logits overflowed the accumulation dtype")
    return logits


def _allowed(
    query: Tensor,
    key: Tensor,
    query_positions: Tensor,
    key_positions: Tensor | None,
    attention_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    _positions("query_positions", query_positions, query.shape[0], query.device)
    if key_positions is None:
        key_positions = torch.arange(key.shape[0], device=key.device)
    _positions("key_positions", key_positions, key.shape[0], key.device)
    if key_positions.numel() > 1 and not (key_positions[1:] > key_positions[:-1]).all():
        raise ValueError("key_positions must be strictly increasing for reverse cumsum")
    mask = key_positions[None, :] <= query_positions[:, None]
    mask = mask.unsqueeze(0).expand(query.shape[1], -1, -1)
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool or attention_mask.device != query.device:
            raise ValueError("attention_mask must be bool on the query device")
        if attention_mask.shape == (query.shape[0], key.shape[0]):
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.shape not in (
            (1, query.shape[0], key.shape[0]),
            (query.shape[1], query.shape[0], key.shape[0]),
        ):
            raise ValueError("attention_mask must be [query,key] or [head,query,key]")
        mask = mask & attention_mask
    return mask, key_positions


@dataclass(frozen=True)
class CoPEPositionTrace:
    """Detached positions from CoPE gates, retaining layout/checkpoint metadata.

    Construct using ``CoPE.position_trace``. Traces carry masked attention
    logits so calibration can verify that supplied positions follow the actual
    contextual gate formula, rather than accepting arbitrary token offsets.
    """

    positions: Tensor
    masked_logits: Tensor
    allowed_mask: Tensor
    query_positions: Tensor
    key_positions: Tensor
    max_positions: int
    checkpoint_id: str
    source_query_indices: Tensor | None = None
    source_key_indices: Tensor | None = None

    def project(
        self,
        query_indices: Tensor,
        key_indices: Tensor,
        *,
        canonical_positions: bool = True,
    ) -> CoPEPositionTrace:
        """Extract fixed-query × fixed-key positions after all contextual gates.

        Source logits remain intact: dynamic tokens' gates still contributed to
        the extracted positions. Canonical metadata maps fixed keys to compact
        ordinals, allowing later insertion of dynamic tokens between chunks.
        Query positions must correspond to selected fixed keys in canonical mode.
        """
        for name, indices, size in (
            ("query_indices", query_indices, self.positions.shape[1]),
            ("key_indices", key_indices, self.positions.shape[2]),
        ):
            if (
                indices.ndim != 1
                or indices.dtype != torch.long
                or indices.device.type != "cpu"
                or (indices < 0).any()
                or (indices >= size).any()
                or torch.unique(indices).numel() != indices.numel()
            ):
                raise ValueError(
                    f"{name} must contain distinct valid CPU int64 indices"
                )
        query_pos = self.query_positions[query_indices]
        key_pos = self.key_positions[key_indices]
        if key_pos.numel() > 1 and not (key_pos[1:] > key_pos[:-1]).all():
            raise ValueError("projected keys must retain causal token order")
        if canonical_positions:
            mapping = {int(pos): index for index, pos in enumerate(key_pos)}
            if any(int(pos) not in mapping for pos in query_pos):
                raise ValueError(
                    "canonical query positions must belong to projected keys"
                )
            query_pos = torch.tensor([mapping[int(pos)] for pos in query_pos])
            key_pos = torch.arange(key_indices.numel())
        source_queries = self.source_query_indices
        source_keys = self.source_key_indices
        if source_queries is None:
            source_queries = torch.arange(self.positions.shape[1])
        if source_keys is None:
            source_keys = torch.arange(self.positions.shape[2])
        return CoPEPositionTrace(
            positions=self.positions[:, query_indices][:, :, key_indices].clone(),
            masked_logits=self.masked_logits,
            allowed_mask=self.allowed_mask,
            query_positions=query_pos.clone(),
            key_positions=key_pos.clone(),
            max_positions=self.max_positions,
            checkpoint_id=self.checkpoint_id,
            source_query_indices=source_queries[query_indices].clone(),
            source_key_indices=source_keys[key_indices].clone(),
        )


class CoPE(nn.Module):
    """Learned contextual bias shared across heads; embeddings [dim,maxpos]."""

    def __init__(
        self,
        head_dim: int,
        max_positions: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if head_dim < 1 or max_positions < 2:
            raise ValueError("head_dim must be positive and max_positions at least two")
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.position_embeddings = nn.Parameter(
            torch.zeros(head_dim, max_positions, device=device, dtype=dtype)
        )

    def contextual_positions(self, logits: Tensor, allowed_mask: Tensor) -> Tensor:
        """Mask before sigmoid, reverse-cumsum over causal keys, then clamp."""
        if logits.ndim != 3 or not logits.is_floating_point():
            raise ValueError("logits must be floating [head,query,key]")
        if allowed_mask.shape != logits.shape or allowed_mask.dtype != torch.bool:
            raise ValueError("allowed_mask must be bool with the logits shape")
        if allowed_mask.device != logits.device:
            raise ValueError("allowed_mask and logits must share a device")
        if not torch.isfinite(logits[allowed_mask]).all():
            raise ValueError("unmasked logits must be finite")
        masked_logits = logits.masked_fill(~allowed_mask, -torch.inf)
        gates = masked_logits.sigmoid()
        return gates.flip(-1).cumsum(-1).flip(-1).clamp(0, self.max_positions - 1)

    def positional_bias(self, query: Tensor, positions: Tensor) -> Tensor:
        """Interpolate q·learned_embedding between floor/ceil contextual positions."""
        _rows("query", query)
        if query.shape[-1] != self.head_dim:
            raise ValueError("query dimension does not match the learned embeddings")
        if query.device != self.position_embeddings.device:
            raise ValueError("query and learned embeddings must share a device")
        if (
            positions.ndim != 3
            or positions.shape[:2] != (query.shape[1], query.shape[0])
            or not positions.is_floating_point()
            or positions.device != query.device
            or not torch.isfinite(positions).all()
            or (positions < 0).any()
            or (positions > self.max_positions - 1).any()
        ):
            raise ValueError(
                "positions must be bounded query-dependent [head,query,key]"
            )
        dtype = _accumulation_dtype(query)
        logits_by_position = torch.einsum(
            "qhd,dp->hqp", query.to(dtype), self.position_embeddings.to(dtype)
        )
        lower = positions.floor().long()
        upper = positions.ceil().long()
        fraction = (positions - lower).to(dtype)
        low_bias = logits_by_position.gather(-1, lower)
        high_bias = logits_by_position.gather(-1, upper)
        return low_bias + fraction * (high_bias - low_bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        query_positions: Tensor,
        key_positions: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        fixed_positions: Tensor | None = None,
    ) -> Tensor:
        _qk_layout(query, key)
        allowed, _ = _allowed(
            query, key, query_positions, key_positions, attention_mask
        )
        positions = (
            self.contextual_positions(_logits(query, key), allowed)
            if fixed_positions is None
            else fixed_positions
        )
        if positions.shape != allowed.shape:
            raise ValueError("fixed_positions must exactly match [head,query,key]")
        return self.positional_bias(query, positions)

    def position_trace(
        self,
        query: Tensor,
        key: Tensor,
        query_positions: Tensor,
        key_positions: Tensor | None = None,
        *,
        checkpoint_id: str,
        attention_mask: Tensor | None = None,
        max_elements: int = 1_000_000,
    ) -> CoPEPositionTrace:
        """Capture an explicitly bounded calibration trace, never used implicitly."""
        _qk_layout(query, key)
        if not checkpoint_id.strip():
            raise ValueError("a trained CoPE checkpoint identifier is required")
        if query.shape[1] * query.shape[0] * key.shape[0] > max_elements:
            raise ValueError(
                "CoPE trace exceeds max_elements; use a smaller calibration"
            )
        allowed, key_positions = _allowed(
            query, key, query_positions, key_positions, attention_mask
        )
        logits = _logits(query, key)
        positions = self.contextual_positions(logits, allowed)
        return CoPEPositionTrace(
            positions=positions.detach().cpu().clone(),
            masked_logits=logits.masked_fill(~allowed, -torch.inf).detach().cpu(),
            allowed_mask=allowed.detach().cpu().clone(),
            query_positions=query_positions.detach().cpu().clone(),
            key_positions=key_positions.detach().cpu().clone(),
            max_positions=self.max_positions,
            checkpoint_id=checkpoint_id,
        )


def cope_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cope: CoPE,
    query_positions: Tensor,
    key_positions: Tensor | None = None,
    *,
    attention_mask: Tensor | None = None,
    visibility_mask: Tensor | None = None,
    fixed_positions: Tensor | None = None,
    query_chunk_size: int = 128,
    return_positions: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Differentiable causal/GQA reference with bounded query-axis working memory.

    Only ``return_positions=True`` assembles a full [head,query,key] trace.
    Values may have a head dimension different from keys, but share KV heads.
    Entirely masked query rows return zero. There is no dropout in this reference.
    ``attention_mask`` excludes keys from both gates and attention (e.g. padding).
    ``visibility_mask`` restricts attention only, after contextual gate counting;
    use this for CacheSlide's dynamic-token-plus-self updated associations.
    """
    _qk_layout(query, key)
    _rows("value", value)
    if (
        value.shape[:2] != key.shape[:2]
        or value.device != key.device
        or value.dtype != key.dtype
    ):
        raise ValueError("K/V token count, heads, dtype and device must match")
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    _positions("query_positions", query_positions, query.shape[0], query.device)
    if key_positions is None:
        key_positions = torch.arange(key.shape[0], device=key.device)
    _positions("key_positions", key_positions, key.shape[0], key.device)
    for name, mask in (
        ("attention_mask", attention_mask),
        ("visibility_mask", visibility_mask),
    ):
        if mask is not None and mask.shape not in (
            (query.shape[0], key.shape[0]),
            (1, query.shape[0], key.shape[0]),
            (query.shape[1], query.shape[0], key.shape[0]),
        ):
            raise ValueError(f"{name} has an incompatible full-query shape")
    if fixed_positions is not None and fixed_positions.shape != (
        query.shape[1],
        query.shape[0],
        key.shape[0],
    ):
        raise ValueError(
            "fixed_positions must be full query-dependent [head,query,key]"
        )
    values = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    outputs, traces = [], []
    for start in range(0, query.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[0])
        q = query[start:end]
        chunk_mask = None
        if attention_mask is not None:
            chunk_mask = attention_mask[..., start:end, :]
        allowed, _ = _allowed(
            q, key, query_positions[start:end], key_positions, chunk_mask
        )
        logits = _logits(q, key)
        positions = (
            cope.contextual_positions(logits, allowed)
            if fixed_positions is None
            else fixed_positions[:, start:end]
        )
        logits = logits + cope.positional_bias(q, positions)
        if not torch.isfinite(logits).all():
            raise ValueError("contextual attention logits are not finite")
        if visibility_mask is not None:
            visibility, _ = _allowed(
                q,
                key,
                query_positions[start:end],
                key_positions,
                visibility_mask[..., start:end, :],
            )
            allowed = allowed & visibility
        # softmax of all -inf is undefined; use zero logits then mask its weights.
        any_allowed = allowed.any(-1, keepdim=True)
        logits = logits.masked_fill(~allowed, -torch.inf)
        logits = torch.where(any_allowed, logits, torch.zeros_like(logits))
        probabilities = logits.softmax(-1).masked_fill(~allowed, 0)
        outputs.append(
            torch.einsum("hqk,khd->qhd", probabilities, values.to(logits.dtype))
        )
        if return_positions:
            traces.append(positions)
    if outputs:
        output = torch.cat(outputs, dim=0).to(value.dtype)
    else:
        output = value.new_empty((0, query.shape[1], value.shape[2]))
    if return_positions:
        trace = (
            torch.cat(traces, dim=1)
            if traces
            else query.new_empty((query.shape[1], 0, key.shape[0]))
        )
        return output, trace
    return output


@dataclass(frozen=True)
class ChunkIdentity:
    role: str
    content_hash: str
    length: int

    def __post_init__(self) -> None:
        if not self.role.strip() or not re.fullmatch(
            r"[0-9a-f]{64}", self.content_hash
        ):
            raise ValueError("chunk needs a nonempty role and lowercase SHA-256 hash")
        if not isinstance(self.length, int) or isinstance(self.length, bool):
            raise ValueError("chunk length must be a positive integer")
        if self.length <= 0:
            raise ValueError("chunk length must be a positive integer")


@dataclass(frozen=True)
class CCPEProfile:
    """A bounded task profile, tied to ordered chunks and trained CoPE checkpoint."""

    trained_profile_version: str
    checkpoint_id: str
    chunks: tuple[ChunkIdentity, ...]
    query_positions: Tensor
    key_positions: Tensor
    canonical_positions: Tensor
    max_positions: int
    histogram_bin_width: float
    sample_count: int

    @classmethod
    def calibrate(
        cls,
        traces: Sequence[CoPEPositionTrace],
        chunks: Sequence[ChunkIdentity],
        *,
        trained_profile_version: str,
        histogram_bin_width: float = 0.25,
        max_elements: int = 1_000_000,
    ) -> CCPEProfile:
        """Most frequent joint quantized encoding; return its first real sample.

        All traces must describe the exact same ordered chunk layout. Each
        position is verified against its original masked-logit gate computation,
        including dynamic tokens removed by projection. Tied joint patterns are
        ordered lexicographically. No independent cell modes are combined into
        an encoding that was never observed. The element budget includes source
        logits, projected samples and the resulting profile.
        """
        if not traces or not trained_profile_version.strip():
            raise ValueError("calibration needs traces and a trained profile version")
        if not math.isfinite(histogram_bin_width) or histogram_bin_width <= 0:
            raise ValueError("histogram_bin_width must be finite and positive")
        if not isinstance(traces[0], CoPEPositionTrace):
            raise ValueError("calibration requires genuine CoPEPositionTrace inputs")
        first = traces[0]
        if (
            not chunks
            or sum(chunk.length for chunk in chunks) != first.key_positions.numel()
        ):
            raise ValueError("ordered chunk lengths must exactly match trace keys")
        if (
            sum(
                trace.positions.numel() + trace.masked_logits.numel()
                for trace in traces
                if isinstance(trace, CoPEPositionTrace)
            )
            > max_elements
        ):
            raise ValueError("CCPE calibration exceeds max_elements")
        if first.positions.ndim != 3 or first.max_positions < 2:
            raise ValueError("invalid contextual trace dimensions")
        if not first.checkpoint_id.strip():
            raise ValueError("trace must identify a trained CoPE checkpoint")
        canonical_samples = []
        for trace in traces:
            if not isinstance(trace, CoPEPositionTrace):
                raise ValueError(
                    "calibration requires genuine CoPEPositionTrace inputs"
                )
            if (
                trace.positions.shape != first.positions.shape
                or trace.max_positions != first.max_positions
                or trace.checkpoint_id != first.checkpoint_id
                or not torch.equal(trace.query_positions, first.query_positions)
                or not torch.equal(trace.key_positions, first.key_positions)
            ):
                raise ValueError("calibration trace layout/checkpoint mismatch")
            if (
                trace.masked_logits.ndim != 3
                or trace.masked_logits.shape[0] != trace.positions.shape[0]
                or trace.allowed_mask.shape != trace.masked_logits.shape
                or trace.allowed_mask.dtype != torch.bool
                or not torch.isfinite(trace.positions).all()
                or not torch.isfinite(trace.masked_logits[trace.allowed_mask]).all()
                or not torch.isneginf(trace.masked_logits[~trace.allowed_mask]).all()
            ):
                raise ValueError("invalid masked contextual trace")
            expected = (
                trace.masked_logits.sigmoid()
                .flip(-1)
                .cumsum(-1)
                .flip(-1)
                .clamp(0, trace.max_positions - 1)
            )
            if trace.source_query_indices is not None:
                expected = expected[:, trace.source_query_indices]
            if trace.source_key_indices is not None:
                expected = expected[:, :, trace.source_key_indices]
            if expected.shape != trace.positions.shape:
                raise ValueError("projected trace source mapping has an invalid shape")
            if not torch.allclose(trace.positions, expected, atol=1e-6, rtol=1e-6):
                raise ValueError("trace positions were not produced by CoPE gates")
            canonical_samples.append(trace.positions.detach().cpu())
        histogram: dict[tuple[int, ...], tuple[int, int]] = {}
        for index, sample in enumerate(canonical_samples):
            pattern = tuple(
                (sample / histogram_bin_width).round().long().flatten().tolist()
            )
            count, representative = histogram.get(pattern, (0, index))
            histogram[pattern] = (count + 1, representative)
        most_frequent = min(
            histogram, key=lambda pattern: (-histogram[pattern][0], pattern)
        )
        canonical = canonical_samples[histogram[most_frequent][1]].clone()
        return cls(
            trained_profile_version=trained_profile_version,
            checkpoint_id=first.checkpoint_id,
            chunks=tuple(chunks),
            query_positions=first.query_positions.detach().cpu().clone(),
            key_positions=first.key_positions.detach().cpu().clone(),
            canonical_positions=canonical,
            max_positions=first.max_positions,
            histogram_bin_width=histogram_bin_width,
            sample_count=len(traces),
        )

    def lookup(
        self,
        chunks: Sequence[ChunkIdentity],
        query_positions: Tensor,
        key_positions: Tensor | None = None,
        *,
        checkpoint_id: str,
        trained_profile_version: str,
        strict: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor | None:
        """Return selected query rows, or fail on unseen layout/version/queries.

        ``strict=False`` returns None on a missing profile so callers can choose
        full contextual recomputation. It never extrapolates decode positions.
        """
        if key_positions is None:
            key_positions = torch.arange(sum(chunk.length for chunk in chunks))
        reason = None
        if tuple(chunks) != self.chunks:
            reason = "ordered chunk identities/lengths do not match CCPE profile"
        elif (
            checkpoint_id != self.checkpoint_id
            or trained_profile_version != self.trained_profile_version
        ):
            reason = "trained checkpoint/profile version does not match"
        elif not torch.equal(key_positions.cpu(), self.key_positions):
            reason = "key positions do not match CCPE profile"
        elif query_positions.ndim != 1 or query_positions.dtype != torch.long:
            reason = "query positions must be an int64 vector"
        else:
            known = {int(pos): index for index, pos in enumerate(self.query_positions)}
            if any(int(pos) not in known for pos in query_positions):
                reason = (
                    "CCPE profile has no contextual trace for these query positions"
                )
        if reason:
            if strict:
                raise ValueError(reason)
            return None
        rows = torch.tensor(
            [known[int(pos)] for pos in query_positions], dtype=torch.long
        )
        return self.canonical_positions[:, rows].to(
            device=device or query_positions.device,
            dtype=dtype or self.canonical_positions.dtype,
        )
