"""Request-local weighted cache adaptation, with explicit paper ambiguities.

Tokens are indexed in assembled-prompt order; all K/V tensors use [token, head,
dimension]. Layer numbers are one based. Initialization observes a full first
layer recomputation. Later calls consume only ``state.active_indices`` rows.

The literal Algorithm 2 comparator is ``cosine < threshold``. Its prose calls
this convergence, although low cosine normally means disagreement. The separate
``distance_lt`` mode is an experiment, not a silent correction of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .policy import WCAConfig, _finite_number


class WCANumericalError(ValueError):
    """Nonfinite WCA data or arithmetic that permits full-prefill recovery.

    Shape, policy, and call-order violations remain ordinary ValueError so a
    runtime can recover from numerical failure without masking integration bugs.
    """


def _kv(name: str, tensor: Tensor) -> None:
    if tensor.ndim != 3 or not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating [token, head, dimension]")
    if min(tensor.shape[1:]) < 1:
        raise ValueError(f"{name} must have nonempty heads/dimension")
    if not torch.isfinite(tensor).all():
        raise WCANumericalError(f"{name} must have finite values")


def _matching(name: str, tensor: Tensor, reference: Tensor) -> None:
    if (tensor.shape, tensor.dtype, tensor.device) != (
        reference.shape,
        reference.dtype,
        reference.device,
    ):
        raise ValueError(f"{name} shape, dtype and device must match its reference")
    _kv(name, tensor)


def _mask(name: str, tensor: Tensor, n: int, device: torch.device) -> None:
    if tensor.shape != (n,) or tensor.dtype != torch.bool or tensor.device != device:
        raise ValueError(f"{name} must be a bool [{n}] tensor on {device}")


def squared_deviation(cached_k: Tensor, recomputed_k: Tensor) -> Tensor:
    """Per-token squared K error, summed over all heads and head dimensions."""
    _kv("cached_k", cached_k)
    _matching("recomputed_k", recomputed_k, cached_k)
    result = (recomputed_k.double() - cached_k.double()).square().sum((1, 2))
    if not torch.isfinite(result).all():
        raise WCANumericalError("squared deviation overflowed float64")
    return result


def adaptation_weight(
    cached_k: Tensor,
    recomputed_k: Tensor,
    *,
    epsilon: float = 1e-8,
    clamp: bool = False,
) -> Tensor:
    """Return raw ||K_new-K_cache||²/(||K_cache||²+epsilon) per token.

    Float64 accumulation avoids float16/float32 squaring overflow. The paper's
    raw ratio can exceed one; clamping is an explicit non-default experiment.
    """
    _kv("cached_k", cached_k)
    _matching("recomputed_k", recomputed_k, cached_k)
    _finite_number("epsilon", epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if type(clamp) is not bool:
        raise ValueError("clamp must be a boolean")
    numerator = (recomputed_k.double() - cached_k.double()).square().sum((1, 2))
    denominator = cached_k.double().square().sum((1, 2)) + epsilon
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise WCANumericalError("adaptation weight overflowed float64")
    result = numerator / denominator
    if not torch.isfinite(result).all():
        raise WCANumericalError(
            "adaptation weight overflowed; inspect the K magnitudes"
        )
    return result.clamp(0, 1) if clamp else result


def mean_head_cosine(a: Tensor, b: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    """Mean cosine over heads; zero norm heads contribute zero."""
    _kv("a", a)
    _matching("b", b, a)
    _finite_number("epsilon", epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    # Scaling first also keeps norms/products safe for very large finite K.
    a64, b64 = a.double(), b.double()
    a_scale = a64.abs().amax(-1, keepdim=True).clamp_min(epsilon)
    b_scale = b64.abs().amax(-1, keepdim=True).clamp_min(epsilon)
    a_scaled, b_scaled = a64 / a_scale, b64 / b_scale
    a_norm = torch.linalg.vector_norm(a_scaled, dim=-1, keepdim=True)
    b_norm = torch.linalg.vector_norm(b_scaled, dim=-1, keepdim=True)
    a_unit = a_scaled / a_norm.clamp_min(torch.finfo(torch.float64).tiny)
    b_unit = b_scaled / b_norm.clamp_min(torch.finfo(torch.float64).tiny)
    return (a_unit * b_unit).sum(-1).clamp(-1, 1).mean(-1)


@dataclass(frozen=True)
class WCAUpdate:
    """Corrected current-layer rows and selection changes for the next layer."""

    indices: Tensor
    fused_k: Tensor
    fused_v: Tensor
    alpha: Tensor
    removed_indices: Tensor
    promoted_indices: Tensor
    next_indices: Tensor


@dataclass
class WCAState:
    """State owned by one request; input masks are cloned during initialization.

    ``previous_layer`` seeds weights from layer one, uses the stored weight to
    fuse current K/V, then updates the weight from the current raw recomputed K.
    ``same_layer`` computes the weight before fusion. Both modes use one weight
    per token for K and V. Promoted tokens retain their most recently observed
    weight (initially layer one's); a runtime must restore their cached hidden
    states before computing the next layer.
    """

    config: WCAConfig
    reused_mask: Tensor
    mandatory_mask: Tensor
    initial_deviation: Tensor
    candidate_mask: Tensor
    selected_mask: Tensor
    removed_mask: Tensor
    alpha: Tensor
    budget: int
    last_layer: int = 1

    @classmethod
    def initialize(
        cls,
        cached_k: Tensor,
        recomputed_k: Tensor,
        reused_mask: Tensor,
        mandatory_mask: Tensor,
        config: WCAConfig | None = None,
    ) -> WCAState:
        config = config or WCAConfig()
        error = squared_deviation(cached_k, recomputed_k)
        n = cached_k.shape[0]
        _mask("reused_mask", reused_mask, n, cached_k.device)
        _mask("mandatory_mask", mandatory_mask, n, cached_k.device)
        if ((~reused_mask) & (~mandatory_mask)).any():
            raise ValueError("every dynamic/non-reused token must be mandatory")
        candidate = reused_mask & (error > 0)
        budget = min(
            math.ceil(config.correction_fraction * int(reused_mask.sum().item())),
            int(candidate.sum().item()),
        )
        state = cls(
            config=config,
            reused_mask=reused_mask.clone(),
            mandatory_mask=mandatory_mask.clone(),
            initial_deviation=error.detach().clone(),
            candidate_mask=candidate.clone(),
            selected_mask=torch.zeros_like(candidate),
            removed_mask=torch.zeros_like(candidate),
            alpha=adaptation_weight(
                cached_k,
                recomputed_k,
                epsilon=config.epsilon,
                clamp=config.clamp_alpha,
            )
            .detach()
            .clone(),
            budget=budget,
        )
        state._promote()
        return state

    @property
    def active_indices(self) -> Tensor:
        """Sorted global token indices to compute at the next layer."""
        return (self.selected_mask | self.mandatory_mask).nonzero().flatten()

    @property
    def selected_indices(self) -> Tensor:
        return self.selected_mask.nonzero().flatten()

    def _promote(self) -> Tensor:
        available = (
            (self.candidate_mask & ~self.selected_mask & ~self.removed_mask)
            .nonzero()
            .flatten()
        )
        slots = max(0, self.budget - int(self.selected_mask.sum().item()))
        # nonzero gives ascending indices; stable sort resolves ties by token id.
        order = torch.argsort(
            self.initial_deviation[available], descending=True, stable=True
        )
        promoted = available[order[:slots]]
        self.selected_mask[promoted] = True
        return promoted

    def update(
        self,
        layer_index: int,
        cached_k: Tensor,
        cached_v: Tensor,
        recomputed_k: Tensor,
        recomputed_v: Tensor,
        computed_indices: Tensor | None = None,
    ) -> WCAUpdate:
        """Fuse sparse recomputed rows against full current-layer cached K/V.

        ``computed_indices`` defaults to active_indices, and must equal it in
        sorted order. Fresh K/V contain only those rows. Dynamic and mandatory
        rows always use the fresh K/V without weighting.
        """
        if layer_index != self.last_layer + 1:
            raise ValueError("layers must update consecutively after full layer one")
        _kv("cached_k", cached_k)
        _matching("cached_v", cached_v, cached_k)
        if cached_k.shape[0] != self.reused_mask.numel():
            raise ValueError("cache token count differs from the request")
        if cached_k.device != self.reused_mask.device:
            raise ValueError("cache device differs from the request")
        indices = self.active_indices
        if computed_indices is not None:
            if (
                computed_indices.dtype != torch.long
                or computed_indices.device != indices.device
                or not torch.equal(computed_indices, indices)
            ):
                raise ValueError("computed_indices must equal sorted active_indices")
        old_k, old_v = cached_k[indices], cached_v[indices]
        _matching("recomputed_k", recomputed_k, old_k)
        _matching("recomputed_v", recomputed_v, old_v)
        next_alpha = adaptation_weight(
            old_k,
            recomputed_k,
            epsilon=self.config.epsilon,
            clamp=self.config.clamp_alpha,
        )
        used_alpha = (
            self.alpha[indices]
            if self.config.weight_update == "previous_layer"
            else next_alpha
        )
        # Mandatory tokens are fresh, even if a reused token was marked mandatory.
        used_alpha = torch.where(
            self.mandatory_mask[indices], torch.ones_like(used_alpha), used_alpha
        )
        weight = used_alpha[:, None, None]
        fused_k = (weight * recomputed_k.double() + (1 - weight) * old_k.double()).to(
            cached_k.dtype
        )
        fused_v = (weight * recomputed_v.double() + (1 - weight) * old_v.double()).to(
            cached_v.dtype
        )
        if not torch.isfinite(fused_k).all() or not torch.isfinite(fused_v).all():
            raise WCANumericalError("weighted K/V overflowed the cache dtype")

        removed = indices[:0]
        if layer_index % self.config.gate_interval == 0:
            cosine = mean_head_cosine(recomputed_k, old_k, epsilon=self.config.epsilon)
            metric = (
                cosine
                if self.config.convergence_mode == "paper_cosine_lt"
                else 1 - cosine
            )
            remove_local = self.selected_mask[indices] & (
                metric < self.config.convergence_threshold
            )
            removed = indices[remove_local]

        # Commit only after validation and numeric checks have passed.
        self.alpha[indices] = next_alpha.detach()
        self.selected_mask[removed] = False
        self.candidate_mask[removed] = False
        self.removed_mask[removed] = True
        promoted = self._promote()
        self.last_layer = layer_index
        return WCAUpdate(
            indices=indices,
            fused_k=fused_k,
            fused_v=fused_v,
            alpha=used_alpha.clone(),
            removed_indices=removed,
            promoted_indices=promoted,
            next_indices=self.active_indices,
        )
