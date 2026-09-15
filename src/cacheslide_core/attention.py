"""Engine-independent CCPE coordinate and WCA visibility policies.

Both policies operate on one query tile. CoPE's gate computation, interpolation
and softmax stay in position.py; neither policy duplicates attention arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import RequestPlan
from .position import ChunkIdentity, validate_contextual_path


@dataclass(frozen=True)
class CanonicalPositionPolicy:
    plan: RequestPlan
    profile: object
    checkpoint_id: str
    mode: str = "strict_contextual"

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {
            "strict_contextual",
            "mixed_bias_override",
        }:
            raise ValueError("unknown CCPE position policy")

    def __call__(self, contextual: Tensor, queries: Tensor, keys: Tensor) -> Tensor:
        ordinal = {position: i for i, position in enumerate(self.plan.fixed_indices)}
        rows = [i for i, p in enumerate(queries.tolist()) if p in ordinal]
        if not rows:
            return contextual
        chunks = tuple(ChunkIdentity(*entry) for entry in self.plan.fixed_layout)
        canonical = self.profile.lookup(
            chunks,
            torch.tensor([ordinal[int(queries[i])] for i in rows]),
            checkpoint_id=self.checkpoint_id,
            trained_profile_version=self.profile.trained_profile_version,
            device=contextual.device,
            dtype=contextual.dtype,
        )
        fixed_columns = [i for i, p in enumerate(keys.tolist()) if p in ordinal]
        fixed_ordinals = [ordinal[int(keys[i])] for i in fixed_columns]
        result = contextual.clone()
        for index, row in enumerate(rows):
            result[:, row, fixed_columns] = canonical[:, index, fixed_ordinals]
        if self.mode == "strict_contextual":
            # Combining canonical fixed cells with current dynamic cells need
            # not yield a sigmoid-gate path. Never silently project either set.
            validate_contextual_path(result, queries, keys)
        elif self.mode != "mixed_bias_override":
            raise ValueError("unknown CCPE position policy")
        return result


@dataclass(frozen=True)
class SelectedAssociationPolicy:
    fixed_indices: tuple[int, ...]
    selected_indices: Tensor

    def __call__(self, queries: Tensor, keys: Tensor) -> Tensor:
        fixed = torch.tensor(self.fixed_indices, device=keys.device, dtype=torch.long)
        selected = torch.isin(queries, self.selected_indices)
        return (
            ~selected[:, None]
            | ~torch.isin(keys, fixed)[None, :]
            | (queries[:, None] == keys[None, :])
        )
