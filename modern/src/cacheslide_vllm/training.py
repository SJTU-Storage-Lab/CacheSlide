"""Actual continued next-token training of CoPE and attention low-rank adapters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .artifacts import save_adapter
from .reference import ReferenceLlama


@dataclass(frozen=True)
class TrainingResult:
    output: Path
    losses: tuple[float, ...]
    training_steps: int
    training_tokens: int


def train_adapter(
    model_dir: str | Path,
    token_sequences: Sequence[Sequence[int] | Tensor],
    output: str | Path,
    *,
    steps: int,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    rank: int = 8,
    max_positions: int = 256,
    query_chunk_size: int = 128,
) -> TrainingResult:
    """Optimize causal CE on pretokenized sequences, cycling through them.

    The local backbone remains frozen. Only CoPE embeddings and QKV/output
    low-rank parameters receive gradients. ``training_tokens`` counts supervised
    next-token targets. Output is created only after all requested optimizer
    updates complete; zero-step or nonfinite runs cannot export trained artifacts.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError(
            "steps must be a positive integer; zero-step export is forbidden"
        )
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("adapter output must be a new directory")
    sequences = []
    for item in token_sequences:
        if isinstance(item, Tensor):
            if item.ndim != 1 or item.dtype != torch.long:
                raise ValueError("training tensors must be one-dimensional int64 ids")
            item = item.detach().cpu().tolist()
        sequence = tuple(item)
        if len(sequence) < 2 or any(
            type(token) is not int or token < 0 for token in sequence
        ):
            raise ValueError(
                "each training sequence needs at least two nonnegative ids"
            )
        sequences.append(sequence)
    if not sequences:
        raise ValueError("at least one training sequence is required")
    model = ReferenceLlama.from_checkpoint(
        model_dir,
        rank=rank,
        max_positions=max_positions,
        device=device,
        dtype=torch.float32,
        query_chunk_size=query_chunk_size,
    )
    if any(max(sequence) >= model.config["vocab_size"] for sequence in sequences):
        raise ValueError("training token id is outside the backbone vocabulary")
    model.train()
    trainable = list(model.adapters.parameters())
    if {id(p) for p in model.parameters() if p.requires_grad} != {
        id(p) for p in trainable
    }:
        raise RuntimeError(
            "only the explicitly registered attention adapters may train"
        )
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0)
    losses, training_tokens = [], 0
    for step in range(steps):
        sequence = sequences[step % len(sequences)]
        ids = torch.tensor(sequence, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids[:-1])
        loss = F.cross_entropy(logits.float(), ids[1:])
        if not torch.isfinite(loss):
            raise ValueError("continued training produced a nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        training_tokens += len(sequence) - 1
    artifact = save_adapter(
        output,
        model_dir,
        list(model.adapters),
        training_steps=steps,
        training_tokens=training_tokens,
        losses=losses,
    )
    return TrainingResult(artifact, tuple(losses), steps, training_tokens)
