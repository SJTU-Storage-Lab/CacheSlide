"""Validated RPDC token layouts, independent of vLLM request internals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    role: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RequestPlan:
    """An exact token partition, not a text-to-token heuristic.

    Reusable chunks retain their relative order. Cache identity includes all
    preceding reusable chunks, so moving a chunk or changing its fixed context
    is a miss. Dynamic chunks deliberately do not participate in that identity.
    """

    operation: str
    namespace: str
    task_id: str
    chunks: tuple[Chunk, ...]
    token_ids: tuple[int, ...]

    @classmethod
    def parse(cls, value: str | Mapping, token_ids: Sequence[int]) -> RequestPlan:
        if isinstance(value, str):
            if len(value.encode()) > 1_048_576:
                raise ValueError("CacheSlide request metadata exceeds 1 MiB")
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise ValueError("cacheslide must be a JSON object or encoded JSON string")
        allowed = {"version", "operation", "namespace", "task_id", "chunks"}
        if (
            set(value) - allowed
            or type(value.get("version")) is not int
            or value["version"] != 1
        ):
            raise ValueError("unknown CacheSlide request field or schema version")
        operation = value.get("operation")
        if not isinstance(operation, str) or operation not in {
            "recompute",
            "populate",
            "reuse",
            "calibrate",
        }:
            raise ValueError(
                "operation must be recompute, populate, reuse, or calibrate"
            )
        tokens = tuple(token_ids)
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("prompt token ids must be nonnegative integers")
        raw_chunks = value.get("chunks")
        if not isinstance(raw_chunks, (list, tuple)) or not raw_chunks:
            raise ValueError("chunks must be a nonempty list")
        if len(raw_chunks) > 1024:
            raise ValueError("at most 1024 chunks are supported")
        chunks, cursor, seen = [], 0, set()
        for entry in raw_chunks:
            if not isinstance(entry, Mapping) or set(entry) != {
                "id",
                "role",
                "start",
                "end",
            }:
                raise ValueError("each chunk needs exactly id, role, start, end")
            chunk_id = _identifier(entry["id"], "chunk id")
            start, end = entry["start"], entry["end"]
            if type(start) is not int or type(end) is not int:
                raise ValueError("chunk boundaries must be integer token offsets")
            if start != cursor or not start < end <= len(tokens):
                raise ValueError(
                    "chunks must partition the prompt without gaps/overlaps"
                )
            if (
                not isinstance(entry["role"], str)
                or entry["role"] not in {"reuse", "recompute"}
                or chunk_id in seen
            ):
                raise ValueError("invalid role or duplicate chunk id")
            chunks.append(Chunk(chunk_id, entry["role"], start, end))
            cursor = end
            seen.add(chunk_id)
        if cursor != len(tokens):
            raise ValueError("chunks must include every prompt token")
        return cls(
            operation,
            _identifier(value.get("namespace"), "namespace"),
            _identifier(value.get("task_id"), "task_id"),
            tuple(chunks),
            tokens,
        )

    @property
    def fixed_indices(self) -> tuple[int, ...]:
        return tuple(
            i for c in self.chunks if c.role == "reuse" for i in range(c.start, c.end)
        )

    @property
    def mandatory_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {len(self.token_ids) - 1}
                | {
                    i
                    for c in self.chunks
                    if c.role == "recompute"
                    for i in range(c.start, c.end)
                }
            )
        )

    @property
    def fixed_layout(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (c.chunk_id, digest(self.token_ids[c.start : c.end]), c.length)
            for c in self.chunks
            if c.role == "reuse"
        )

    def cache_key(self, model_identity: str, layer: int) -> str:
        # Entire ordered fixed layout protects fixed-to-fixed dependencies.
        return digest(
            [
                "CacheSlide-KV-v1",
                model_identity,
                self.namespace,
                self.task_id,
                self.fixed_layout,
                layer,
            ]
        )

    def profile_key(self, model_identity: str, layer: int) -> str:
        return digest(["CacheSlide-CCPE-v1", self.cache_key(model_identity, layer)])

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "operation": self.operation,
                "namespace": self.namespace,
                "task_id": self.task_id,
                "chunks": [
                    {"id": c.chunk_id, "role": c.role, "start": c.start, "end": c.end}
                    for c in self.chunks
                ],
            },
            separators=(",", ":"),
        )
