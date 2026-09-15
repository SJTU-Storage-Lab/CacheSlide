"""Bounded, strict token-ID JSONL inputs shared by both engine adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import RequestPlan


@dataclass(frozen=True)
class InputCase:
    case_id: str
    token_ids: tuple[int, ...]
    plan: RequestPlan | None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def read_cases(path: str | Path, *, require_plan: bool = True) -> list[InputCase]:
    """Read JSONL: id (optional), prompt_token_ids, exact cacheslide metadata."""
    cases, seen = [], set()
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(line.encode()) > 1_048_576:
                raise ValueError(f"input line {line_number} exceeds 1 MiB")
            row = json.loads(
                line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if not isinstance(row, dict) or set(row) - {
                "id",
                "prompt_token_ids",
                "cacheslide",
            }:
                raise ValueError(
                    f"input line {line_number} has unknown or invalid fields"
                )
            tokens = row.get("prompt_token_ids")
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
            ):
                raise ValueError(
                    "prompt_token_ids must be nonempty nonnegative integer ids"
                )
            case_id = row.get("id", str(line_number))
            if (
                not isinstance(case_id, str)
                or not case_id
                or len(case_id) > 256
                or any(ord(char) < 32 for char in case_id)
                or case_id in seen
            ):
                raise ValueError("case ids must be distinct nonempty strings")
            metadata = row.get("cacheslide")
            if require_plan and metadata is None:
                raise ValueError("each request needs exact cacheslide chunk metadata")
            plan = RequestPlan.parse(metadata, tokens) if metadata is not None else None
            cases.append(InputCase(case_id, tuple(tokens), plan))
            seen.add(case_id)
    if not cases:
        raise ValueError("input JSONL contains no cases")
    return cases
