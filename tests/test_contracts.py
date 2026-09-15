import copy
import json

import pytest

from cacheslide_core.contracts import RequestPlan


def request_metadata():
    return {
        "version": 1,
        "operation": "reuse",
        "namespace": "test-model",
        "task_id": "task-1",
        "chunks": [
            {"id": "document", "role": "reuse", "start": 0, "end": 2},
            {"id": "query", "role": "recompute", "start": 2, "end": 3},
        ],
    }


@pytest.mark.parametrize("invalid", [[], {}, None, True, 1, 1.0, "unknown"])
@pytest.mark.parametrize("field", ["operation", "role"])
def test_invalid_operation_and_role_raise_validation_errors(field, invalid):
    metadata = request_metadata()
    if field == "role":
        metadata["chunks"][0]["role"] = invalid
    else:
        metadata[field] = invalid
    with pytest.raises(ValueError):
        RequestPlan.parse(json.dumps(metadata), [10, 20, 30])


@pytest.mark.parametrize("invalid", [True, False, 1.0, "1", [], {}, None, 2])
def test_schema_version_requires_the_exact_integer(invalid):
    metadata = request_metadata()
    metadata["version"] = invalid
    with pytest.raises(ValueError, match="schema version"):
        RequestPlan.parse(metadata, [10, 20, 30])


def test_valid_plan_round_trips_and_snapshots_input():
    metadata = request_metadata()
    tokens = [10, 20, 30]
    plan = RequestPlan.parse(metadata, tokens)
    assert RequestPlan.parse(plan.to_json(), tokens) == plan
    metadata["chunks"][0]["id"] = "changed"
    tokens[0] = 99
    assert plan.chunks[0].chunk_id == "document"
    assert plan.token_ids == (10, 20, 30)
    assert plan.fixed_indices == (0, 1)
    assert plan.mandatory_indices == (2,)


def test_cache_identity_includes_ordered_template():
    original = RequestPlan.parse(request_metadata(), [10, 20, 30])
    moved = request_metadata()
    moved["chunks"] = [
        {"id": "prefix", "role": "recompute", "start": 0, "end": 2},
        {"id": "document", "role": "reuse", "start": 2, "end": 4},
        {"id": "query", "role": "recompute", "start": 4, "end": 5},
    ]
    shifted = RequestPlan.parse(moved, [98, 99, 10, 20, 31])
    assert shifted.cache_key("backbone-adapter", 2) != original.cache_key(
        "backbone-adapter", 2
    )
    changed = RequestPlan.parse(copy.deepcopy(moved), [98, 99, 10, 21, 31])
    assert changed.cache_key("backbone-adapter", 2) != original.cache_key(
        "backbone-adapter", 2
    )
    assert original.cache_key("other-backbone", 2) != original.cache_key(
        "backbone-adapter", 2
    )


def test_same_template_accepts_dynamic_length_and_content_changes():
    metadata = request_metadata()
    original = RequestPlan.parse(metadata, [10, 20, 30])
    metadata["chunks"][-1]["end"] = 5
    changed = RequestPlan.parse(metadata, [10, 20, 98, 99, 31])
    assert original.cache_key("m", 2) == changed.cache_key("m", 2)
    assert original.profile_key("m", 2) == changed.profile_key("m", 2)


@pytest.mark.parametrize("boundary", [True, 1.0, "1"])
def test_chunk_boundaries_reject_noninteger_offsets(boundary):
    metadata = request_metadata()
    metadata["chunks"][0]["start"] = boundary
    with pytest.raises(ValueError, match="integer token offsets"):
        RequestPlan.parse(metadata, [10, 20, 30])
