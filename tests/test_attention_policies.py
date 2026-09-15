import pytest
import torch

from cacheslide_vllm.attention import (
    CanonicalPositionPolicy,
    SelectedAssociationPolicy,
)
from cacheslide_vllm.contracts import RequestPlan
from cacheslide_vllm.position import (
    CCPEPositionError,
    CCPEProfile,
    ChunkIdentity,
    CoPE,
    cope_attention,
)


def plan_with_dynamic_length(length):
    return RequestPlan.parse(
        {
            "version": 1,
            "namespace": "policy-test",
            "task_id": "two-fixed-chunks",
            "operation": "reuse",
            "chunks": [
                {"id": "A", "role": "reuse", "start": 0, "end": 1},
                {
                    "id": "update",
                    "role": "recompute",
                    "start": 1,
                    "end": 1 + length,
                },
                {
                    "id": "B",
                    "role": "reuse",
                    "start": 1 + length,
                    "end": 2 + length,
                },
            ],
        },
        [11, *([9] * length), 22],
    )


def calibrated_profile():
    cope = CoPE(1, 16, dtype=torch.float64)
    original = plan_with_dynamic_length(1)
    trace = cope.position_trace(
        torch.ones(3, 2, 1, dtype=torch.float64),
        torch.zeros(3, 1, 1, dtype=torch.float64),
        torch.arange(3),
        checkpoint_id="trained-cope",
    )
    fixed = torch.tensor(original.fixed_indices)
    profile = CCPEProfile.calibrate(
        [trace.project(fixed, fixed)],
        [ChunkIdentity(*entry) for entry in original.fixed_layout],
        trained_profile_version="task-profile-v1",
    )
    return cope, profile


def test_strict_policy_rejects_real_three_to_five_token_hybrid():
    cope, profile = calibrated_profile()
    plan = plan_with_dynamic_length(3)
    query = torch.ones(1, 2, 1, dtype=torch.float64)
    key = torch.zeros(5, 1, 1, dtype=torch.float64)
    policy = CanonicalPositionPolicy(plan, profile, "trained-cope")
    with pytest.raises(CCPEPositionError, match="nonincreasing"):
        cope_attention(
            query,
            key,
            key,
            cope,
            torch.tensor([4]),
            positions_transform=policy,
        )
    mixed = CanonicalPositionPolicy(
        plan, profile, "trained-cope", mode="mixed_bias_override"
    )
    output, trace = cope_attention(
        query,
        key,
        key,
        cope,
        torch.tensor([4]),
        positions_transform=mixed,
        return_positions=True,
    )
    expected = torch.tensor([[[1.5, 2, 1.5, 1, 0.5]]], dtype=torch.float64)
    torch.testing.assert_close(trace, expected.expand(2, -1, -1))
    assert output.dtype == trace.dtype == torch.float64


def test_strict_policy_accepts_same_layout_actual_contextual_path():
    cope, profile = calibrated_profile()
    plan = plan_with_dynamic_length(1)
    q = torch.ones(3, 2, 1, dtype=torch.float64)
    k = torch.zeros(3, 1, 1, dtype=torch.float64)
    trace = cope.position_trace(q, k, torch.arange(3), checkpoint_id="trained-cope")
    policy = CanonicalPositionPolicy(plan, profile, "trained-cope")
    actual = policy(trace.positions, torch.arange(3), torch.arange(3))
    torch.testing.assert_close(actual, trace.positions)
    assert actual.dtype == torch.float64


def test_canonical_policy_query_and_key_tiles_use_actual_to_fixed_ordinals():
    cope, profile = calibrated_profile()
    plan = plan_with_dynamic_length(3)
    trace = cope.position_trace(
        torch.ones(5, 2, 1, dtype=torch.float64),
        torch.zeros(5, 1, 1, dtype=torch.float64),
        torch.arange(5),
        checkpoint_id="trained-cope",
    )
    policy = CanonicalPositionPolicy(
        plan, profile, "trained-cope", mode="mixed_bias_override"
    )
    original = trace.positions.clone()
    full = policy(trace.positions, torch.arange(5), torch.arange(5))
    qtile, ktile = torch.tensor([4, 1, 0]), torch.tensor([0, 2, 4])
    tile = policy(trace.positions[:, qtile][:, :, ktile], qtile, ktile)
    torch.testing.assert_close(tile, full[:, qtile][:, :, ktile])
    torch.testing.assert_close(trace.positions, original)  # no in-place mutation
    torch.testing.assert_close(full[:, 1:4], original[:, 1:4])  # dynamic queries
    assert tile.dtype == torch.float64
    dynamic_only = original[:, 1:2]
    assert policy(dynamic_only, torch.tensor([1]), torch.arange(5)) is dynamic_only


def test_selected_association_query_key_tiles_and_future_keys_stay_causal():
    policy = SelectedAssociationPolicy((0, 2, 4), torch.tensor([4, 2]))
    queries, keys = torch.tensor([4, 1, 2]), torch.tensor([0, 1, 2, 3, 4, 5])
    mask = policy(queries, keys)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [False, True, False, True, True, True],
        [True, True, True, True, True, True],
        [False, True, True, True, False, True],
    ]
    torch.testing.assert_close(policy(queries[1:], keys[2:]), mask[1:, 2:])
    # Visibility may include future dynamic keys; the shared kernel must still
    # intersect it with causality, rather than let the policy replace the mask.
    cope = CoPE(1, 16, dtype=torch.float64)
    q = torch.ones(3, 1, 1, dtype=torch.float64)
    k = torch.zeros(6, 1, 1, dtype=torch.float64)
    v = torch.arange(6, dtype=torch.float64)[:, None, None]
    actual = cope_attention(
        q, k, v, cope, queries, visibility_transform=policy, query_chunk_size=1
    )
    torch.testing.assert_close(
        actual[:, 0, 0], torch.tensor([8 / 3, 0.5, 1.5], dtype=torch.float64)
    )


def test_wca_association_is_applied_after_all_causal_gates():
    cope = CoPE(1, 16, dtype=torch.float64)
    with torch.no_grad():
        cope.position_embeddings.copy_(torch.arange(16, dtype=torch.float64)[None])
    q = torch.ones(1, 1, 1, dtype=torch.float64)
    k = torch.zeros(5, 1, 1, dtype=torch.float64)
    v = torch.tensor([1, 10, 100, 1000, 10000], dtype=torch.float64)[:, None, None]
    queries, keys = torch.tensor([4]), torch.arange(5)
    association = SelectedAssociationPolicy((0, 2, 4), torch.tensor([4]))
    output, positions = cope_attention(
        q,
        k,
        v,
        cope,
        queries,
        visibility_transform=association,
        return_positions=True,
    )
    expected_positions = torch.tensor([[[2.5, 2, 1.5, 1, 0.5]]], dtype=torch.float64)
    torch.testing.assert_close(positions, expected_positions)
    visible = association(queries, keys)[0]
    expected = expected_positions[0, 0, visible].softmax(0) @ v[visible, 0, 0]
    torch.testing.assert_close(output[0, 0, 0], expected)
    wrongly_gated, wrong_positions = cope_attention(
        q,
        k,
        v,
        cope,
        queries,
        attention_mask=association(queries, keys),
        return_positions=True,
    )
    assert wrong_positions[0, 0, 1] == 1.5
    assert not torch.allclose(output, wrongly_gated)
    assert output.dtype == positions.dtype == torch.float64
