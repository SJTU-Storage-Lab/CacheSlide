import pytest
import torch

from cacheslide_vllm.wca import (
    WCAConfig,
    WCAState,
    adaptation_weight,
    mean_head_cosine,
    squared_deviation,
)


def make_state(config=None):
    cached = torch.ones(6, 1, 2)
    fresh = torch.tensor([5, 4, 3, 2, 1, 100.0])[:, None, None].expand_as(cached)
    reused = torch.tensor([True, True, True, True, True, False])
    mandatory = ~reused
    state = WCAState.initialize(cached, fresh, reused, mandatory, config)
    return state, cached, fresh, reused, mandatory


def test_positive_reused_candidates_and_ceil_budget():
    state, _, _, reused, mandatory = make_state()
    assert state.budget == 2  # ceil(.26 * 5), capped by positive reusable errors
    assert state.selected_indices.tolist() == [0, 1]
    assert state.active_indices.tolist() == [0, 1, 5]
    assert state.candidate_mask.tolist() == [True, True, True, True, False, False]
    reused[:] = False
    mandatory[:] = False
    assert state.reused_mask.sum() == 5
    assert state.mandatory_mask[-1]


def test_zero_errors_and_deterministic_tie_breaking():
    cached = torch.ones(7, 2, 3)
    mask = torch.ones(7, dtype=torch.bool)
    state = WCAState.initialize(cached, cached + 1, mask, ~mask)
    assert state.selected_indices.tolist() == [0, 1]
    assert torch.equal(squared_deviation(cached, cached + 1), torch.full((7,), 6.0))
    empty = WCAState.initialize(cached, cached, mask, ~mask)
    assert empty.budget == 0
    assert empty.active_indices.numel() == 0


def test_weight_raw_ratio_zero_norm_and_wide_accumulation():
    cached = torch.zeros(1, 1, 2, dtype=torch.float16)
    fresh = torch.ones_like(cached)
    alpha = adaptation_weight(cached, fresh, epsilon=0.5)
    assert alpha.dtype == torch.float64
    assert alpha.item() == 4
    assert adaptation_weight(cached, fresh, epsilon=0.5, clamp=True).item() == 1
    assert adaptation_weight(cached, cached).item() == 0
    huge = torch.full((1, 1, 2), 1e30)
    assert adaptation_weight(huge, 2 * huge).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="overflow"):
        adaptation_weight(
            torch.zeros(1, 1, 1, dtype=torch.float64),
            torch.full((1, 1, 1), 1e308, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "mode, expected_alpha", [("previous_layer", 4), ("same_layer", 1)]
)
def test_previous_layer_and_same_layer_weight_order(mode, expected_alpha):
    cached = torch.ones(2, 1, 1)
    fresh_first = torch.tensor([3.0, 9.0])[:, None, None]
    reused = torch.tensor([True, False])
    state = WCAState.initialize(
        cached,
        fresh_first,
        reused,
        ~reused,
        WCAConfig(correction_fraction=1, epsilon=1e-20, weight_update=mode),
    )
    raw_k = torch.tensor([2.0, 7.0])[:, None, None]
    raw_v = torch.tensor([4.0, 8.0])[:, None, None]
    result = state.update(2, cached, cached * 2, raw_k, raw_v)
    assert result.alpha.tolist() == pytest.approx([expected_alpha, 1])
    assert result.fused_k[:, 0, 0].tolist() == pytest.approx([1 + expected_alpha, 7])
    assert result.fused_v[:, 0, 0].tolist() == pytest.approx(
        [2 + 2 * expected_alpha, 8]
    )
    assert state.alpha[0].item() == pytest.approx(1)  # raw-current ratio, not fused


def test_literal_gate_promotes_initial_error_and_never_reselects():
    state, cached, _, _, _ = make_state(WCAConfig(gate_interval=2))
    rows = state.active_indices
    result = state.update(2, cached, cached, -cached[rows], cached[rows])
    assert result.removed_indices.tolist() == [0, 1]
    assert result.promoted_indices.tolist() == [2, 3]
    assert result.next_indices.tolist() == [2, 3, 5]
    assert not state.candidate_mask[:2].any()
    rows = state.active_indices
    state.update(3, cached, cached, -cached[rows], cached[rows])
    result = state.update(4, cached, cached, -cached[rows], cached[rows])
    assert result.removed_indices.tolist() == [2, 3]
    assert result.promoted_indices.numel() == 0
    assert result.next_indices.tolist() == [5]


def test_distance_gate_is_separate_and_default_interval_is_four():
    literal, cached, _, _, _ = make_state()
    distance, _, _, _, _ = make_state(WCAConfig(convergence_mode="distance_lt"))
    for layer in (2, 3, 4):
        for state in (literal, distance):
            rows = state.active_indices
            result = state.update(layer, cached, cached, cached[rows], cached[rows])
            if layer < 4:
                assert result.removed_indices.numel() == 0
    assert literal.selected_indices.tolist() == [0, 1]
    assert distance.selected_indices.tolist() == [2, 3]


def test_cosine_is_mean_over_heads_and_handles_zero():
    a = torch.tensor([[[1.0, 0], [0, 1]], [[0, 0], [0, 0]]])
    b = torch.tensor([[[1.0, 0], [0, -1]], [[1, 1], [1, 1]]])
    assert mean_head_cosine(a, b).tolist() == [0, 0]


def test_request_isolation_and_sparse_identity_validation():
    first, cached, _, _, _ = make_state(WCAConfig(gate_interval=2))
    second, _, _, _, _ = make_state(WCAConfig(gate_interval=2))
    rows = first.active_indices
    first.update(2, cached, cached, -cached[rows], cached[rows])
    assert second.selected_indices.tolist() == [0, 1]
    with pytest.raises(ValueError, match="computed_indices"):
        second.update(2, cached, cached, cached[rows], cached[rows], rows.flip(0))
    with pytest.raises(ValueError, match="consecutively"):
        second.update(3, cached, cached, cached[rows], cached[rows])
    with pytest.raises(ValueError, match="shape, dtype"):
        second.update(2, cached, cached, cached, cached)


def test_mandatory_reused_queries_are_always_fresh():
    cached = torch.ones(3, 1, 1)
    reused = torch.tensor([True, True, False])
    mandatory = torch.tensor([False, True, True])
    state = WCAState.initialize(cached, cached + 2, reused, mandatory)
    result = state.update(2, cached, cached, cached * 2, cached * 3)
    assert result.fused_k[1:, 0, 0].tolist() == [2, 2]
    assert result.fused_v[1:, 0, 0].tolist() == [3, 3]


def test_invalid_masks_nonfinite_and_transactional_overflow():
    cached = torch.ones(1, 1, 1)
    with pytest.raises(ValueError, match="mandatory"):
        WCAState.initialize(
            cached, cached, torch.tensor([False]), torch.tensor([False])
        )
    with pytest.raises(ValueError, match="finite"):
        adaptation_weight(cached, cached * torch.nan)
    half = torch.zeros(1, 1, 1, dtype=torch.float16)
    state = WCAState.initialize(
        half, half + 1, torch.tensor([True]), torch.tensor([False])
    )
    with pytest.raises(ValueError, match="overflow"):
        state.update(2, half, half, half + 1, half + 1)
    assert state.last_layer == 1
    assert state.selected_indices.tolist() == [0]
