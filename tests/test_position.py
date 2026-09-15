from dataclasses import replace

import pytest
import torch

from cacheslide_vllm.position import CCPEProfile, ChunkIdentity, CoPE, cope_attention


def test_gate_masking_reverse_cumsum_and_clamp():
    cope = CoPE(2, 2)
    logits = torch.zeros(1, 3, 3)
    allowed = torch.ones(3, 3, dtype=torch.bool).tril().unsqueeze(0)
    logits[~allowed] = 1000  # future keys must never influence contextual counts
    positions = cope.contextual_positions(logits, allowed)
    expected = torch.tensor([[[0.5, 0, 0], [1, 0.5, 0], [1, 1, 0.5]]])
    torch.testing.assert_close(positions, expected)


def test_interpolation_and_gradients_through_gate_and_embeddings():
    cope = CoPE(1, 4, dtype=torch.float64)
    with torch.no_grad():
        cope.position_embeddings.copy_(torch.tensor([[0, 2, 6, 10]]))
    query = torch.tensor([[[2.0]]], dtype=torch.float64, requires_grad=True)
    positions = torch.tensor(
        [[[0.25, 1.5, 3.0]]], dtype=torch.float64, requires_grad=True
    )
    bias = cope.positional_bias(query, positions)
    torch.testing.assert_close(bias, torch.tensor([[[1, 8, 20]]], dtype=torch.float64))
    bias.sum().backward()
    assert query.grad.abs().sum() > 0
    assert positions.grad.abs().sum() > 0
    assert cope.position_embeddings.grad.abs().sum() > 0

    cope.zero_grad()
    query = torch.tensor([[[0.7]], [[0.5]]], dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[[0.2]], [[0.4]]], dtype=torch.float64, requires_grad=True)
    cope(query, key, torch.arange(2)).sum().backward()
    assert key.grad.abs().sum() > 0  # K affects bias through contextual gates


def test_integer_position_interpolation_keeps_the_linear_gradient():
    cope = CoPE(1, 4, dtype=torch.float64)
    with torch.no_grad():
        cope.position_embeddings.copy_(torch.arange(4, dtype=torch.float64)[None])
    query = torch.ones(1, 1, 1, dtype=torch.float64)
    positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    # With a linear embedding table, bias == position even at integer knots.
    assert torch.autograd.gradcheck(
        lambda p: cope.positional_bias(query, p), (positions,)
    )
    cope.positional_bias(query, positions).sum().backward()
    torch.testing.assert_close(positions.grad, torch.ones_like(positions))
    edge = torch.tensor([[[3.0]]], dtype=torch.float64, requires_grad=True)
    cope.positional_bias(query, edge).sum().backward()
    assert edge.grad.item() == 0  # The final table entry has no right neighbour.


def test_integer_contextual_count_backpropagates_only_through_causal_keys():
    cope = CoPE(1, 4, dtype=torch.float64)
    with torch.no_grad():
        cope.position_embeddings.copy_(torch.arange(4, dtype=torch.float64)[None])
    query = torch.ones(1, 1, 1, dtype=torch.float64)
    key = torch.zeros(3, 1, 1, dtype=torch.float64, requires_grad=True)
    bias = cope(query, key, torch.tensor([1]))
    # Two causal sigmoid(0) gates yield exactly p=1; the future gate is masked.
    assert bias[0, 0, 0].item() == 1
    bias[0, 0, 0].backward()
    torch.testing.assert_close(
        key.grad[:, 0, 0], torch.tensor([0.25, 0.25, 0.0], dtype=torch.float64)
    )


def dense_reference(q, k, v, cope, q_positions, k_positions):
    outputs = []
    for qi, pos in enumerate(q_positions):
        heads = []
        for hi in range(q.shape[1]):
            kv_head = hi // (q.shape[1] // k.shape[1])
            logits = (q[qi, hi] @ k[:, kv_head].T) / q.shape[-1] ** 0.5
            mask = k_positions <= pos
            gate = logits.masked_fill(~mask, -torch.inf).sigmoid()
            positions = gate.flip(0).cumsum(0).flip(0).clamp(0, cope.max_positions - 1)
            table = q[qi, hi] @ cope.position_embeddings
            floor, ceil = positions.floor().long(), positions.ceil().long()
            bias = table[floor] + (positions - floor) * (table[ceil] - table[floor])
            probability = (logits + bias).masked_fill(~mask, -torch.inf).softmax(0)
            heads.append(probability @ v[:, kv_head])
        outputs.append(torch.stack(heads))
    return torch.stack(outputs)


def test_gqa_chunking_arbitrary_selected_rows_and_decode():
    torch.manual_seed(3)
    cope = CoPE(3, 7, dtype=torch.float64)
    with torch.no_grad():
        cope.position_embeddings.normal_()
    q = torch.randn(3, 4, 3, dtype=torch.float64)
    k = torch.randn(6, 2, 3, dtype=torch.float64)
    v = torch.randn(6, 2, 5, dtype=torch.float64)
    q_positions, k_positions = torch.tensor([4, 1, 5]), torch.arange(6)
    expected = dense_reference(q, k, v, cope, q_positions, k_positions)
    actual, positions = cope_attention(
        q, k, v, cope, q_positions, query_chunk_size=1, return_positions=True
    )
    torch.testing.assert_close(actual, expected)
    assert positions.shape == (4, 3, 6)
    decode = cope_attention(q[-1:], k, v, cope, torch.tensor([5]))
    torch.testing.assert_close(decode, expected[-1:])


def test_future_keys_values_do_not_change_earlier_output():
    torch.manual_seed(8)
    cope = CoPE(2, 6)
    with torch.no_grad():
        cope.position_embeddings.normal_()
    q, k, v = [torch.randn(4, 2, 2) for _ in range(3)]
    original = cope_attention(q, k, v, cope, torch.arange(4))
    k[2:] *= -100
    v[2:] += 100
    changed = cope_attention(q, k, v, cope, torch.arange(4))
    torch.testing.assert_close(changed[:2], original[:2])


def test_all_masked_rows_zero_and_fixed_trace_validation():
    cope = CoPE(2, 4)
    q, k, v = [torch.ones(2, 1, 2) for _ in range(3)]
    mask = torch.tensor([[False, False], [True, True]])
    output = cope_attention(q, k, v, cope, torch.arange(2), attention_mask=mask)
    assert output[0].abs().sum() == 0
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="query-dependent"):
        cope_attention(q, k, v, cope, torch.arange(2), fixed_positions=torch.arange(2))
    with pytest.raises(ValueError, match="strictly increasing"):
        cope(q, k, torch.arange(2), torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="finite"):
        cope(q * torch.nan, k, torch.arange(2))


def test_association_visibility_preserves_full_contextual_gate_counts():
    cope = CoPE(1, 8)
    q, k = torch.ones(1, 1, 1), torch.zeros(3, 1, 1)
    v = torch.tensor([1.0, 10.0, 100.0])[:, None, None]
    visibility = torch.tensor([[True, False, True]])
    output, trace = cope_attention(
        q,
        k,
        v,
        cope,
        torch.tensor([2]),
        visibility_mask=visibility,
        return_positions=True,
    )
    assert output.item() == 50.5
    torch.testing.assert_close(trace, torch.tensor([[[1.5, 1.0, 0.5]]]))
    _, structurally_masked_trace = cope_attention(
        q,
        k,
        v,
        cope,
        torch.tensor([2]),
        attention_mask=visibility,
        return_positions=True,
    )
    assert structurally_masked_trace[0, 0, 0] == 1


def make_trace(value=0.0):
    cope = CoPE(1, 8)
    q = torch.ones(3, 1, 1)
    k = torch.full((3, 1, 1), float(value))
    trace = cope.position_trace(q, k, torch.arange(3), checkpoint_id="trained-cope-v1")
    return trace


def test_profile_joint_histogram_keeps_observed_trace_and_strict_identity():
    traces = [make_trace(0), make_trace(0.7), make_trace(0.7)]
    chunks = [
        ChunkIdentity("system", "a" * 64, 1),
        ChunkIdentity("context", "b" * 64, 2),
    ]
    profile = CCPEProfile.calibrate(traces, chunks, trained_profile_version="task-v1")
    torch.testing.assert_close(profile.canonical_positions, traces[1].positions)
    selected = profile.lookup(
        chunks,
        torch.tensor([2, 0]),
        checkpoint_id="trained-cope-v1",
        trained_profile_version="task-v1",
    )
    torch.testing.assert_close(selected, traces[1].positions[:, [2, 0]])
    with pytest.raises(ValueError, match="identities"):
        profile.lookup(
            chunks[::-1],
            torch.tensor([0]),
            checkpoint_id="trained-cope-v1",
            trained_profile_version="task-v1",
        )
    assert (
        profile.lookup(
            chunks,
            torch.tensor([3]),
            checkpoint_id="trained-cope-v1",
            trained_profile_version="task-v1",
            strict=False,
        )
        is None
    )
    with pytest.raises(ValueError, match="version"):
        profile.lookup(
            chunks,
            torch.tensor([0]),
            checkpoint_id="untrained",
            trained_profile_version="task-v1",
        )


def test_projected_fixed_ordinals_preserve_dynamic_context_gates():
    trace = make_trace(0)
    fixed = torch.tensor([0, 2])
    projected = trace.project(fixed, fixed)
    assert projected.query_positions.tolist() == [0, 1]
    assert projected.key_positions.tolist() == [0, 1]
    assert projected.positions[0, 1, 0].item() == 1.5  # dynamic middle token counted
    chunks = [ChunkIdentity("fixed", "c" * 64, 2)]
    profile = CCPEProfile.calibrate([projected], chunks, trained_profile_version="v1")
    result = profile.lookup(
        chunks,
        torch.tensor([1]),
        checkpoint_id="trained-cope-v1",
        trained_profile_version="v1",
    )
    assert result.shape == (1, 1, 2)
    assert result[0, 0, 0].item() == 1.5


def test_profile_rejects_fabricated_positions_and_oversized_trace():
    trace = make_trace()
    chunks = [ChunkIdentity("fixed", "c" * 64, 3)]
    with pytest.raises(ValueError, match="CoPE gates"):
        CCPEProfile.calibrate(
            [replace(trace, positions=trace.positions + 1)],
            chunks,
            trained_profile_version="v1",
        )
    with pytest.raises(ValueError, match="max_elements"):
        CCPEProfile.calibrate(
            [trace], chunks, trained_profile_version="v1", max_elements=2
        )
    with pytest.raises(ValueError, match="genuine"):
        CCPEProfile.calibrate([torch.arange(3)], chunks, trained_profile_version="v1")
    with pytest.raises(ValueError, match="max_elements"):
        CoPE(1, 4).position_trace(
            torch.ones(3, 1, 1),
            torch.ones(3, 1, 1),
            torch.arange(3),
            checkpoint_id="trained",
            max_elements=5,
        )


def test_joint_histogram_tie_is_order_independent_and_uses_real_pattern():
    cope = CoPE(1, 8)
    q = torch.ones(3, 1, 1)
    traces = [
        cope.position_trace(
            q,
            torch.tensor(values)[:, None, None],
            torch.arange(3),
            checkpoint_id="trained",
        )
        for values in ([0.0, 0.0, 3.0], [0.0, 3.0, 0.0], [3.0, 0.0, 0.0])
    ]
    chunks = [ChunkIdentity("fixed", "d" * 64, 3)]
    first = CCPEProfile.calibrate(traces, chunks, trained_profile_version="v1")
    reversed_samples = CCPEProfile.calibrate(
        traces[::-1], chunks, trained_profile_version="v1"
    )
    torch.testing.assert_close(
        first.canonical_positions, reversed_samples.canonical_positions
    )
    assert any(
        torch.equal(first.canonical_positions, trace.positions) for trace in traces
    )
    independent_modes = (
        torch.stack([trace.positions for trace in traces]).mode(0).values
    )
    assert not any(torch.equal(independent_modes, trace.positions) for trace in traces)
