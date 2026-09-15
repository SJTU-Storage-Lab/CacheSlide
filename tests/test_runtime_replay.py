"""A preempted native request must rebuild prompt + generated suffix correctly."""

from dataclasses import replace

import pytest
import torch

from cacheslide_core.runtime import CacheSlideRuntime
from cacheslide_vllm.integration import StepContext
from tests.test_runtime import build_runtime, decode, plan, prefill
from tests.test_runtime_paged import CPUBlockPool, bind


@torch.inference_mode()
def replay(model, runtime, request, suffix, request_id="r"):
    ids = (*request.token_ids, *suffix)
    positions = torch.arange(len(ids))
    step = StepContext(
        request_id,
        request.token_ids,
        tuple(positions.tolist()),
        {"cacheslide": request.to_json()},
        replay_token_ids=ids,
    )
    hidden = runtime.run(model, model.embed_tokens(torch.tensor(ids)), positions, step)
    return model.lm_head(hidden)[0]


@pytest.mark.parametrize("paged", [False, True])
def test_preempted_reuse_dense_replay_preserves_profiles_and_decode(tmp_path, paged):
    model, runtime = build_runtime(tmp_path)
    oracle = CacheSlideRuntime(
        replace(runtime.settings, cache_root=str(tmp_path / "oracle")),
        runtime.bundle,
        model_dtype=torch.float32,
    )
    if paged:
        runtime.arena_factory = CPUBlockPool(runtime)
    request = plan((4, 10), operation="reuse")
    suffix = (3, 4)
    complete = (*request.token_ids, *suffix)
    dense_plan = replace(
        request,
        operation="recompute",
        token_ids=complete,
        chunks=(
            *request.chunks[:-1],
            replace(request.chunks[-1], end=len(complete)),
        ),
    )
    try:
        prefill(model, runtime, plan(operation="populate"))
        prefill(model, runtime, request)
        old_arenas = list(runtime.requests["r"].arenas.values())
        actual = replay(model, runtime, request, suffix)
        metrics = runtime.last_metrics
        assert metrics["replayed_tokens"] == metrics["decode_tokens"] == 2
        assert metrics["prompt_tokens"] == len(request.token_ids)
        assert not metrics["cache_hit"] and metrics["fallback"]
        assert metrics["layer_rows"] == [len(complete)] * len(model.layers)
        assert metrics["snapshot_bytes_written"] == 0
        assert runtime.requests["r"].plan == request
        if paged:
            for old in old_arenas:
                with pytest.raises(RuntimeError):
                    old.gather([0])
        bind(model, oracle)
        expected = prefill(model, oracle, dense_plan)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for index, token in enumerate((5, 6)):
            position = len(complete) + index
            bind(model, runtime)
            actual = decode(model, runtime, request, token, position)
            bind(model, oracle)
            expected = decode(model, oracle, dense_plan, token, position)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        runtime.close()
        oracle.close()


def test_replay_budget_includes_generated_suffix_and_keeps_live_state(tmp_path):
    model, runtime = build_runtime(tmp_path, max_prompt_tokens=9)
    request = plan()
    try:
        prefill(model, runtime, request)
        original = runtime.requests["r"]
        with pytest.raises(ValueError, match="token budget"):
            replay(model, runtime, request, (3, 4))
        assert runtime.requests["r"] is original
    finally:
        runtime.close()
