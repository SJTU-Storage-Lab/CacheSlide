"""Actual shared runtime + native-layout SGLang token arena on a tiny CPU model."""

from __future__ import annotations

import json
from dataclasses import replace

import torch

from cacheslide_core.artifacts import AdapterBundle
from cacheslide_core.context import StepContext
from cacheslide_core.fixtures import tiny_checkpoint, tiny_plan
from cacheslide_core.reference import ReferenceLlama
from cacheslide_core.runtime import CacheSlideRuntime

from .pool import SGLangTokenKV


def prepare_fixture(args, root):
    if args.train_device != "cpu" or args.calibration_device != "cpu":
        raise ValueError("--smoke is CPU-only; use native mode for GPU experiments")
    if args.context_length < 9 + args.max_tokens:
        raise ValueError("context-length is too small for the tiny smoke fixture")
    args.model = str(root / "tiny-model")
    args.input = str(root / "cases.jsonl")
    tiny_checkpoint(root / "tiny-model")
    with (root / "cases.jsonl").open("x") as stream:
        for i, plan in enumerate((tiny_plan(), tiny_plan((4, 10)))):
            stream.write(
                json.dumps(
                    {
                        "id": f"tiny-{i}",
                        "prompt_token_ids": list(plan.token_ids),
                        "cacheslide": json.loads(plan.to_json()),
                    }
                )
                + "\n"
            )


class CPUArenaFactory:
    """Use separate strided K/V tensors and a permuted scheduler-owned slot row."""

    def __init__(self, config, capacity):
        self.row = tuple(reversed(range(1, capacity + 1)))
        self.buffers = [
            tuple(
                torch.full(
                    (
                        capacity + 1,
                        config["num_key_value_heads"],
                        config["head_dim"] * 2,
                    ),
                    -999.0,
                )[..., ::2]
                for _ in range(2)
            )
            for _ in range(config["num_hidden_layers"])
        ]

    def __call__(self, layer, prompt_length, max_selected, *, existing=None):
        if existing is not None:
            existing.update_slot_mapping(self.row)
            return existing
        return SGLangTokenKV(
            *self.buffers[layer],
            self.row,
            prompt_length,
            max_selected,
            request_id="cpu-fixture",
            layer_index=layer,
        )


@torch.inference_mode()
def run_smoke(args, settings):
    bundle = AdapterBundle(args.adapter, args.model)
    model = ReferenceLlama.from_checkpoint(
        args.model,
        rank=bundle.metadata["rank"],
        max_positions=bundle.metadata["max_positions"],
        query_chunk_size=args.query_chunk_size,
    )
    model.load_adapters(bundle)
    paged = CacheSlideRuntime(settings, bundle, model_dtype=torch.float32)
    dense = CacheSlideRuntime(
        replace(settings, cache_root=settings.cache_root + "-oracle"),
        bundle,
        model_dtype=torch.float32,
    )
    paged.arena_factory = CPUArenaFactory(bundle.config, 9 + args.max_tokens)
    records = []

    def bind(runtime):
        for layer in model.layers:
            layer.self_attn.attention_handler = runtime.attention

    def forward(runtime, plan, ids, positions, rid):
        bind(runtime)
        step = StepContext(
            rid, plan.token_ids, tuple(positions), {"cacheslide": plan.to_json()}
        )
        hidden = runtime.run(
            model, model.embed_tokens(torch.tensor(ids)), torch.tensor(positions), step
        )
        return model.lm_head(hidden)[0]

    def invoke(plan, operation):
        request = replace(plan, operation=operation)
        rid = f"smoke-{len(records)}"
        pos = list(range(len(request.token_ids)))
        expected = forward(dense, request, request.token_ids, pos, rid)
        actual = forward(paged, request, request.token_ids, pos, rid)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        generated = [int(actual[-1].argmax())]
        for offset in range(args.max_tokens - 1):
            pos = [len(request.token_ids) + offset]
            expected = forward(dense, request, [generated[-1]], pos, rid)
            actual = forward(paged, request, [generated[-1]], pos, rid)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            generated.append(int(actual[-1].argmax()))
        for layer, arena in paged.requests[rid].arenas.items():
            size = len(request.token_ids) + args.max_tokens - 1
            actual_k, actual_v = arena.gather(range(size))
            expected_k, expected_v = dense.requests[rid].dense_kv[layer]
            torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
            torch.testing.assert_close(actual_v, expected_v, atol=0, rtol=0)
        metrics = dict(paged.last_metrics)
        if metrics["position_policy_fallback"]:
            # Independent plain-CoPE check, not just equality of two retry paths.
            bind(paged)
            complete = [*request.token_ids, *generated[:-1]]
            plain = model(torch.tensor(complete))[-1]
            torch.testing.assert_close(actual[-1], plain, atol=3e-6, rtol=3e-6)
        elif operation == "reuse" and (not metrics["cache_hit"] or metrics["fallback"]):
            raise RuntimeError(
                "smoke reuse did not hit or take the verified CCPE guard"
            )
        records.append(
            {
                "request_id": rid,
                "operation": operation,
                "prompt_tokens": len(request.token_ids),
                "output_ids": generated,
                "runtime_metrics": metrics,
            }
        )
        dense.release(rid)
        paged.release(rid)

    try:
        invoke(tiny_plan(), "populate")
        for _ in range(args.warmup + args.repeats):
            for plan in (tiny_plan(), tiny_plan((4, 10))):
                invoke(plan, "recompute")
                invoke(plan, "reuse")
    finally:
        dense.close()
        paged.close()
    reused = [row for row in records if row["operation"] == "reuse"]
    hits = sum(row["runtime_metrics"]["cache_hit"] for row in reused)
    summary = {
        "executed": True,
        "mode": "cpu_smoke",
        "correctness_passed": True,
        "native_sglang_executed": False,
        "device": "cpu",
        "fixture": "random six-layer GQA; not a trained language model",
        "requests": len(records),
        "reuse_requests": len(reused),
        "cache_hit_requests": hits,
        "shifted_nonprefix_reuse_passed": hits == len(reused),
        "guarded_fallback_requests": sum(
            r["runtime_metrics"]["position_policy_fallback"] for r in records
        ),
        "ccpe_position_policy": args.ccpe_position_policy,
        "calibration_layer": args.calibration_layer,
        "native_nhd_token_arena_exercised": True,
        "dense_and_token_arena_logits_close": True,
        "dense_and_token_arena_kv_close": True,
        "decode_steps_per_request": args.max_tokens - 1,
        "training_steps": bundle.metadata["training_steps"],
        "timing_kind": "not_measured",
        "baseline_over_reuse_latency_ratio": None,
        "gpu_performance_claim": False,
        "model_quality_claim": False,
    }
    return records, summary
