"""Self-contained random tiny CPU fixture; not a model-quality/performance claim."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from cacheslide_core.artifacts import AdapterBundle
from cacheslide_core.config import CacheSlideSettings
from cacheslide_core.fixtures import tiny_checkpoint as tiny_checkpoint
from cacheslide_core.fixtures import tiny_plan as tiny_plan
from cacheslide_core.reference import ReferenceLlama
from cacheslide_core.runtime import CacheSlideRuntime

from .integration import StepContext
from .paged import NativePagedKV


class _CPUBlockPool:
    def __init__(self, config: dict, capacity: int):
        blocks = (capacity + 1) // 2
        self.row = list(reversed(range(blocks)))
        self.caches = [
            # Exercise real strides, not a dense stand-in for the page adapter.
            torch.full(
                (blocks, config["num_key_value_heads"], 2, 4 * config["head_dim"]),
                -999.0,
            )[..., ::2]
            for _ in range(config["num_hidden_layers"])
        ]

    def __call__(self, layer, prompt_length, max_selected, *, existing=None):
        if existing is not None:
            existing.update_block_table(self.row)
            return existing
        return NativePagedKV(
            self.caches[layer],
            self.row,
            prompt_length,
            max_selected,
            request_id="cpu-smoke",
            layer_index=layer,
        )


def run_smoke(args, root: Path, manifest: dict, run_stage) -> dict:
    from .workflow import WorkflowError, _options, _save

    print(
        "CPU smoke: random tiny weights; no GPU benchmark or quality claim.",
        file=sys.stderr,
        flush=True,
    )
    model_path, adapter_path = root / "tiny-model", root / "adapter"
    profile_path, input_path = root / "profiles", root / "cases.jsonl"
    tiny_checkpoint(model_path)
    plans = [tiny_plan(), tiny_plan((4, 10))]
    with input_path.open("x") as stream:
        for index, plan in enumerate(plans):
            stream.write(
                json.dumps(
                    {
                        "id": f"tiny-{index}",
                        "prompt_token_ids": list(plan.token_ids),
                        "cacheslide": json.loads(plan.to_json()),
                    }
                )
                + "\n"
            )
    run_stage(
        "train",
        [
            "train",
            *_options(
                model=model_path,
                input=input_path,
                output=adapter_path,
                steps=args.steps,
                lr=args.lr,
                rank=args.rank,
                max_positions=args.max_positions,
                query_chunk_size=args.query_chunk_size,
                device="cpu",
            ),
        ],
        root,
        manifest,
        module="cacheslide_vllm.workflow_smoke",
    )
    run_stage(
        "calibrate",
        [
            "calibrate",
            *_options(
                model=model_path,
                adapter=adapter_path,
                input=input_path,
                output=profile_path,
                device="cpu",
                max_elements=args.max_profile_elements,
                profile_version=args.profile_version,
            ),
        ],
        root,
        manifest,
    )
    bundle = AdapterBundle(adapter_path, model_path)
    model = ReferenceLlama.from_checkpoint(
        model_path,
        rank=bundle.metadata["rank"],
        max_positions=bundle.metadata["max_positions"],
        device="cpu",
        query_chunk_size=args.query_chunk_size,
    )
    model.load_adapters(bundle)
    runtime = CacheSlideRuntime(
        CacheSlideSettings(
            artifact_path=str(adapter_path),
            profile_path=str(profile_path),
            cache_root=args.cache_root,
            query_chunk_size=args.query_chunk_size,
            max_prompt_tokens=args.max_model_len,
            max_profile_elements=args.max_profile_elements,
            cpu_budget_bytes=args.cpu_budget_bytes,
            disk_budget_bytes=args.disk_budget_bytes,
            calibration_layer=args.calibration_layer,
            correction_fraction=args.correction_fraction,
            convergence_mode=args.convergence_mode,
            weight_update=args.weight_update,
            selected_attention=args.selected_attention,
            ccpe_position_policy=args.ccpe_position_policy,
        ),
        bundle,
        model_dtype=torch.float32,
    )
    runtime.arena_factory = _CPUBlockPool(bundle.config, 9 + args.max_tokens)
    for layer in model.layers:
        layer.self_attn.attention_handler = runtime.attention
    records = []

    @torch.inference_mode()
    def invoke(plan, operation, phase):
        request = replace(plan, operation=operation)
        request_id = f"smoke-{len(records)}"
        ids = torch.tensor(request.token_ids, dtype=torch.long)
        positions = torch.arange(len(ids))
        metadata = {"cacheslide": request.to_json()}
        step = StepContext(
            request_id, request.token_ids, tuple(positions.tolist()), metadata
        )
        try:
            hidden = runtime.run(model, model.embed_tokens(ids), positions, step)
            logits, _ = model.lm_head(hidden)
            first_logits = logits[-1].clone()
            generated = [int(first_logits.argmax())]
            for offset in range(args.max_tokens - 1):
                position = len(ids) + offset
                step = StepContext(request_id, request.token_ids, (position,), metadata)
                hidden = runtime.run(
                    model,
                    model.embed_tokens(torch.tensor([generated[-1]])),
                    torch.tensor([position]),
                    step,
                )
                logits, _ = model.lm_head(hidden)
                if not torch.isfinite(logits).all():
                    raise WorkflowError("CPU smoke produced nonfinite decode logits")
                generated.append(int(logits[-1].argmax()))
            if not torch.isfinite(first_logits).all():
                raise WorkflowError("CPU smoke produced nonfinite prefill logits")
            metrics = dict(runtime.last_metrics)
            guarded = (
                args.ccpe_position_policy == "strict_contextual"
                and metrics["position_policy_fallback"]
            )
            if guarded:
                expected = model(torch.tensor(request.token_ids))[-1]
                torch.testing.assert_close(first_logits, expected, atol=3e-6, rtol=3e-6)
            if operation == "reuse" and not (
                (metrics["cache_hit"] and not metrics["fallback"]) or guarded
            ):
                raise WorkflowError(
                    "CPU smoke reuse missed without verified safety fallback"
                )
            records.append(
                {
                    "request_id": request_id,
                    "operation": operation,
                    "phase": phase,
                    "prompt_tokens": len(ids),
                    "generated_token_ids": generated,
                    "runtime_metrics": metrics,
                }
            )
            return first_logits
        finally:
            runtime.release(request_id)

    try:
        invoke(plans[0], "populate", "setup")
        for _ in range(args.warmup):
            invoke(plans[0], "recompute", "warmup")
            invoke(plans[0], "reuse", "warmup")
        for _ in range(args.repeats):
            baseline = invoke(plans[0], "recompute", "correctness")
            reused = invoke(plans[0], "reuse", "correctness")
            torch.testing.assert_close(reused, baseline, atol=3e-6, rtol=3e-6)
            invoke(plans[1], "recompute", "correctness")
            invoke(plans[1], "reuse", "correctness")
    finally:
        runtime.close()
    reused = [record for record in records if record["operation"] == "reuse"]
    hits = [record for record in reused if record["runtime_metrics"]["cache_hit"]]
    if not all(
        record["runtime_metrics"]["computed_token_layers"]
        < record["runtime_metrics"]["dense_token_layers"]
        for record in hits
    ):
        raise WorkflowError("CPU smoke did not demonstrate selective token-layer work")
    with (root / "raw_outputs.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    summary = {
        "executed": True,
        "mode": "cpu_smoke",
        "correctness_passed": True,
        "device": "cpu",
        "native_vllm_executed": False,
        "fixture": "random tiny six-layer GQA safetensors; not a language model",
        "checkpoint_seed": 92,
        "adapter_seed": 9,
        "training_steps": bundle.metadata["training_steps"],
        "adapter_identity": bundle.identity,
        "requests": len(records),
        "reuse_requests": len(reused),
        "shifted_nonprefix_reuse_passed": len(hits) == len(reused),
        "cache_hit_requests": len(hits),
        "ccpe_position_policy": args.ccpe_position_policy,
        "calibration_layer": args.calibration_layer,
        "guarded_fallback_requests": sum(
            record["runtime_metrics"]["position_policy_fallback"] for record in records
        ),
        "unchanged_prefill_logits_close": True,
        "max_tokens": args.max_tokens,
        "decode_steps_per_request": args.max_tokens - 1,
        "packed_page_adapter_exercised": True,
        "gpu_performance_claim": False,
        "model_quality_claim": False,
        "timing_kind": "not_measured",
        "baseline_over_reuse_latency_ratio": None,
    }
    _save(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    # Reuse the actual CLI training path, but fix initialization for this fixture.
    from .cli import main

    torch.manual_seed(9)
    torch.set_num_threads(1)
    raise SystemExit(main())
