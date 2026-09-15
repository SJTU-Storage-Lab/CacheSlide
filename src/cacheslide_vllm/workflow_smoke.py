"""Self-contained random tiny CPU fixture; not a model-quality/performance claim."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from .artifacts import AdapterBundle
from .config import CacheSlideSettings
from .contracts import RequestPlan
from .integration import StepContext
from .paged import NativePagedKV
from .reference import ReferenceLlama
from .runtime import CacheSlideRuntime


def tiny_checkpoint(path: Path) -> None:
    """Write a local six-layer GQA safetensors checkpoint, with no network access."""
    path.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }
    generator = torch.Generator(device="cpu").manual_seed(92)

    def random(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    weights = {
        "model.embed_tokens.weight": random(16, 8),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": random(16, 8),
    }
    for layer in range(6):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = torch.ones(8)
        for name, shape in {
            "self_attn.q_proj": (8, 8),
            "self_attn.k_proj": (4, 8),
            "self_attn.v_proj": (4, 8),
            "self_attn.o_proj": (8, 8),
            "mlp.gate_proj": (12, 8),
            "mlp.up_proj": (12, 8),
            "mlp.down_proj": (8, 12),
        }.items():
            weights[prefix + name + ".weight"] = random(*shape)
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(path / "model.safetensors"))


def tiny_plan(dynamic=(3,)) -> RequestPlan:
    tokens = (1, 2, *dynamic, 5, 6, 7, 8, 9)
    split = 2 + len(dynamic)
    return RequestPlan.parse(
        {
            "version": 1,
            "operation": "recompute",
            "namespace": "cpu-smoke",
            "task_id": "synthetic-only",
            "chunks": [
                {"id": "A", "role": "reuse", "start": 0, "end": 2},
                {"id": "dynamic", "role": "recompute", "start": 2, "end": split},
                {"id": "B", "role": "reuse", "start": split, "end": split + 4},
                {
                    "id": "query",
                    "role": "recompute",
                    "start": split + 4,
                    "end": len(tokens),
                },
            ],
        },
        tokens,
    )


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
            if operation == "reuse" and not (
                metrics["cache_hit"] and not metrics["fallback"]
            ):
                raise WorkflowError("CPU smoke reuse missed or fell back")
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
    if not all(
        record["runtime_metrics"]["computed_token_layers"]
        < record["runtime_metrics"]["dense_token_layers"]
        for record in reused
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
        "shifted_nonprefix_reuse_passed": True,
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
