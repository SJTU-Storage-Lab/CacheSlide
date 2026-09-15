"""Explicit local training, calibration and pinned native-engine workflows."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .compat import (
    CompatibilityError,
    compatibility_manifest,
    verify_installed_vllm,
    verify_vllm_sources,
)
from .config import CacheSlideSettings
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


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def _engine_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="local safetensors backbone")
    parser.add_argument(
        "--adapter", required=True, help="verified trained adapter directory"
    )
    parser.add_argument("--profiles", help="optional calibrated CCPE profile directory")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--input", required=True, help="typed prompt/metadata JSONL")
    parser.add_argument("--output", required=True, help="new local output directory")
    parser.add_argument("--max-model-len", type=_positive, default=4096)
    parser.add_argument("--max-tokens", type=_positive, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--model-runner",
        choices=("v2", "v1"),
        default="v2",
        help="native vLLM runner (v2 is the 0.29.0 default; v1 is legacy opt-in)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--cpu-budget-bytes", type=_positive, default=536_870_912)
    parser.add_argument("--disk-budget-bytes", type=_positive, default=2_147_483_648)
    parser.add_argument("--max-profile-elements", type=_positive, default=4_194_304)
    parser.add_argument("--query-chunk-size", type=_positive, default=64)
    parser.add_argument("--calibration-layer", type=_nonnegative, default=0)
    parser.add_argument(
        "--ccpe-position-policy",
        choices=("strict_contextual", "mixed_bias_override"),
        default="strict_contextual",
    )
    parser.add_argument("--correction-fraction", type=float, default=0.26)
    parser.add_argument(
        "--convergence-mode",
        choices=("paper_cosine_lt", "distance_lt"),
        default="paper_cosine_lt",
    )
    parser.add_argument(
        "--weight-update",
        choices=("previous_layer", "same_layer"),
        default="previous_layer",
    )
    parser.add_argument(
        "--selected-attention",
        choices=("updated_and_self", "full_causal"),
        default="updated_and_self",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly construct and run the native GPU engine",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cacheslide")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="continued causal CE training on token ids"
    )
    train.add_argument("--model", required=True)
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=_positive, required=True)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--rank", type=_positive, default=8)
    train.add_argument("--max-positions", type=_positive, default=256)
    train.add_argument("--query-chunk-size", type=_positive, default=128)
    train.add_argument("--device", default="cpu")
    calibrate = commands.add_parser(
        "calibrate", help="calibrate genuine contextual traces"
    )
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--adapter", required=True)
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--device", default="cpu")
    calibrate.add_argument("--max-elements", type=_positive, default=1_000_000)
    calibrate.add_argument("--profile-version", default="v1")
    inspect = commands.add_parser(
        "inspect", help="read-only artifact/contract inspection"
    )
    inspect.add_argument("--model")
    inspect.add_argument("--adapter")
    inspect.add_argument("--profiles")
    check = commands.add_parser(
        "check-engine", help="verify audited source hashes without CUDA"
    )
    check.add_argument("--source-root")
    check.add_argument("--version", default="0.29.0")
    generate = commands.add_parser(
        "generate", help="explicit native token-id generation"
    )
    _engine_options(generate)
    bench = commands.add_parser("bench", help="aligned offline recompute/reuse latency")
    _engine_options(bench)
    bench.add_argument(
        "--seed-input", help="populate seed JSONL; defaults to first per layout"
    )
    bench.add_argument("--warmup", type=_nonnegative, default=1)
    bench.add_argument("--repeats", type=_positive, default=3)
    bench.add_argument("--backend", choices=("native", "reference"), default="native")
    bench.add_argument(
        "--token-f1",
        action="store_true",
        help="also report token-id multiset F1; not semantic answer accuracy",
    )
    return root


def engine_kwargs(args: argparse.Namespace) -> dict:
    """Pure configuration construction: this function never imports vLLM/CUDA."""
    if not math.isfinite(args.gpu_memory_utilization) or not (
        0 < args.gpu_memory_utilization <= 1
    ):
        raise ValueError("gpu-memory-utilization must be in (0,1]")
    settings = {
        "artifact_path": str(Path(args.adapter).resolve()),
        "cache_root": str(Path(args.cache_root).resolve()),
        "cpu_budget_bytes": args.cpu_budget_bytes,
        "disk_budget_bytes": args.disk_budget_bytes,
        "max_prompt_tokens": args.max_model_len,
        "max_profile_elements": args.max_profile_elements,
        "query_chunk_size": args.query_chunk_size,
        "calibration_layer": args.calibration_layer,
        "correction_fraction": args.correction_fraction,
        "convergence_mode": args.convergence_mode,
        "weight_update": args.weight_update,
        "selected_attention": args.selected_attention,
        "ccpe_position_policy": args.ccpe_position_policy,
    }
    if args.profiles:
        settings["profile_path"] = str(Path(args.profiles).resolve())
    settings = asdict(CacheSlideSettings.from_mapping(settings))
    return {
        "model": str(Path(args.model).resolve()),
        "skip_tokenizer_init": True,
        "trust_remote_code": False,
        "load_format": "safetensors",
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
        "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "kv_cache_dtype": "auto",
        "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
        "attention_config": {"backend": "FLASH_ATTN"},
        "worker_cls": "cacheslide_vllm.worker.CacheSlideWorker",
        "hf_overrides": {"architectures": ["CacheSlideLlamaForCausalLM"]},
        "additional_config": {"cacheslide": settings},
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 0,
        "generation_config": "vllm",
        "disable_log_stats": True,
    }


def create_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    if not args.run:
        raise ValueError("native engine construction requires explicit --run")
    kwargs = engine_kwargs(args)
    verify_installed_vllm()
    from .artifacts import AdapterBundle
    from .profiles import ProfileBundle

    adapter = AdapterBundle(args.adapter, args.model)
    if args.profiles:
        ProfileBundle(
            args.profiles, adapter.identity, max_elements=args.max_profile_elements
        )
    # Must be set before importing/constructing the engine and its worker processes.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
        "1" if getattr(args, "model_runner", "v2") == "v2" else "0"
    )
    from vllm import LLM, SamplingParams

    from .plugin import register

    register()
    return LLM(**kwargs), SamplingParams


@contextmanager
def engine_session(args: argparse.Namespace):
    engine, sampling = create_engine(args)
    try:
        yield engine, sampling
    finally:
        # The public worker RPC drains only the CacheSlide store. Native engine
        # shutdown retains ownership of its GPU resources.
        engine.collective_rpc("cacheslide_close")


def generate_one(
    engine: Any,
    sampling_class: Any,
    case: InputCase,
    *,
    operation: str,
    max_tokens: int,
) -> dict:
    if case.plan is None:
        raise ValueError("native generation requires validated chunk metadata")
    plan = replace(case.plan, operation=operation)
    sampling = sampling_class(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=True,
        detokenize=False,
        extra_args={"cacheslide": json.loads(plan.to_json())},
    )
    start = time.perf_counter()
    results = engine.generate(
        {"prompt_token_ids": list(case.token_ids)},
        sampling_params=sampling,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - start
    if len(results) != 1 or len(results[0].outputs) != 1:
        raise RuntimeError("native benchmark requires exactly one completion per case")
    completion = results[0].outputs[0]
    generated = list(completion.token_ids)
    receipt, receipt_error = {}, None
    request_id = str(results[0].request_id)
    try:
        receipts = engine.collective_rpc("cacheslide_metrics")
        if len(receipts) != 1 or not isinstance(receipts[0], dict):
            raise ValueError("expected exactly one worker metrics receipt")
        receipt = receipts[0]
        if (
            receipt.get("request_id") != request_id
            or receipt.get("operation") != operation
            or receipt.get("prompt_tokens") != len(case.token_ids)
        ):
            raise ValueError("worker receipt does not match this generated request")
    except Exception as error:
        receipt_error = str(error)
    return {
        "case_id": case.case_id,
        "operation": operation,
        "prompt_token_ids": list(case.token_ids),
        "generated_token_ids": generated,
        "input_tokens": len(case.token_ids),
        "output_tokens": len(generated),
        "elapsed_seconds": elapsed,
        "request_id": request_id,
        "finish_reason": completion.finish_reason,
        "runtime_metrics": receipt,
        "metrics_valid": receipt_error is None,
        "metrics_error": receipt_error,
    }


def _token_f1(first: list[int], second: list[int]) -> float:
    if not first and not second:
        return 1.0
    overlap = sum((Counter(first) & Counter(second)).values())
    return 2 * overlap / (len(first) + len(second))


def _validate_native_cases(args: argparse.Namespace, cases: list[InputCase]) -> None:
    if any(
        len(case.token_ids) + args.max_tokens > args.max_model_len for case in cases
    ):
        raise ValueError("prompt plus max-tokens exceeds max-model-len")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be a new directory")


def _write_results(output: str | Path, records: list[dict], summary: dict) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "raw_outputs.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _benchmark_runs(engine, sampling, args, cases, seeds):
    records = []

    def invoke(case, operation, phase, repeat):
        result = generate_one(
            engine, sampling, case, operation=operation, max_tokens=args.max_tokens
        )
        result.update(phase=phase, repeat=repeat)
        records.append(result)
        return result

    for seed in seeds.values():
        invoke(seed, "populate", "setup", 0)
    for repeat in range(args.warmup):
        for case in cases:
            invoke(case, "recompute", "warmup", repeat)
            invoke(case, "reuse", "warmup", repeat)
    pairs = []
    for repeat in range(args.repeats):
        for case in cases:
            operations = (
                ("recompute", "reuse") if repeat % 2 == 0 else ("reuse", "recompute")
            )
            results = {
                operation: invoke(case, operation, "measured", repeat)
                for operation in operations
            }
            baseline, reused = results["recompute"], results["reuse"]
            pair = {
                "case_id": case.case_id,
                "repeat": repeat,
                "input_tokens": len(case.token_ids),
                "baseline_output_tokens": baseline["output_tokens"],
                "reuse_output_tokens": reused["output_tokens"],
                "baseline_seconds": baseline["elapsed_seconds"],
                "reuse_seconds": reused["elapsed_seconds"],
                "baseline_metrics_valid": baseline["metrics_valid"],
                "baseline_fallback": baseline["runtime_metrics"].get("fallback"),
                "reuse_metrics_valid": reused["metrics_valid"],
                "reuse_cache_hit": reused["runtime_metrics"].get("cache_hit"),
                "reuse_fallback": reused["runtime_metrics"].get("fallback"),
                "exact_generated_token_agreement": baseline["generated_token_ids"]
                == reused["generated_token_ids"],
            }
            if args.token_f1:
                pair["token_id_multiset_f1"] = _token_f1(
                    baseline["generated_token_ids"], reused["generated_token_ids"]
                )
            pairs.append(pair)
    return records, pairs


def benchmark(args: argparse.Namespace, cases: list[InputCase]) -> dict:
    """One warm native engine, matched queries, setup/warmup excluded from means."""
    if args.backend != "native":
        raise ValueError(
            "the reference reuse benchmark is not connected; use --backend native"
        )
    _validate_native_cases(args, cases)
    supplied_seeds = read_cases(args.seed_input) if args.seed_input else cases
    seeds = {}
    for case in supplied_seeds:
        if case.plan is None or not case.plan.fixed_indices:
            raise ValueError("benchmark seed cases require reusable fixed chunks")
        seeds.setdefault(case.plan.cache_key("benchmark-layout", 0), case)
    for case in cases:
        if case.plan is None or case.plan.cache_key("benchmark-layout", 0) not in seeds:
            raise ValueError("each measured layout requires a matching populate seed")
    _validate_native_cases(args, list(seeds.values()))
    kwargs = engine_kwargs(args)
    if not args.run:
        return {
            "executed": False,
            "requires": "--run",
            "backend": "native",
            "model_runner": args.model_runner,
            "cases": len(cases),
            "populate_seeds": len(seeds),
            "engine": kwargs,
            "timing_kind": "offline generation; not streaming TTFT",
            "reuse_profile_configured": bool(args.profiles),
        }
    if not args.profiles:
        raise ValueError(
            "bench --run requires --profiles to validate actual cache reuse"
        )
    with engine_session(args) as (engine, sampling):
        records, pairs = _benchmark_runs(engine, sampling, args, cases, seeds)
    baseline_mean = statistics.mean(pair["baseline_seconds"] for pair in pairs)
    reuse_mean = statistics.mean(pair["reuse_seconds"] for pair in pairs)
    baseline_valid = all(
        pair["baseline_metrics_valid"] and pair["baseline_fallback"] is False
        for pair in pairs
    )
    reuse_valid = all(
        pair["reuse_metrics_valid"]
        and pair["reuse_cache_hit"] is True
        and pair["reuse_fallback"] is False
        for pair in pairs
    )
    latency_valid = all(
        math.isfinite(value) and value > 0 for value in (baseline_mean, reuse_mean)
    ) and all(
        pair["baseline_output_tokens"] == pair["reuse_output_tokens"] == args.max_tokens
        for pair in pairs
    )
    summary = {
        "executed": True,
        "backend": "native_vllm",
        "model_runner": args.model_runner,
        "engine": kwargs,
        "timing_kind": (
            "offline_prefill_plus_one_token_latency_seconds"
            if args.max_tokens == 1
            else "offline_generation_latency_seconds"
        ),
        "is_streaming_ttft": False,
        "comparison": "same trained CoPE/LoRA model: recompute versus CacheSlide reuse",
        "warmup_rounds_excluded": args.warmup,
        "populate_seeds_excluded": len(seeds),
        "repeats": args.repeats,
        "measured_pairs": len(pairs),
        "baseline_mean_seconds": baseline_mean,
        "reuse_mean_seconds": reuse_mean,
        "baseline_validation_passed": baseline_valid,
        "reuse_validation_passed": reuse_valid,
        "latency_validation_passed": latency_valid,
        "baseline_over_reuse_latency_ratio": baseline_mean / reuse_mean
        if baseline_valid and reuse_valid and latency_valid
        else None,
        "exact_generated_token_agreement_rate": statistics.mean(
            pair["exact_generated_token_agreement"] for pair in pairs
        ),
        "pairs": pairs,
    }
    if args.token_f1:
        summary["mean_token_id_multiset_f1"] = statistics.mean(
            pair["token_id_multiset_f1"] for pair in pairs
        )
        summary["token_f1_definition"] = (
            "Multiset overlap of generated token ids; ignores order and is not "
            "semantic accuracy."
        )
    _write_results(args.output, records, summary)
    return summary


def _dispatch(args: argparse.Namespace) -> dict:
    if args.command == "check-engine":
        manifest = (
            verify_vllm_sources(args.source_root, version=args.version)
            if args.source_root
            else verify_installed_vllm()
        )
        return {
            "verified": True,
            "vllm_version": manifest["vllm_version"],
            "commit": manifest["vllm_git_commit"],
            "files": len(manifest["sha256"]),
        }
    if args.command == "inspect":
        result = {"compatibility": compatibility_manifest()}
        if args.adapter:
            if not args.model:
                raise ValueError(
                    "--adapter inspection requires --model for backbone verification"
                )
            from .artifacts import AdapterBundle

            adapter = AdapterBundle(args.adapter, args.model)
            result["adapter"] = {
                "identity": adapter.identity,
                "metadata": adapter.metadata,
            }
            if args.profiles:
                from .profiles import ProfileBundle

                profile = ProfileBundle(args.profiles, adapter.identity)
                result["profiles"] = {
                    "identity": profile.identity,
                    "metadata": profile.metadata,
                }
        elif args.profiles:
            raise ValueError("--profiles inspection requires --adapter and --model")
        return result
    cases = read_cases(args.input, require_plan=args.command != "train")
    if args.command == "train":
        from .training import train_adapter

        trained = train_adapter(
            args.model,
            [case.token_ids for case in cases],
            args.output,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            rank=args.rank,
            max_positions=args.max_positions,
            query_chunk_size=args.query_chunk_size,
        )
        return {**asdict(trained), "output": str(trained.output)}
    if args.command == "calibrate":
        from .profiles import calibrate_profiles

        output = calibrate_profiles(
            args.model,
            args.adapter,
            [case.plan for case in cases],
            args.output,
            device=args.device,
            max_elements=args.max_elements,
            trained_profile_version=args.profile_version,
        )
        return {"output": str(output), "calibration_requests": len(cases)}
    if args.command == "bench":
        return benchmark(args, cases)
    _validate_native_cases(args, cases)
    if any(case.plan.operation == "calibrate" for case in cases):
        raise ValueError("use the calibrate command for calibration requests")
    if not args.run:
        return {
            "executed": False,
            "requires": "--run",
            "cases": len(cases),
            "model_runner": args.model_runner,
            "engine": engine_kwargs(args),
        }
    with engine_session(args) as (engine, sampling):
        records = [
            generate_one(
                engine,
                sampling,
                case,
                operation=case.plan.operation,
                max_tokens=args.max_tokens,
            )
            for case in cases
        ]
    summary = {
        "executed": True,
        "cases": len(cases),
        "backend": "native_vllm",
        "model_runner": args.model_runner,
        "timing_kind": "offline_generation_latency_seconds",
        "is_streaming_ttft": False,
    }
    _write_results(args.output, records, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (ValueError, OSError, CompatibilityError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
