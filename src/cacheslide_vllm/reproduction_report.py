"""Offline, fail-closed reporting of real saved generation experiments.

This module does not run or fabricate inference. It decodes saved token IDs,
scores held-out reference answers and distinguishes trained-adapter recompute
from an untouched model baseline. Existing blocking benchmark durations are
never renamed TTFT/QPS. The CLI only reads local models and input artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .cli import _reject_constant, _unique_object
from .contracts import digest
from .reproduction_metrics import HOTPOTQA_SCORER, METRICS, score_answer

VARIANTS = frozenset({"native_baseline", "adapter_recompute", "cacheslide"})


def immutable_revision(value):
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(char in "0123456789abcdef" for char in value)
    )


def read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line.encode()) > 16 * 1024 * 1024:
                raise ValueError(f"oversized JSONL row {number}")
            row = json.loads(
                line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be objects")
            rows.append(row)
    if not rows:
        raise ValueError("empty JSONL input")
    return rows


def token_ids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(token) is int and token >= 0 for token in value)
    )


def validate_evaluation(rows, training, calibration):
    """Reject exact prompt/case leakage; documents may intentionally be shared.

    This cannot prove semantic/source-document split separation. Immutable
    dataset revisions and the upstream split construction must be audited too.
    """
    evaluation = {}
    eval_hashes = set()
    for row in rows:
        required = {"id", "dataset", "split", "revision", "prompt_token_ids"}
        if not required.issubset(row) or any(
            not isinstance(row[name], str) or not row[name]
            for name in ("id", "dataset", "split", "revision")
        ):
            raise ValueError("evaluation requires case/dataset/split/revision identity")
        if row["split"].lower() in {"train", "training"}:
            raise ValueError("evaluation must use an explicitly held-out split")
        if not immutable_revision(row["revision"]):
            raise ValueError("evaluation requires a commit or source-content hash")
        if row["id"] in evaluation or not token_ids(row["prompt_token_ids"]):
            raise ValueError("duplicate case id or invalid evaluation prompt tokens")
        if row.get("metric") not in METRICS:
            raise ValueError("unknown metric; SWE-bench requires real sandbox tests")
        references = row.get("references")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(reference, str) for reference in references)
        ):
            raise ValueError("every evaluation case requires reference answers")
        fingerprint = digest(row["prompt_token_ids"])
        if fingerprint in eval_hashes:
            raise ValueError("duplicate evaluation prompts would overweight cases")
        evaluation[row["id"]] = row
        eval_hashes.add(fingerprint)
    if not evaluation:
        raise ValueError("empty evaluation")
    for label, preparation in (("training", training), ("calibration", calibration)):
        if not preparation:
            raise ValueError(f"{label} provenance input is required")
        for row in preparation:
            if not token_ids(row.get("prompt_token_ids")):
                raise ValueError(f"invalid {label} prompt")
            if row.get("id") in evaluation or digest(row["prompt_token_ids"]) in (
                eval_hashes
            ):
                raise ValueError(f"evaluation/{label} exact case or prompt leakage")
    return evaluation


def validate_manifest(manifest, baseline):
    if not isinstance(manifest, dict):
        raise ValueError("experiment manifest must be an object")
    required = {"model", "model_revision", "tokenizer", "tokenizer_revision"}
    if any(not isinstance(manifest.get(k), str) or not manifest[k] for k in required):
        raise ValueError("manifest requires model and tokenizer pinned identities")
    if not all(
        immutable_revision(manifest[name])
        for name in ("model_revision", "tokenizer_revision")
    ):
        raise ValueError(
            "model/tokenizer revisions must be immutable commit/content hashes"
        )
    variants = manifest.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("manifest variants must be an object")
    for name in (baseline, "cacheslide"):
        if not isinstance(variants.get(name), dict):
            raise ValueError("manifest must describe both compared variants")
    slide = variants["cacheslide"]
    adapter = slide.get("adapter_sha256")
    if (
        not isinstance(adapter, str)
        or len(adapter) != 64
        or any(char not in "0123456789abcdef" for char in adapter)
        or slide.get("position_encoding") != "trained_cope"
    ):
        raise ValueError("CacheSlide requires a trained CoPE adapter SHA256")
    base = variants[baseline]
    if baseline == "adapter_recompute":
        if (
            base.get("adapter_sha256") != adapter
            or base.get("position_encoding") != "trained_cope"
        ):
            raise ValueError("recompute diagnostic must use the identical adapter")
    elif base.get("adapter_sha256") is not None or base.get(
        "position_encoding"
    ) not in {"rope", "alibi"}:
        raise ValueError("native baseline must retain original RoPE/ALiBi, no adapter")
    generation = manifest.get("generation")
    if not isinstance(generation, dict) or generation.get("temperature") != 0:
        raise ValueError("paired generation needs a recorded deterministic protocol")
    if generation.get("batch_size") != 1 or generation.get("beam_width") != 1:
        raise ValueError("this paired report supports batch 1, no beam only")
    maximum = generation.get("max_new_tokens")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("positive max_new_tokens is required")
    return maximum


def variant_for(row):
    variant = row.get("variant")
    if variant is None:
        variant = {"recompute": "adapter_recompute", "reuse": "cacheslide"}.get(
            row.get("operation")
        )
    return variant


def _positive_finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def measured_ttft(row):
    """Require a real first-token event; blocking duration is insufficient."""
    event = row.get("first_token_event")
    if event is None:
        return None
    if not isinstance(event, dict) or event.get("clock") != "client_monotonic":
        raise ValueError("TTFT needs one client monotonic clock")
    arrival, first = event.get("arrival_seconds"), event.get("first_token_seconds")
    if (
        type(arrival) not in (int, float)
        or type(first) not in (int, float)
        or not math.isfinite(arrival)
        or not math.isfinite(first)
        or arrival < 0
        or first <= arrival
        or event.get("source") != "stream_token_event"
        or event.get("generated_token_count") != 1
    ):
        raise ValueError("invalid arrival/first generated token timing evidence")
    result = first - arrival
    if result > row["elapsed_seconds"]:
        raise ValueError("TTFT exceeds full generation interval")
    return result


def validate_record(row, case, variant, maximum):
    """Validate the real vLLM CLI fields, not fabricated SGLang receipts.

    This RPC snapshot binds request ID, operation and prompt length. Unlike the
    SGLang receipt, it contains neither output digest nor resource-release
    attestation; saved prompt IDs and native completion fields are checked here.
    """
    errors = []
    ids = row.get("generated_token_ids")
    if not token_ids(ids) or len(ids) != maximum:
        errors.append("invalid_or_wrong_length_generated_token_ids")
    if row.get("prompt_token_ids") != case["prompt_token_ids"] or row.get(
        "input_tokens"
    ) != len(case["prompt_token_ids"]):
        errors.append("full_prompt_or_length_mismatch")
    if not token_ids(ids) or row.get("output_tokens") != len(ids):
        errors.append("output_token_count_mismatch")
    if row.get("finish_reason") != "length":
        errors.append("not_a_complete_fixed_budget_generation")
    request_id = row.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        errors.append("missing_native_request_id")
    if not _positive_finite(row.get("elapsed_seconds")):
        errors.append("invalid_generation_duration")
    if variant == "native_baseline":
        if row.get("status") != "complete" or row.get("variant") != variant:
            errors.append("native_baseline_not_explicitly_complete")
    else:
        metrics = row.get("runtime_metrics")
        if not isinstance(metrics, dict):
            return errors + ["missing_vllm_worker_metrics"]
        expected = "recompute" if variant == "adapter_recompute" else "reuse"
        if not (
            row.get("metrics_valid") is True
            and row.get("metrics_error") is None
            and row.get("operation") == expected
            and metrics.get("request_id") == request_id
            and metrics.get("operation") == expected
            and metrics.get("prompt_tokens") == len(case["prompt_token_ids"])
            and metrics.get("fallback") is False
        ):
            errors.append("invalid_vllm_request_bound_metrics")
        if variant == "cacheslide" and metrics.get("cache_hit") is not True:
            errors.append("reuse_is_not_cache_hit")
    return errors


def build_report(
    records, evaluation, decode, manifest, *, baseline, repeats, warmup_min=1
):
    """Complete coverage only: invalid/missing cases cannot be silently dropped."""
    if baseline not in {"native_baseline", "adapter_recompute"}:
        raise ValueError("choose a native baseline or same-adapter diagnostic")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    if type(warmup_min) is not int or warmup_min < 1:
        raise ValueError("warm TTFT reporting requires at least one warmup")
    maximum = validate_manifest(manifest, baseline)
    indexed, warm = {}, defaultdict(set)
    errors = []
    variants = {baseline, "cacheslide"}
    measured_started = False
    for row in records:
        phase, variant = row.get("phase"), variant_for(row)
        if phase == "setup":
            continue
        if phase not in {"warmup", "measured"}:
            raise ValueError("unknown raw record phase")
        if variant not in variants:
            continue
        case_id, repeat = row.get("case_id"), row.get("repeat")
        if case_id not in evaluation or type(repeat) is not int or repeat < 0:
            raise ValueError("unexpected case/repeat in raw outputs")
        key = case_id, variant, repeat
        invalid = validate_record(row, evaluation[case_id], variant, maximum)
        if phase == "warmup":
            if measured_started:
                errors.append("warmup_after_measurement")
            if repeat in warm[(case_id, variant)]:
                errors.append("duplicate_warmup")
            if invalid:
                errors.extend(f"warmup:{case_id}:{variant}:{e}" for e in invalid)
            warm[(case_id, variant)].add(repeat)
        else:
            measured_started = True
            if repeat >= repeats or key in indexed:
                raise ValueError("duplicate or excess measured repeat")
            indexed[key] = row, invalid
    pairs = []
    for case_id, case in evaluation.items():
        for variant in variants:
            if len(warm[(case_id, variant)]) < warmup_min:
                errors.append(f"missing_warmup:{case_id}:{variant}")
        for repeat in range(repeats):
            pair = {"case_id": case_id, "dataset": case["dataset"], "repeat": repeat}
            pair.update(metric=case["metric"], references=case["references"])
            for variant in (baseline, "cacheslide"):
                item = indexed.get((case_id, variant, repeat))
                if item is None:
                    errors.append(f"missing_measurement:{case_id}:{variant}:{repeat}")
                    pair[variant] = None
                    continue
                row, invalid = item
                errors.extend(f"{case_id}:{variant}:{repeat}:{e}" for e in invalid)
                ids = row.get("generated_token_ids")
                text = decode(ids) if token_ids(ids) else None
                if text is not None and not isinstance(text, str):
                    raise ValueError("tokenizer decoder must return text")
                pair[variant] = {
                    "text": text,
                    "output_ids": ids,
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "ttft_seconds": measured_ttft(row) if not invalid else None,
                    "answer_score": score_answer(
                        text, case["references"], case["metric"]
                    )
                    if text is not None and not invalid
                    else None,
                    "validation_errors": invalid,
                }
            scores = [
                pair.get(variant, {}).get("answer_score")
                if pair.get(variant) is not None
                else None
                for variant in (baseline, "cacheslide")
            ]
            pair["score_delta_percentage_points"] = (
                100 * (scores[1] - scores[0])
                if all(score is not None for score in scores)
                else None
            )
            pairs.append(pair)
    valid = not errors
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair["dataset"], pair["metric"])].append(pair)
    groups = []
    for (dataset, metric), group in sorted(grouped.items()):
        row = {"dataset": dataset, "metric": metric, "measured_pairs": len(group)}
        if valid:
            base = statistics.mean(p[baseline]["answer_score"] for p in group)
            slide = statistics.mean(p["cacheslide"]["answer_score"] for p in group)
            row.update(
                baseline_mean_score=base,
                cacheslide_mean_score=slide,
                score_delta_percentage_points=100 * (slide - base),
            )
        groups.append(row)
    summary = {
        "schema_version": 1,
        "input_record_schema": "cacheslide_vllm.cli.generate_one",
        "worker_output_digest_and_release_attested": False,
        "status": "valid_partial_experiment" if valid else "invalid_experiment",
        "paper_results_reproduced": False,
        "baseline_variant": baseline,
        "comparison_kind": "paper_native_baseline"
        if baseline == "native_baseline"
        else "same_adapter_recompute_diagnostic",
        "validation_errors": errors,
        "expected_pairs": len(evaluation) * repeats,
        "datasets": groups,
        "generation_latency_speedup_ratio_of_means": None,
        "generation_latency_mean_of_pair_ratios": None,
        "ttft_speedup_ratio_of_means": None,
        "concurrent_qps": None,
        "goodput_correct_completions_per_second": None,
        "swe_bench_resolved_rate": None,
        "manifest": manifest,
        "hotpotqa_answer_scorer": HOTPOTQA_SCORER,
        "limitations": [
            "This report is not proof of the FAST paper's numerical results.",
            "Exact prompt disjointness does not certify semantic split separation.",
            "ROUGE-L is non-stemmed Unicode-word LCS recall; paper scorer unspecified.",
            "Native baseline and adapter recompute are different comparisons.",
            "Blocking generation latency is not first-token latency or QPS.",
            "No SWE-bench resolved rate without real environment test execution.",
            "vLLM worker metrics bind request ID, operation and prompt length; "
            "they do not attest output digest or resource release.",
        ],
    }
    if valid:
        a = statistics.mean(p[baseline]["elapsed_seconds"] for p in pairs)
        b = statistics.mean(p["cacheslide"]["elapsed_seconds"] for p in pairs)
        summary["baseline_mean_generation_seconds"] = a
        summary["cacheslide_mean_generation_seconds"] = b
        summary["generation_latency_speedup_ratio_of_means"] = a / b
        summary["generation_latency_mean_of_pair_ratios"] = statistics.mean(
            p[baseline]["elapsed_seconds"] / p["cacheslide"]["elapsed_seconds"]
            for p in pairs
        )
        if all(p[v]["ttft_seconds"] is not None for p in pairs for v in variants):
            a = statistics.mean(p[baseline]["ttft_seconds"] for p in pairs)
            b = statistics.mean(p["cacheslide"]["ttft_seconds"] for p in pairs)
            summary["baseline_mean_ttft_seconds"] = a
            summary["cacheslide_mean_ttft_seconds"] = b
            summary["ttft_speedup_ratio_of_means"] = a / b
    return pairs, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "evaluation", "training", "calibration", "manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--baseline", choices=("adapter_recompute", "native_baseline"), required=True
    )
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup-min", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("report output must be a new directory")
        if not args.tokenizer.is_dir():
            raise ValueError("tokenizer must be an existing local directory")
        evaluation = validate_evaluation(
            read_jsonl(args.evaluation),
            read_jsonl(args.training),
            read_jsonl(args.calibration),
        )
        manifest = json.loads(
            args.manifest.read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer.resolve()),
            local_files_only=True,
            trust_remote_code=False,
        )
        pairs, summary = build_report(
            read_jsonl(args.records),
            evaluation,
            lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            manifest,
            baseline=args.baseline,
            repeats=args.repeats,
            warmup_min=args.warmup_min,
        )
        summary["input_sha256"] = {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("records", "evaluation", "training", "calibration", "manifest")
        }
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "paired_outputs.jsonl").open("x") as stream:
            for pair in pairs:
                stream.write(
                    json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n"
                )
        with (args.output / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0 if not summary["validation_errors"] else 2
    except (ValueError, OSError, ImportError) as exc:
        print(f"reproduction report failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
