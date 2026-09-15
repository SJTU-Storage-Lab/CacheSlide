"""Warm paired native measurements; no inference about TTFT or concurrent QPS."""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from dataclasses import replace


def token_f1(left, right):
    if not left or not right:
        return float(left == right)
    return 2 * sum((Counter(left) & Counter(right)).values()) / (len(left) + len(right))


def run_pairs(engine, cases, seeds, *, max_tokens, warmup, repeats):
    if not cases or any(type(n) is not int or n <= 0 for n in (max_tokens, repeats)):
        raise ValueError("benchmark requires cases, positive max_tokens and repeats")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    records, pairs = [], []

    def invoke(case, operation, phase, repeat):
        request = replace(case.plan, operation=operation)
        start = time.perf_counter()
        result = engine.generate(request, max_new_tokens=max_tokens)
        elapsed = time.perf_counter() - start
        receipt = result.get("meta_info", {}).get("cacheslide_receipt", {})
        metrics = receipt.get("metrics", {})
        ids = result.get("output_ids")
        valid_ids = isinstance(ids, list) and all(
            type(t) is int and t >= 0 for t in ids
        )
        valid = (
            receipt.get("status") == "complete"
            and receipt.get("resources_released") is True
            and receipt.get("output_ids") == ids
            and metrics.get("prompt_tokens") == len(request.token_ids)
            and metrics.get("operation") == operation
            and valid_ids
        )
        row = {
            "case_id": case.case_id,
            "phase": phase,
            "repeat": repeat,
            "operation": operation,
            "prompt_tokens": len(request.token_ids),
            "output_ids": ids,
            "elapsed_seconds": elapsed,
            "receipt_valid": valid,
            "receipt": receipt,
        }
        records.append(row)
        return row

    seeded = set()
    for case in seeds:
        identity = case.plan.cache_key("population-template", 0)
        if identity not in seeded:
            invoke(case, "populate", "setup", 0)
            seeded.add(identity)
    for repeat in range(warmup):
        for case in cases:
            for operation in ("recompute", "reuse"):
                invoke(case, operation, "warmup", repeat)
    for repeat in range(repeats):
        for case in cases:
            order = (
                ("recompute", "reuse") if repeat % 2 == 0 else ("reuse", "recompute")
            )
            output = {op: invoke(case, op, "measured", repeat) for op in order}
            base, reuse = output["recompute"], output["reuse"]
            bm, rm = (
                base["receipt"].get("metrics", {}),
                reuse["receipt"].get("metrics", {}),
            )
            outputs_valid = all(
                isinstance(row["output_ids"], list)
                and len(row["output_ids"]) == max_tokens
                for row in (base, reuse)
            )
            pair_valid = (
                base["receipt_valid"]
                and reuse["receipt_valid"]
                and outputs_valid
                and bm.get("fallback") is False
                and rm.get("fallback") is False
                and rm.get("cache_hit") is True
                and all(
                    math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
                    for row in (base, reuse)
                )
            )
            pairs.append(
                {
                    "case_id": case.case_id,
                    "repeat": repeat,
                    "valid": pair_valid,
                    "baseline_seconds": base["elapsed_seconds"],
                    "reuse_seconds": reuse["elapsed_seconds"],
                    "baseline_fallback": bm.get("fallback"),
                    "reuse_fallback": rm.get("fallback"),
                    "reuse_cache_hit": rm.get("cache_hit"),
                    "exact_token_agreement": base["output_ids"] == reuse["output_ids"]
                    if outputs_valid
                    else None,
                    "token_id_multiset_f1": token_f1(
                        base["output_ids"], reuse["output_ids"]
                    )
                    if outputs_valid
                    else None,
                }
            )
    baseline = statistics.mean(p["baseline_seconds"] for p in pairs)
    reuse = statistics.mean(p["reuse_seconds"] for p in pairs)
    valid = bool(pairs) and all(p["valid"] for p in pairs)
    summary = {
        "executed": True,
        "backend": "native_sglang",
        "pairs": pairs,
        "measured_pairs": len(pairs),
        "validation_passed": valid,
        "timing_kind": "offline_validated_generation_latency_seconds",
        "is_streaming_ttft": False,
        "is_concurrent_qps": False,
        "includes_receipt_validation_and_cleanup": True,
        "setup_and_warmup_excluded": True,
        "baseline_mean_seconds": baseline,
        "reuse_mean_seconds": reuse,
        "baseline_over_reuse_latency_ratio": baseline / reuse if valid else None,
        "exact_generated_token_agreement_rate": statistics.mean(
            p["exact_token_agreement"] for p in pairs
        )
        if all(p["exact_token_agreement"] is not None for p in pairs)
        else None,
        "mean_token_id_multiset_f1": statistics.mean(
            p["token_id_multiset_f1"] for p in pairs
        )
        if all(p["token_id_multiset_f1"] is not None for p in pairs)
        else None,
        "accuracy_definition": "Token multiset overlap; not dataset answer F1.",
    }
    return records, summary
