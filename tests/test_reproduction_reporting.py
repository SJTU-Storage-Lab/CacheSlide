import copy
import json

import pytest

from cacheslide_core.contracts import digest
from cacheslide_sglang.reproduction_metrics import score_answer
from cacheslide_sglang.reproduction_report import (
    build_report,
    main,
    read_jsonl,
    validate_evaluation,
)


def fixture():
    evaluation = [
        {
            "id": "held-out-case",
            "dataset": "hotpot_qa",
            "split": "validation",
            "revision": "0" * 40,
            "prompt_token_ids": [11, 12],
            "references": ["The red fox"],
            "metric": "qa_f1",
        }
    ]
    training = [{"id": "train-case", "prompt_token_ids": [21, 22]}]
    calibration = [{"id": "calibration-case", "prompt_token_ids": [31, 32]}]
    manifest = {
        "model": "local-mistral",
        "model_revision": "1" * 40,
        "tokenizer": "local-mistral",
        "tokenizer_revision": "1" * 40,
        "variants": {
            name: {"adapter_sha256": "a" * 64, "position_encoding": "trained_cope"}
            for name in ("adapter_recompute", "cacheslide")
        },
        "generation": {
            "temperature": 0,
            "batch_size": 1,
            "beam_width": 1,
            "max_new_tokens": 10,
        },
    }
    records = []
    for phase in ("warmup", "measured"):
        for operation in ("recompute", "reuse"):
            ids = [3, 4] if operation == "recompute" else [3]
            records.append(
                {
                    "case_id": "held-out-case",
                    "phase": phase,
                    "repeat": 0,
                    "operation": operation,
                    "output_ids": ids,
                    "prompt_tokens": 2,
                    "elapsed_seconds": 100
                    if phase == "warmup"
                    else (4 if operation == "recompute" else 2),
                    "receipt_valid": True,
                    "receipt": {
                        "status": "complete",
                        "resources_released": True,
                        "input_digest": digest([11, 12]),
                        "output_ids": ids,
                        "metrics": {
                            "operation": operation,
                            "prompt_tokens": 2,
                            "fallback": False,
                            "cache_hit": operation == "reuse",
                        },
                    },
                }
            )
    return evaluation, training, calibration, manifest, records


def report(records=None, manifest=None, **kwargs):
    evaluation, training, calibration, default_manifest, default_records = fixture()
    cases = validate_evaluation(evaluation, training, calibration)
    return build_report(
        default_records if records is None else records,
        cases,
        lambda ids: " ".join({3: "red", 4: "fox"}[token] for token in ids),
        default_manifest if manifest is None else manifest,
        baseline=kwargs.pop("baseline", "adapter_recompute"),
        repeats=1,
        **kwargs,
    )


def test_rouge_l_is_recall_not_f1_and_preserves_order():
    assert score_answer("a b c d", ["a b"], "rouge_l_recall") == 1
    assert score_answer("c b a", ["a b c"], "rouge_l_recall") == pytest.approx(1 / 3)
    assert score_answer("Hello, WORLD!", ["hello world"], "rouge_l_recall") == 1


@pytest.mark.parametrize("metric", ["qa_f1", "exact_match", "rouge_l_recall"])
def test_reference_selection_and_empty_answers(metric):
    assert score_answer("fox", ["dog", "fox"], metric) == 1
    assert score_answer("", [""], metric) == 1
    assert score_answer("", ["fox"], metric) == 0
    assert score_answer("fox", [""], metric) == 0


def test_qa_f1_normalizes_answers_but_is_not_token_id_consistency():
    assert score_answer("The red fox!", ["red fox"], "qa_f1") == 1
    assert score_answer("red", ["red fox"], "qa_f1") == pytest.approx(2 / 3)
    assert score_answer("red red red", ["red red"], "qa_f1") == pytest.approx(0.8)
    assert score_answer("red", ["red fox"], "exact_match") == 0
    with pytest.raises(ValueError, match="SWE"):
        score_answer("tests passed", ["tests passed"], "swe_resolved")


@pytest.mark.parametrize("answer", ["yes", "no", "noanswer"])
def test_official_hotpot_categorical_answers_require_exact_normalized_match(answer):
    assert score_answer(answer.upper() + "!", [answer], "hotpotqa_f1") == 1
    assert score_answer(answer.upper() + "!", [answer], "hotpotqa_em") == 1
    assert score_answer(answer + " extra", [answer], "hotpotqa_f1") == 0
    assert score_answer(answer, [answer + " extra"], "hotpotqa_f1") == 0
    assert score_answer(answer + " extra", [answer], "hotpotqa_em") == 0
    assert score_answer(answer + " extra", [answer], "qa_f1") > 0


def test_official_hotpot_empty_f1_is_zero_while_empty_em_is_one():
    assert score_answer("", [""], "hotpotqa_f1") == 0
    assert score_answer("The a", [""], "hotpotqa_f1") == 0
    assert score_answer("", [""], "hotpotqa_em") == 1
    assert score_answer("red fox", ["red fox fox"], "hotpotqa_f1") == pytest.approx(0.8)
    with pytest.raises(ValueError, match="one gold answer"):
        score_answer("red", ["red", "fox"], "hotpotqa_f1")


@pytest.mark.parametrize("split", ["training", "calibration"])
@pytest.mark.parametrize("identity", ["id", "prompt_token_ids"])
def test_exact_leaks_fail_hard(split, identity):
    evaluation, training, calibration, _, _ = fixture()
    rows = training if split == "training" else calibration
    rows[0][identity] = evaluation[0][identity]
    with pytest.raises(ValueError, match="leakage"):
        validate_evaluation(evaluation, training, calibration)


def test_missing_training_provenance_or_train_evaluation_split_rejected():
    evaluation, training, calibration, _, _ = fixture()
    with pytest.raises(ValueError, match="provenance"):
        validate_evaluation(evaluation, [], calibration)
    evaluation[0]["split"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        validate_evaluation(evaluation, training, calibration)


def test_report_decodes_text_scores_gold_excludes_warmup_and_has_no_fake_ttft():
    pairs, summary = report()
    assert pairs[0]["adapter_recompute"]["text"] == "red fox"
    assert pairs[0]["cacheslide"]["text"] == "red"
    assert summary["status"] == "valid_partial_experiment"
    assert summary["paper_results_reproduced"] is False
    assert summary["comparison_kind"] == "same_adapter_recompute_diagnostic"
    assert summary["generation_latency_speedup_ratio_of_means"] == 2
    assert summary["baseline_mean_generation_seconds"] == 4
    assert summary["cacheslide_mean_generation_seconds"] == 2
    assert summary["datasets"][0]["score_delta_percentage_points"] == pytest.approx(
        -100 / 3
    )
    assert summary["ttft_speedup_ratio_of_means"] is None
    assert summary["concurrent_qps"] is None
    assert summary["goodput_correct_completions_per_second"] is None
    assert summary["swe_bench_resolved_rate"] is None


def test_only_true_first_token_events_enable_ttft():
    *_, records = fixture()
    for row in records:
        row["first_token_event"] = {
            "clock": "client_monotonic",
            "arrival_seconds": 10.0,
            "first_token_seconds": 13.0 if row["operation"] == "recompute" else 11.0,
            "source": "stream_token_event",
            "generated_token_count": 1,
        }
    _, summary = report(records=records)
    assert summary["ttft_speedup_ratio_of_means"] == 3
    assert summary["generation_latency_speedup_ratio_of_means"] == 2


@pytest.mark.parametrize("invalid", ["no_token", "second_token", "wall_clock", "late"])
def test_invalid_ttft_evidence_rejected(invalid):
    *_, records = fixture()
    event = {
        "clock": "client_monotonic",
        "arrival_seconds": 0.0,
        "first_token_seconds": 1.0,
        "source": "stream_token_event",
        "generated_token_count": 1,
    }
    if invalid == "no_token":
        event["source"] = "http_headers_received"
    elif invalid == "second_token":
        event["generated_token_count"] = 2
    elif invalid == "wall_clock":
        event["clock"] = "time.time"
    else:
        event["first_token_seconds"] = 50
    records[-1]["first_token_event"] = event
    with pytest.raises(ValueError):
        report(records=records)


@pytest.mark.parametrize(
    "invalid", ["fallback", "miss", "digest", "receipt", "duration"]
)
def test_invalid_measurement_cannot_be_dropped_to_claim_speedup(invalid):
    *_, records = fixture()
    row = records[-1]
    if invalid == "fallback":
        row["receipt"]["metrics"]["fallback"] = True
    elif invalid == "miss":
        row["receipt"]["metrics"]["cache_hit"] = False
    elif invalid == "digest":
        row["receipt"]["input_digest"] = digest([999, 999])
    elif invalid == "receipt":
        row["receipt_valid"] = False
    else:
        row["elapsed_seconds"] = float("nan")
    _, summary = report(records=records)
    assert summary["status"] == "invalid_experiment"
    assert summary["generation_latency_speedup_ratio_of_means"] is None
    assert "baseline_mean_score" not in summary["datasets"][0]


def test_duplicate_missing_and_unwarmed_measurements_detected():
    *_, records = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        report(records=records + [copy.deepcopy(records[-1])])
    _, summary = report(records=records[:-1])
    assert any("missing_measurement" in error for error in summary["validation_errors"])
    _, summary = report(records=records[2:])
    assert any("missing_warmup" in error for error in summary["validation_errors"])
    with pytest.raises(ValueError, match="warmup"):
        report(warmup_min=0)


def test_native_baseline_cannot_be_inferred_from_recompute():
    *_, manifest, records = fixture()
    manifest["variants"]["native_baseline"] = {
        "adapter_sha256": None,
        "position_encoding": "rope",
    }
    _, summary = report(records=records, manifest=manifest, baseline="native_baseline")
    assert summary["status"] == "invalid_experiment"
    for row in records:
        if row["operation"] == "recompute":
            row["variant"] = "native_baseline"
            row["input_digest"] = digest([11, 12])
            row["status"] = "complete"
    _, summary = report(records=records, manifest=manifest, baseline="native_baseline")
    assert summary["status"] == "valid_partial_experiment"
    assert summary["comparison_kind"] == "paper_native_baseline"
    manifest["variants"]["native_baseline"]["adapter_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="no adapter"):
        report(records=records, manifest=manifest, baseline="native_baseline")


def test_different_adapter_diagnostic_rejected():
    *_, manifest, _ = fixture()
    manifest["variants"]["adapter_recompute"]["adapter_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identical adapter"):
        report(manifest=manifest)


@pytest.mark.parametrize("revision", ["main", "latest", "placeholder", "Z" * 40])
def test_mutable_or_placeholder_revisions_rejected(revision):
    evaluation, training, calibration, manifest, _ = fixture()
    evaluation[0]["revision"] = revision
    with pytest.raises(ValueError, match="hash"):
        validate_evaluation(evaluation, training, calibration)
    manifest["model_revision"] = revision
    with pytest.raises(ValueError, match="immutable"):
        report(manifest=manifest)


@pytest.mark.parametrize("content", ['{"a":1,"a":2}\n', '{"a":NaN}\n', "[]\n"])
def test_json_input_rejects_duplicate_nonfinite_and_nonobject_rows(tmp_path, content):
    path = tmp_path / "records.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_cli_new_path_guard_precedes_tokenizer_or_output_mutation(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    args = []
    for name in ("records", "evaluation", "training", "calibration", "manifest"):
        args += ["--" + name, str(tmp_path / name)]
    args += [
        "--tokenizer",
        str(tmp_path),
        "--output",
        str(output),
        "--baseline",
        "adapter_recompute",
        "--repeats",
        "1",
    ]
    assert main(args) == 2
    assert list(output.iterdir()) == []


def test_report_serializable_without_nan():
    pairs, summary = report()
    json.dumps({"pairs": pairs, "summary": summary}, allow_nan=False)
