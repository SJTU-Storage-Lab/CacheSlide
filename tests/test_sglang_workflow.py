import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheslide_core.inputs import read_cases
from cacheslide_sglang import benchmark, workflow


def cases_file(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case",
                "prompt_token_ids": [1, 2, 3],
                "cacheslide": {
                    "version": 1,
                    "namespace": "test",
                    "task_id": "qa",
                    "operation": "reuse",
                    "chunks": [
                        {"id": "fixed", "role": "reuse", "start": 0, "end": 2},
                        {"id": "query", "role": "recompute", "start": 2, "end": 3},
                    ],
                },
            }
        )
        + "\n"
    )
    return path


def test_plan_does_not_create_outputs_or_import_sglang(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("subprocess in read-only plan")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert (
        workflow.main(
            [
                "--model",
                str(tmp_path / "model"),
                "--input",
                str(cases_file(tmp_path)),
                "--output",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["executed"] is False
    assert not (tmp_path / "results").exists()
    assert "sglang" not in sys.modules


@pytest.mark.parametrize("variant", [False, True])
def test_actual_cpu_workflow_trains_and_compares_native_token_layout(
    tmp_path, capsys, variant
):
    root = tmp_path / "smoke"
    args = ["--smoke", "--output", str(root), "--warmup", "0", "--repeats", "1"]
    if variant:
        args += [
            "--ccpe-position-policy",
            "mixed_bias_override",
            "--calibration-layer",
            "1",
        ]
    assert workflow.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["correctness_passed"]
    assert summary["native_sglang_executed"] is False
    assert summary["dense_and_token_arena_logits_close"]
    assert summary["dense_and_token_arena_kv_close"]
    assert summary["shifted_nonprefix_reuse_passed"] is variant
    assert summary["guarded_fallback_requests"] == (0 if variant else 2)
    assert summary["training_steps"] == 2
    assert summary["decode_steps_per_request"] == 3
    assert json.loads((root / "workflow.json").read_text())["status"] == "completed"
    assert (root / "logs/train.stdout.log").is_file()
    assert (root / "logs/calibrate.stdout.log").is_file()


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "reuse_fallback",
        "baseline_fallback",
        "miss",
        "missing_receipt",
        "short_output",
    ],
)
def test_warm_paired_metrics_invalidate_false_speedups(tmp_path, monkeypatch, failure):
    cases = read_cases(cases_file(tmp_path))
    clock = SimpleNamespace(value=0.0, calls=0)

    class Engine:
        def generate(self, plan, max_new_tokens):
            clock.calls += 1
            clock.value += (
                100 if clock.calls <= 3 else (4 if plan.operation == "recompute" else 2)
            )
            ids = [4] if failure == "short_output" else [4, 5]
            metrics = {
                "operation": plan.operation,
                "prompt_tokens": len(plan.token_ids),
                "cache_hit": plan.operation == "reuse" and failure != "miss",
                "fallback": failure
                == (
                    "baseline_fallback"
                    if plan.operation == "recompute"
                    else plan.operation + "_fallback"
                ),
            }
            receipt = {
                "status": "complete",
                "resources_released": True,
                "output_ids": ids,
                "metrics": metrics,
            }
            return {
                "output_ids": ids,
                "meta_info": {"cacheslide_receipt": receipt}
                if failure != "missing_receipt"
                else {},
            }

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock.value)
    records, summary = benchmark.run_pairs(
        Engine(), cases, cases, max_tokens=2, warmup=1, repeats=2
    )
    assert len(records) == 7 and len(summary["pairs"]) == 2
    assert summary["baseline_mean_seconds"] == 4
    assert summary["reuse_mean_seconds"] == 2
    assert summary["baseline_over_reuse_latency_ratio"] == (
        2 if failure is None else None
    )
    assert summary["is_streaming_ttft"] is False
    assert summary["is_concurrent_qps"] is False


def test_stage_failure_preserves_diagnostics_without_backend_switch(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        workflow.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7)
    )
    root = tmp_path / "failed"
    assert workflow.main(["--smoke", "--output", str(root)]) == 2
    assert "No backend fallback" in capsys.readouterr().err
    assert json.loads((root / "workflow.json").read_text())["status"] == "failed"
    assert (root / "logs/train.stderr.log").is_file()
    assert not (root / "summary.json").exists()


def test_sglang_launcher_outside_repo_avoids_cwd_shadow(tmp_path):
    fake = tmp_path / "cacheslide_sglang"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise RuntimeError('shadowed')")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CACHESLIDE_PYTHON=sys.executable)
    result = subprocess.run(
        [str(root / "run_cacheslide_benchmark.sh"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "SGLang" in result.stdout


@pytest.mark.parametrize(
    "option,value",
    [
        ("--train-input", "cases.jsonl"),
        ("--calibration-input", "cases.jsonl"),
        ("--seed-input", "cases.jsonl"),
        ("--train-device", "cuda"),
        ("--calibration-device", "cuda"),
        ("--mem-fraction-static", "1"),
    ],
)
def test_smoke_rejects_foreign_inputs_and_gpu_before_creating_output(
    tmp_path, option, value
):
    output = tmp_path / "result"
    assert workflow.main(["--smoke", "--output", str(output), option, value]) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "kind", ["equal_cache", "parent_cache", "child_venv", "parent_venv"]
)
def test_invalid_path_overlap_is_rejected_without_side_effects(tmp_path, kind):
    output = tmp_path / "new" / "result"
    paths = {
        "equal_cache": ["--cache-root", str(output)],
        "parent_cache": ["--cache-root", str(output.parent)],
        "child_venv": ["--venv", str(output / "env")],
        "parent_venv": ["--venv", str(output.parent)],
    }
    assert workflow.main(["--smoke", "--output", str(output), *paths[kind]]) == 2
    assert not output.parent.exists()


@pytest.mark.parametrize("equals_form", [False, True])
def test_install_reexec_removes_new_path_environment_argument(
    tmp_path, monkeypatch, equals_form
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workflow.subprocess, "run", run)
    environment = str(tmp_path / "env")
    option = [f"--venv={environment}"] if equals_form else ["--venv", environment]
    assert (
        workflow.main(
            ["--smoke", "--install", "--output", str(tmp_path / "result"), *option]
        )
        == 0
    )
    command, kwargs = calls[-1]
    assert "--install" not in command
    assert not any(arg == "--venv" or arg.startswith("--venv=") for arg in command)
    assert "PYTHONPATH" not in kwargs["env"]
    assert "cacheslide_sglang.workflow" in command


@pytest.mark.parametrize("capacity", ["--context-length", "--max-total-tokens"])
def test_prompt_and_output_must_fit_before_engine_or_training(
    tmp_path, capacity, capsys
):
    output = tmp_path / "result"
    assert (
        workflow.main(
            [
                "--model",
                str(tmp_path / "model"),
                "--input",
                str(cases_file(tmp_path)),
                "--output",
                str(output),
                "--run",
                capacity,
                "6",
                "--max-tokens",
                "4",
            ]
        )
        == 2
    )
    assert "exceeds token capacity" in capsys.readouterr().err
    assert not output.exists()


def test_default_install_environment_cannot_overlap_cache(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must reject overlapping default venv before installation")

    monkeypatch.setattr(workflow.subprocess, "run", forbidden)
    assert (
        workflow.main(
            [
                "--smoke",
                "--install",
                "--output",
                str(tmp_path / "result"),
                "--cache-root",
                str(tmp_path / ".sglang-result"),
            ]
        )
        == 2
    )
    assert not (tmp_path / ".sglang-result").exists()


def test_native_validation_failure_preserves_metrics_and_returns_nonzero(
    tmp_path, monkeypatch
):
    from cacheslide_sglang import compat, integration

    monkeypatch.setattr(compat, "verify_compatibility", lambda: {})
    monkeypatch.setattr(
        integration, "create_engine", lambda *a, **kw: nullcontext(object())
    )
    monkeypatch.setattr(
        benchmark,
        "run_pairs",
        lambda *a, **kw: (
            [{"failure": "cache_miss"}],
            {"validation_passed": False, "baseline_over_reuse_latency_ratio": None},
        ),
    )
    output = tmp_path / "result"
    assert (
        workflow.main(
            [
                "--model",
                str(tmp_path / "model"),
                "--input",
                str(cases_file(tmp_path)),
                "--adapter",
                str(tmp_path / "adapter"),
                "--profiles",
                str(tmp_path / "profiles"),
                "--output",
                str(output),
                "--run",
            ]
        )
        == 2
    )
    assert (
        json.loads((output / "workflow.json").read_text())["status"]
        == "validation_failed"
    )
    assert (
        json.loads((output / "summary.json").read_text())["validation_passed"] is False
    )
    assert (output / "raw_outputs.jsonl").read_text().strip()
