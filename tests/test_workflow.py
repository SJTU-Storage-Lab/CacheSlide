"""Workflow orchestration, fail-closed execution, and actual offline CPU smoke."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheslide_vllm import cli, workflow


def native_input(tmp_path, extra=()):
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt_token_ids": [1, 2, 3],
                "cacheslide": {
                    "version": 1,
                    "operation": "reuse",
                    "namespace": "workflow-test",
                    "task_id": "test",
                    "chunks": [
                        {"id": "fixed", "role": "reuse", "start": 0, "end": 2},
                        {"id": "query", "role": "recompute", "start": 2, "end": 3},
                    ],
                },
            }
        )
        + "\n"
    )
    return [
        "--model",
        str(model),
        "--input",
        str(cases),
        "--output",
        str(tmp_path / "run"),
        *extra,
    ]


def test_native_plan_uses_existing_cli_and_never_executes(
    tmp_path, monkeypatch, capsys
):
    arguments = native_input(tmp_path)
    monkeypatch.setattr(workflow.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    assert workflow.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["executed"] is False and plan["requires"] == "--run"
    assert plan["vllm_version"] == "0.29.0"
    assert plan["model_downloaded"] is False
    assert [stage["name"] for stage in plan["stages"]] == [
        "check-engine",
        "train",
        "calibrate",
        "benchmark",
    ]
    parsed = [
        cli.parser().parse_args(stage["cli_arguments"]) for stage in plan["stages"]
    ]
    assert parsed[1].steps == 20
    assert parsed[3].backend == "native" and parsed[3].run
    assert parsed[3].model_runner == "v2"
    assert parsed[3].warmup == 1 and parsed[3].repeats == 3
    assert not (tmp_path / "run").exists()


def test_existing_artifacts_skip_training_and_calibration(tmp_path):
    adapter, profiles = tmp_path / "adapter", tmp_path / "profiles"
    adapter.mkdir()
    profiles.mkdir()
    args = workflow.prepare(
        workflow.parser().parse_args(
            native_input(
                tmp_path,
                (
                    "--adapter",
                    str(adapter),
                    "--profiles",
                    str(profiles),
                    "--max-tokens",
                    "5",
                    "--model-runner",
                    "v1",
                    "--token-f1",
                ),
            )
        )
    )
    stages = workflow.native_stages(args)
    assert [name for name, _ in stages] == ["check-engine", "benchmark"]
    benchmark = cli.parser().parse_args(stages[-1][1])
    assert benchmark.adapter == str(adapter) and benchmark.profiles == str(profiles)
    assert benchmark.max_tokens == 5 and benchmark.model_runner == "v1"
    assert benchmark.token_f1


def test_cache_path_can_target_new_external_storage(tmp_path):
    cache = tmp_path / "ssd-cache"
    args = workflow.prepare(
        workflow.parser().parse_args(
            native_input(
                tmp_path,
                ("--cache-root", str(cache)),
            )
        )
    )
    benchmark = cli.parser().parse_args(workflow.native_stages(args)[-1][1])
    assert benchmark.cache_root == str(cache)
    assert not cache.exists()


def test_existing_cache_is_not_reused_implicitly(tmp_path, capsys):
    cache = tmp_path / "ssd-cache"
    cache.mkdir()
    assert (
        workflow.main(
            [
                "--smoke",
                "--output",
                str(tmp_path / "run"),
                "--cache-root",
                str(cache),
            ]
        )
        == 2
    )
    assert "cache-root must be a new path" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "dangling-symlink"])
def test_output_collision_never_overwrites(tmp_path, kind, capsys):
    output = tmp_path / "exists"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("preserve")
    else:
        output.symlink_to(tmp_path / "missing")
    assert workflow.main(["--smoke", "--output", str(output)]) == 2
    assert "new path" in capsys.readouterr().err
    if kind == "file":
        assert output.read_text() == "preserve"
    elif kind == "dangling-symlink":
        assert output.is_symlink()


def test_no_model_prints_usage_without_downloading(monkeypatch, capsys):
    monkeypatch.setattr(workflow.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    assert workflow.main([]) == 2
    assert "No model is downloaded" in capsys.readouterr().out
    assert workflow.main(["--run"]) == 2
    assert "requires --model" in capsys.readouterr().err


def test_missing_dependencies_give_explicit_install_guidance(
    tmp_path, monkeypatch, capsys
):
    args = native_input(tmp_path, ("--run",))
    monkeypatch.setattr(workflow.importlib.util, "find_spec", lambda _: None)
    assert workflow.main(args) == 2
    error = capsys.readouterr().err
    assert "--install" in error and "CACHESLIDE_PYTHON" in error
    assert "vllm==0.29.0" in error
    assert not (tmp_path / "run").exists()


def test_failed_native_stage_stops_without_reference_fallback(
    tmp_path, monkeypatch, capsys
):
    args = native_input(tmp_path, ("--run",))
    monkeypatch.setattr(workflow, "_dependencies", lambda **kwargs: None)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(workflow.subprocess, "run", fail)
    assert workflow.main(args) == 2
    assert "No backend fallback" in capsys.readouterr().err
    assert len(calls) == 1 and calls[0][-1] == "check-engine"
    manifest = json.loads((tmp_path / "run" / "workflow.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["mode"] == "native_vllm"
    assert manifest["stages"][0]["returncode"] == 7
    assert not (tmp_path / "run" / "adapter").exists()


def test_invalid_benchmark_receipts_do_not_report_workflow_success(
    tmp_path,
    monkeypatch,
    capsys,
):
    args = native_input(tmp_path, ("--run",))
    monkeypatch.setattr(workflow, "_dependencies", lambda **kwargs: None)

    def run_stage(name, arguments, root, manifest):
        if name == "benchmark":
            (root / "benchmark").mkdir()
            (root / "benchmark" / "summary.json").write_text(
                json.dumps(
                    {
                        "executed": True,
                        "baseline_validation_passed": True,
                        "reuse_validation_passed": False,
                        "latency_validation_passed": True,
                        "baseline_over_reuse_latency_ratio": None,
                    }
                )
            )

    monkeypatch.setattr(workflow, "_run_stage", run_stage)
    assert workflow.main(args) == 2
    assert "did not validate" in capsys.readouterr().err
    manifest = json.loads((tmp_path / "run" / "workflow.json").read_text())
    assert manifest["status"] == "failed"
    assert (tmp_path / "run" / "benchmark" / "summary.json").exists()


def test_install_creates_only_new_venv_and_continues_same_request(
    tmp_path,
    monkeypatch,
):
    calls, created = [], []

    class Builder:
        def __init__(self, *, with_pip):
            assert with_pip

        def create(self, path):
            created.append(path)

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 9)

    monkeypatch.setattr(workflow.venv, "EnvBuilder", Builder)
    monkeypatch.setattr(workflow.subprocess, "run", run)
    output, environment = tmp_path / "run", tmp_path / "isolated"
    arguments = [
        "--smoke",
        "--install",
        "--venv",
        str(environment),
        "--output",
        str(output),
        "--max-tokens",
        "3",
    ]
    assert workflow.main(arguments) == 9
    assert created == [environment]
    assert calls[0][:4] == [str(environment / "bin/python"), "-m", "pip", "install"]
    assert calls[0][0] != sys.executable
    assert calls[1][:4] == [
        str(environment / "bin/python"),
        "-P",
        "-m",
        "cacheslide_vllm.workflow",
    ]
    assert "--install" not in calls[1]
    assert "--smoke" in calls[1] and "3" in calls[1]
    assert not output.exists()


def test_install_rejects_existing_environment(tmp_path, monkeypatch, capsys):
    environment = tmp_path / "existing"
    environment.mkdir()
    marker = environment / "marker"
    marker.write_text("user environment")
    monkeypatch.setattr(
        workflow.venv, "EnvBuilder", lambda **k: pytest.fail("modified")
    )
    assert (
        workflow.main(
            [
                "--smoke",
                "--install",
                "--venv",
                str(environment),
                "--output",
                str(tmp_path / "new"),
            ]
        )
        == 2
    )
    assert "new path" in capsys.readouterr().err
    assert marker.read_text() == "user environment"


def test_install_requires_execution_intent(tmp_path, capsys):
    assert workflow.main(native_input(tmp_path, ("--install",))) == 2
    assert "requires --run or --smoke" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("arguments", [["--help"], ["--smoke", "--steps", "0"]])
def test_help_and_parser_error_without_scientific_dependencies(tmp_path, arguments):
    environment = dict(os.environ, PYTHONPATH=str(Path(workflow.__file__).parents[1]))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-P",
            "-m",
            "cacheslide_vllm.workflow",
            *arguments,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == (0 if arguments == ["--help"] else 2)
    assert "ModuleNotFoundError" not in result.stderr


def test_native_plan_is_standard_library_only(tmp_path):
    arguments = native_input(tmp_path)
    environment = dict(os.environ, PYTHONPATH=str(Path(workflow.__file__).parents[1]))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-P",
            "-m",
            "cacheslide_vllm.workflow",
            *arguments,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["executed"] is False
    assert not (tmp_path / "run").exists()


def test_shell_works_outside_repo_without_cwd_package_shadowing(tmp_path):
    fake = tmp_path / "cacheslide_vllm"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise RuntimeError('cwd shadow')")
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, CACHESLIDE_PYTHON=sys.executable)
    result = subprocess.run(
        [
            "bash",
            str(repository / "run_cacheslide_benchmark.sh"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--smoke" in result.stdout


def test_real_cpu_smoke_trains_calibrates_reuses_and_decodes(tmp_path, capsys):
    output = tmp_path / "smoke"
    assert (
        workflow.main(
            [
                "--smoke",
                "--output",
                str(output),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--max-tokens",
                "4",
                "--ccpe-position-policy",
                "mixed_bias_override",
                "--calibration-layer",
                "1",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    summary = receipt["result"]
    assert receipt["mode"] == "cpu_smoke"
    assert summary["correctness_passed"] is True
    assert summary["training_steps"] == 2 and summary["adapter_identity"]
    assert summary["decode_steps_per_request"] == 3
    assert summary["shifted_nonprefix_reuse_passed"] is True
    assert summary["native_vllm_executed"] is False
    assert summary["gpu_performance_claim"] is False
    assert summary["baseline_over_reuse_latency_ratio"] is None
    assert summary["timing_kind"] == "not_measured"
    records = [
        json.loads(line)
        for line in (output / "raw_outputs.jsonl").read_text().splitlines()
    ]
    assert len(records) == 5 and len(records[0]["generated_token_ids"]) == 4
    reused = [record for record in records if record["operation"] == "reuse"]
    assert [record["prompt_tokens"] for record in reused] == [8, 9]
    for record in reused:
        metrics = record["runtime_metrics"]
        assert metrics["cache_hit"] and not metrics["fallback"]
        assert metrics["decode_tokens"] == 3
        assert metrics["computed_token_layers"] < metrics["dense_token_layers"]
        assert len(metrics["paged_layers"]) == 6
    manifest = json.loads((output / "workflow.json").read_text())
    assert manifest["run_id"] == receipt["run_id"]
    assert manifest["status"] == "completed"
    assert [stage["name"] for stage in manifest["stages"]] == ["train", "calibrate"]
    assert all(stage["returncode"] == 0 for stage in manifest["stages"])
    assert "vllm" not in sys.modules


def test_default_cpu_smoke_verifies_strict_position_fallback(tmp_path, capsys):
    assert (
        workflow.main(
            [
                "--smoke",
                "--output",
                str(tmp_path / "strict-smoke"),
                "--warmup",
                "0",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)["result"]
    assert summary["correctness_passed"]
    assert summary["ccpe_position_policy"] == "strict_contextual"
    assert summary["calibration_layer"] == 0
    assert summary["guarded_fallback_requests"] > 0
    assert not summary["shifted_nonprefix_reuse_passed"]
    assert summary["baseline_over_reuse_latency_ratio"] is None
