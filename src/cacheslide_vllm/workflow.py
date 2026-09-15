"""One explicit workflow: train, calibrate, then benchmark the same warm engine.

Native execution delegates to the existing CLI in separate processes. The CPU
smoke is a correctness exercise with random tiny weights, never a GPU benchmark.
Importing this module, --help, and native planning need only the standard library.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import uuid
import venv
from datetime import UTC, datetime
from pathlib import Path


class WorkflowError(RuntimeError):
    """A failed workflow is not silently replaced by a different backend."""


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="run_cacheslide_benchmark.sh",
        description=(
            "Local checkpoint → real adapter training → CCPE calibration → "
            "populate/warmup → matched native recompute/reuse benchmark. "
            "No model is downloaded. Native mode only plans until --run is supplied."
        ),
        epilog=(
            "CPU correctness check: --smoke --output /new/smoke. Native: "
            "--model /local/model --input cases.jsonl --output /new/run --run. "
            "Use --adapter and optionally --profiles to reuse verified artifacts."
        ),
    )
    result.add_argument("--smoke", action="store_true", help="run tiny CPU correctness")
    result.add_argument(
        "--run", action="store_true", help="execute native GPU workflow"
    )
    result.add_argument("--model", help="existing local safetensors model directory")
    result.add_argument("--input", help="measured typed token/chunk JSONL")
    result.add_argument("--train-input", help="training JSONL; defaults to --input")
    result.add_argument(
        "--calibration-input", help="calibration JSONL; defaults to --input"
    )
    result.add_argument("--seed-input", help="populate JSONL; CLI defaults per layout")
    result.add_argument("--adapter", help="existing adapter; skip training")
    result.add_argument("--profiles", help="existing profiles; requires --adapter")
    result.add_argument(
        "--output", help="new directory; default ./cacheslide-runs/<unique run id>"
    )
    result.add_argument(
        "--cache-root", help="new cache directory (e.g. SSD); default <output>/cache"
    )
    result.add_argument(
        "--install",
        action="store_true",
        help="create a NEW isolated venv, install dependencies, then run this request",
    )
    result.add_argument(
        "--venv", help="new install venv; default sibling <output>.venv; never reused"
    )
    result.add_argument("--steps", type=_positive, help="default native 20; smoke 2")
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--rank", type=_positive, help="default native 8; smoke 2")
    result.add_argument(
        "--max-positions", type=_positive, help="default native 256; smoke 16"
    )
    result.add_argument("--train-device", default="cpu")
    result.add_argument("--calibration-device", default="cpu")
    result.add_argument("--profile-version", default="v1")
    result.add_argument("--max-profile-elements", type=_positive, default=1_000_000)
    result.add_argument("--max-model-len", type=_positive, default=4096)
    result.add_argument(
        "--max-tokens", type=_positive, help="default native 1; smoke 4"
    )
    result.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    result.add_argument("--model-runner", choices=("v2", "v1"), default="v2")
    result.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    result.add_argument("--cpu-budget-bytes", type=_positive, default=536_870_912)
    result.add_argument("--disk-budget-bytes", type=_positive, default=2_147_483_648)
    result.add_argument("--query-chunk-size", type=_positive, default=64)
    result.add_argument("--calibration-layer", type=_nonnegative, default=1)
    result.add_argument("--correction-fraction", type=float, default=0.26)
    result.add_argument(
        "--convergence-mode",
        choices=("paper_cosine_lt", "distance_lt"),
        default="paper_cosine_lt",
    )
    result.add_argument(
        "--weight-update",
        choices=("previous_layer", "same_layer"),
        default="previous_layer",
    )
    result.add_argument(
        "--selected-attention",
        choices=("updated_and_self", "full_causal"),
        default="updated_and_self",
    )
    result.add_argument("--warmup", type=_nonnegative, default=1)
    result.add_argument("--repeats", type=_positive, default=3)
    result.add_argument("--token-f1", action="store_true")
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_path(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} must be a new path: {path}")


def prepare(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve paths and validate intent without importing torch or creating files."""
    if sys.version_info < (3, 12):  # noqa: UP036 -- source script checks its interpreter
        raise WorkflowError("Python 3.12 or newer is required; set CACHESLIDE_PYTHON")
    args.run_id = uuid.uuid4().hex
    args.mode = "cpu_smoke" if args.smoke else "native_vllm"
    output = args.output or str(
        Path.cwd()
        / "cacheslide-runs"
        / f"{args.mode}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{args.run_id[:8]}"
    )
    # Check the original spelling before resolve() follows a dangling symlink.
    _new_path(Path(output).expanduser(), "output")
    args.output = str(Path(output).expanduser().resolve())
    _new_path(Path(args.output), "output")
    cache = Path(args.cache_root or str(Path(args.output) / "cache")).expanduser()
    _new_path(cache, "cache-root")
    cache = cache.resolve()
    _new_path(cache, "cache-root")
    if (cache == Path(args.output) or Path(args.output) in cache.parents) and (
        cache != Path(args.output) / "cache"
    ):
        raise ValueError("inside --output, --cache-root must be <output>/cache")
    args.cache_root = str(cache)
    args.steps = args.steps if args.steps is not None else (2 if args.smoke else 20)
    args.rank = args.rank if args.rank is not None else (2 if args.smoke else 8)
    args.max_positions = args.max_positions or (16 if args.smoke else 256)
    args.max_tokens = args.max_tokens or (4 if args.smoke else 1)
    if args.max_positions < 2:
        raise ValueError("--max-positions must be at least 2")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(args.gpu_memory_utilization) or not (
        0 < args.gpu_memory_utilization <= 1
    ):
        raise ValueError("--gpu-memory-utilization must be in (0,1]")
    if not math.isfinite(args.correction_fraction) or not (
        0 <= args.correction_fraction <= 1
    ):
        raise ValueError("--correction-fraction must be in [0,1]")
    if args.install and not (args.run or args.smoke):
        raise ValueError(
            "--install requires --run or --smoke; planning makes no writes"
        )
    if args.profiles and not args.adapter:
        raise ValueError("--profiles requires --adapter; profiles bind exact weights")
    inputs = (
        "model",
        "input",
        "train_input",
        "calibration_input",
        "seed_input",
        "adapter",
        "profiles",
    )
    if args.smoke:
        if any(getattr(args, name) for name in inputs):
            raise ValueError(
                "--smoke creates its own tiny model/data; omit model/artifacts"
            )
        if args.train_device != "cpu" or args.calibration_device != "cpu":
            raise ValueError("--smoke is CPU-only")
        if args.calibration_layer >= 5:
            raise ValueError("--smoke needs a sparse layer after calibration (0..4)")
        if args.max_tokens + 9 > args.max_model_len:
            raise ValueError("smoke prompt plus max-tokens exceeds max-model-len")
    else:
        if not args.model or not args.input:
            raise ValueError(
                "native workflow requires --model /local/model --input cases.jsonl; "
                "no model is downloaded. Use --smoke for the self-contained CPU check"
            )
        for name in inputs:
            value = getattr(args, name)
            if value:
                path = Path(value).expanduser().resolve()
                is_directory = name in {"model", "adapter", "profiles"}
                if not (path.is_dir() if is_directory else path.is_file()):
                    raise ValueError(
                        f"--{name.replace('_', '-')} must exist locally: {path}"
                    )
                setattr(args, name, str(path))
        if not (Path(args.model) / "config.json").is_file():
            raise ValueError("--model must contain config.json and local safetensors")
        args.train_input = args.train_input or args.input
        args.calibration_input = args.calibration_input or args.input
        _validate_cases(args)
    return args


def _validate_cases(args: argparse.Namespace) -> None:
    from .cli import read_cases

    cases = read_cases(args.input)
    seeds = read_cases(args.seed_input) if args.seed_input else cases
    calibration = read_cases(args.calibration_input) if not args.profiles else []
    for case in [*cases, *seeds, *calibration]:
        if not case.plan.fixed_indices:
            raise ValueError("native workflow requires reusable fixed chunks")
    for case in [*cases, *seeds]:
        if len(case.token_ids) + args.max_tokens > args.max_model_len:
            raise ValueError("prompt plus max-tokens exceeds max-model-len")

    def key(case):
        return case.plan.cache_key("workflow-layout", 0)

    needed = {key(case) for case in cases}
    if not needed.issubset({key(case) for case in seeds}):
        raise ValueError("each measured layout requires a matching populate seed")
    if calibration and not needed.issubset({key(case) for case in calibration}):
        raise ValueError("each measured layout requires matching calibration data")
    if not args.adapter:
        if any(
            len(case.token_ids) < 2
            for case in read_cases(args.train_input, require_plan=False)
        ):
            raise ValueError("each training sequence needs at least two tokens")


def _options(**values) -> list[str]:
    result = []
    for name, value in values.items():
        if value is not None:
            result.extend(("--" + name.replace("_", "-"), str(value)))
    return result


def native_stages(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Use the existing CLI arguments, not a second implementation of benchmark."""
    root = Path(args.output)
    adapter = args.adapter or str(root / "adapter")
    profiles = args.profiles or str(root / "profiles")
    stages = [("check-engine", ["check-engine"])]
    if not args.adapter:
        stages.append(
            (
                "train",
                [
                    "train",
                    *_options(
                        model=args.model,
                        input=args.train_input,
                        output=adapter,
                        steps=args.steps,
                        lr=args.lr,
                        rank=args.rank,
                        max_positions=args.max_positions,
                        query_chunk_size=args.query_chunk_size,
                        device=args.train_device,
                    ),
                ],
            )
        )
    if not args.profiles:
        stages.append(
            (
                "calibrate",
                [
                    "calibrate",
                    *_options(
                        model=args.model,
                        adapter=adapter,
                        input=args.calibration_input,
                        output=profiles,
                        device=args.calibration_device,
                        max_elements=args.max_profile_elements,
                        profile_version=args.profile_version,
                    ),
                ],
            )
        )
    bench = [
        "bench",
        *_options(
            model=args.model,
            adapter=adapter,
            profiles=profiles,
            input=args.input,
            seed_input=args.seed_input,
            output=root / "benchmark",
            cache_root=args.cache_root,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            dtype=args.dtype,
            model_runner=args.model_runner,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_budget_bytes=args.cpu_budget_bytes,
            disk_budget_bytes=args.disk_budget_bytes,
            max_profile_elements=args.max_profile_elements,
            query_chunk_size=args.query_chunk_size,
            calibration_layer=args.calibration_layer,
            correction_fraction=args.correction_fraction,
            convergence_mode=args.convergence_mode,
            weight_update=args.weight_update,
            selected_attention=args.selected_attention,
            warmup=args.warmup,
            repeats=args.repeats,
            backend="native",
        ),
        "--run",
    ]
    if args.token_f1:
        bench.append("--token-f1")
    stages.append(("benchmark", bench))
    return stages


def _environment() -> dict[str, str]:
    result = dict(os.environ)
    source = str(Path(__file__).resolve().parent.parent)
    result["PYTHONPATH"] = source + os.pathsep + result.get("PYTHONPATH", "")
    result["PYTHONSAFEPATH"] = "1"
    return result


def _dependencies(*, native: bool) -> None:
    needed = ["torch", "safetensors"] + (["vllm"] if native else [])
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:
        raise WorkflowError(
            f"Missing dependencies: {', '.join(missing)}. "
            "Select an existing environment "
            "with CACHESLIDE_PYTHON, or repeat with --install to create a NEW isolated "
            "venv (native pins vllm==0.29.0). The current environment was not changed."
        )


def _install(args: argparse.Namespace, argv: list[str]) -> int:
    """Only create fresh environments; never pip-install into the calling one."""
    location = Path(args.venv or (args.output + ".venv")).expanduser()
    _new_path(location, "installation venv")
    location = location.resolve()
    output = Path(args.output)
    if location == output or output in location.parents:
        raise ValueError("installation venv must be outside the new output directory")
    repository = Path(__file__).resolve().parents[2]
    if not (repository / "pyproject.toml").is_file():
        raise WorkflowError(
            "--install needs a source checkout; use an existing environment"
        )
    location.mkdir(parents=True, exist_ok=False)
    print(f"Creating isolated environment: {location}", file=sys.stderr, flush=True)
    venv.EnvBuilder(with_pip=True).create(location)
    python = location / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirement = str(repository) + ("" if args.smoke else "[engine]")
    requirements = [requirement]
    if not args.smoke:
        from .compat import compatibility_manifest

        requirements.append("vllm==" + compatibility_manifest()["vllm_version"])
    subprocess.run([str(python), "-m", "pip", "install", *requirements], check=True)
    # Keep the initially chosen unique output path when the caller omitted one.
    forwarded = [item for item in argv if item != "--install"]
    if "--output" not in forwarded and not any(
        item.startswith("--output=") for item in forwarded
    ):
        forwarded.extend(("--output", args.output))
    return subprocess.run(
        [str(python), "-P", "-m", "cacheslide_vllm.workflow", *forwarded],
        env=_environment(),
        check=False,
    ).returncode


def _save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _run_stage(
    name: str,
    arguments: list[str],
    root: Path,
    manifest: dict,
    *,
    module: str = "cacheslide_vllm.cli",
) -> None:
    command = [sys.executable, "-P", "-m", module, *arguments]
    receipt = {"name": name, "command": command, "started_at": _now()}
    manifest["stages"].append(receipt)
    _save(root / "workflow.json", manifest)
    print(
        f"CacheSlide stage: {name} (logs: {root / 'logs'})", file=sys.stderr, flush=True
    )
    with (
        (root / "logs" / f"{name}.stdout.log").open("x") as stdout,
        (root / "logs" / f"{name}.stderr.log").open("x") as stderr,
    ):
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            env=_environment(),
            check=False,
        )
    receipt.update(returncode=completed.returncode, completed_at=_now())
    _save(root / "workflow.json", manifest)
    if completed.returncode:
        raise WorkflowError(
            f"{name} failed (exit {completed.returncode}); inspect "
            f"{root / 'logs' / (name + '.stderr.log')}. No backend fallback was used."
        )


def execute(args: argparse.Namespace) -> dict:
    from .compat import compatibility_manifest

    contract = compatibility_manifest()
    if not args.smoke and not args.run:
        return {
            "executed": False,
            "mode": args.mode,
            "requires": "--run",
            "vllm_version": contract["vllm_version"],
            "output": args.output,
            "model_downloaded": False,
            "stages": [
                {"name": name, "cli_arguments": command}
                for name, command in native_stages(args)
            ],
            "comparison": "same trained CoPE/LoRA model, same warm native engine",
            "timing_kind": "offline generation; not streaming TTFT",
        }
    _dependencies(native=not args.smoke)
    if not args.smoke:
        _validate_cases(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "mode": args.mode,
        "status": "running",
        "started_at": _now(),
        "python": sys.executable,
        "vllm_contract": {
            "version": contract["vllm_version"],
            "commit": contract["vllm_git_commit"],
        },
        "arguments": vars(args),
        "stages": [],
        "model_downloaded": False,
        "gpu_performance_claim": False,
        "is_streaming_ttft": False,
    }
    _save(root / "workflow.json", manifest)
    try:
        if args.smoke:
            from .workflow_smoke import run_smoke

            result = run_smoke(args, root, manifest, _run_stage)
        else:
            for name, command in native_stages(args):
                _run_stage(name, command, root, manifest)
            result = json.loads((root / "benchmark" / "summary.json").read_text())
            if not all(
                result.get(key) is True
                for key in (
                    "executed",
                    "baseline_validation_passed",
                    "reuse_validation_passed",
                    "latency_validation_passed",
                )
            ):
                raise WorkflowError(
                    "benchmark receipts/lengths/reuse did not validate; raw results "
                    "are preserved, and the workflow is not successful"
                )
        manifest.update(status="completed", completed_at=_now(), result=result)
    except BaseException as error:
        manifest.update(status="failed", completed_at=_now(), error=str(error))
        _save(root / "workflow.json", manifest)
        raise
    _save(root / "workflow.json", manifest)
    return {
        "executed": True,
        "mode": args.mode,
        "run_id": args.run_id,
        "output": str(root),
        "workflow": str(root / "workflow.json"),
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    argument_parser = parser()
    if not arguments:
        argument_parser.print_help()
        return 2
    try:
        args = prepare(argument_parser.parse_args(arguments))
        if args.install:
            return _install(args, arguments)
        result = execute(args)
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        AssertionError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
