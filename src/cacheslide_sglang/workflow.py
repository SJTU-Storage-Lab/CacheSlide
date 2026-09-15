"""One-command SGLang preparation and benchmark with explicit execution gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from cacheslide_core.config import CacheSlideSettings
from cacheslide_core.inputs import read_cases


class BenchmarkValidationError(RuntimeError):
    """Measurements exist, but are not a successful benchmark."""


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parser():
    p = argparse.ArgumentParser(description="CacheSlide + pinned SGLang workflow")
    for name in (
        "model",
        "input",
        "train-input",
        "calibration-input",
        "seed-input",
        "adapter",
        "profiles",
        "output",
        "cache-root",
        "venv",
    ):
        p.add_argument("--" + name)
    for name in ("smoke", "run", "install"):
        p.add_argument("--" + name, action="store_true")
    p.add_argument("--steps", type=positive)
    p.add_argument("--rank", type=positive)
    p.add_argument("--max-positions", type=positive)
    p.add_argument("--max-tokens", type=positive, default=4)
    p.add_argument("--context-length", "--max-model-len", type=positive, default=4096)
    p.add_argument("--max-total-tokens", type=positive)
    p.add_argument("--query-chunk-size", type=positive, default=64)
    p.add_argument("--max-profile-elements", type=positive, default=4_194_304)
    p.add_argument("--cpu-budget-bytes", type=positive, default=536_870_912)
    p.add_argument("--disk-budget-bytes", type=positive, default=2_147_483_648)
    p.add_argument("--warmup", type=nonnegative, default=1)
    p.add_argument("--repeats", type=positive, default=3)
    p.add_argument("--calibration-layer", type=nonnegative, default=0)
    p.add_argument(
        "--ccpe-position-policy",
        default="strict_contextual",
        choices=("strict_contextual", "mixed_bias_override"),
    )
    p.add_argument(
        "--selected-attention",
        default="updated_and_self",
        choices=("updated_and_self", "full_causal"),
    )
    p.add_argument(
        "--convergence-mode",
        default="paper_cosine_lt",
        choices=("paper_cosine_lt", "distance_lt"),
    )
    p.add_argument(
        "--weight-update",
        default="previous_layer",
        choices=("previous_layer", "same_layer"),
    )
    p.add_argument("--correction-fraction", type=float, default=0.26)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    p.add_argument("--mem-fraction-static", type=float, default=0.5)
    p.add_argument("--train-device", default="cpu")
    p.add_argument("--calibration-device", default="cpu")
    return p


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stage(root, name, arguments):
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    print(f"CacheSlide stage: {name}", file=sys.stderr, flush=True)
    with (
        (logs / (name + ".stdout.log")).open("x") as stdout,
        (logs / (name + ".stderr.log")).open("x") as stderr,
    ):
        result = subprocess.run(
            [sys.executable, "-P", "-m", "cacheslide_core.commands", *arguments],
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {logs}. No backend fallback.")


def settings_for(args, root):
    return CacheSlideSettings(
        artifact_path=args.adapter,
        profile_path=args.profiles,
        cache_root=args.cache_root or str(root / "cache"),
        cpu_budget_bytes=args.cpu_budget_bytes,
        disk_budget_bytes=args.disk_budget_bytes,
        max_prompt_tokens=args.context_length,
        max_profile_elements=args.max_profile_elements,
        query_chunk_size=args.query_chunk_size,
        calibration_layer=args.calibration_layer,
        ccpe_position_policy=args.ccpe_position_policy,
        selected_attention=args.selected_attention,
        weight_update=args.weight_update,
        correction_fraction=args.correction_fraction,
        convergence_mode=args.convergence_mode,
    )


def normalize(args):
    if not args.output:
        raise ValueError("--output must name a new result directory")
    for name in (
        "model",
        "input",
        "train_input",
        "calibration_input",
        "seed_input",
        "adapter",
        "profiles",
        "output",
        "cache_root",
        "venv",
    ):
        value = getattr(args, name)
        if value is not None:
            path = Path(value).expanduser()
            if name in {"output", "cache_root", "venv"} and (
                path.exists() or path.is_symlink()
            ):
                raise ValueError(f"--{name.replace('_', '-')} must be a new path")
            setattr(args, name, str(path.resolve()))
    if not args.smoke and (not args.model or not args.input):
        raise ValueError("native workflow requires --model and --input")
    if args.smoke and any(
        (
            args.model,
            args.input,
            args.train_input,
            args.calibration_input,
            args.seed_input,
            args.adapter,
            args.profiles,
        )
    ):
        raise ValueError(
            "--smoke uses its own random fixture, not supplied model artifacts"
        )
    if args.smoke and any(
        device != "cpu" for device in (args.train_device, args.calibration_device)
    ):
        raise ValueError("--smoke is CPU-only")
    output = Path(args.output)
    if args.cache_root:
        cache = Path(args.cache_root)
        if cache == output or cache in output.parents:
            raise ValueError("--cache-root must not equal or contain --output")
    if args.venv or args.install:
        environment = Path(args.venv or output.with_name(".sglang-" + output.name))
        if environment.exists() or environment.is_symlink():
            raise ValueError("installation environment must be new")
        paths = [output] + ([Path(args.cache_root)] if args.cache_root else [])
        if any(
            environment == path
            or environment in path.parents
            or path in environment.parents
            for path in paths
        ):
            raise ValueError("--venv must be separate from result and cache paths")
    if args.profiles and not args.adapter:
        raise ValueError("--profiles requires its verified --adapter")
    if args.install and not (args.smoke or args.run):
        raise ValueError("--install requires --smoke or --run")
    for name in ("lr", "mem_fraction_static"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be positive and finite")
    if args.mem_fraction_static >= 1:
        raise ValueError("mem_fraction_static must be < 1")
    args.steps = args.steps or (2 if args.smoke else 20)
    args.rank = args.rank or (2 if args.smoke else 8)
    args.max_positions = args.max_positions or (16 if args.smoke else 256)
    # Validate numerical policies without importing Torch or starting an engine.
    CacheSlideSettings(
        "/validation/adapter",
        "/validation/cache",
        correction_fraction=args.correction_fraction,
        calibration_layer=args.calibration_layer,
    )
    return args


def install_and_reexec(args, original):
    project = Path(__file__).resolve().parents[2]
    if not (project / "pyproject.toml").is_file():
        raise ValueError(
            "--install requires the repository launcher, not an installed wheel"
        )
    environment = Path(
        args.venv
        or str(Path(args.output).with_name(".sglang-" + Path(args.output).name))
    )
    if environment.exists() or environment.is_symlink():
        raise ValueError("installation environment must be new")
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment / "bin/python"
    target = str(project) + ("[test]" if args.smoke else "[engine,test]")
    subprocess.run([str(python), "-m", "pip", "install", target], check=True)
    child = [
        value
        for value in original
        if value != "--install" and not value.startswith("--venv=")
    ]
    # venv is now populated, so it is no longer a new-path input to the child.
    if "--venv" in child:
        index = child.index("--venv")
        del child[index : index + 2]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(python), "-P", "-m", "cacheslide_sglang.workflow", *child],
        env=env,
        check=False,
    ).returncode


def run(args):
    root = Path(args.output)
    cases = None
    seeds = None
    if not args.smoke:
        cases = read_cases(args.input)
        seeds = read_cases(args.seed_input) if args.seed_input else cases
        capacity = min(
            args.context_length, args.max_total_tokens or args.context_length
        )
        for case in [*cases, *seeds]:
            if len(case.plan.token_ids) + args.max_tokens > capacity:
                raise ValueError(
                    f"case {case.case_id!r} prompt plus output "
                    f"exceeds token capacity {capacity}"
                )
    if not (args.run or args.smoke):
        return {
            "executed": False,
            "requires": "--run",
            "engine": "sglang==0.5.19",
            "cases": len(cases),
            "output": str(root),
            "stages": ["check-engine", "train", "calibrate", "bench"],
        }
    if not args.smoke:
        from .compat import verify_compatibility

        verify_compatibility()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "engine": "sglang",
        "mode": "cpu_smoke" if args.smoke else "native",
        "arguments": vars(args).copy(),
    }
    save(root / "workflow.json", manifest)
    try:
        if args.smoke:
            from .smoke import prepare_fixture

            prepare_fixture(args, root)
        if not args.adapter:
            args.adapter = str(root / "adapter")
            stage(
                root,
                "train",
                [
                    "train",
                    "--model",
                    args.model,
                    "--input",
                    args.train_input or args.input,
                    "--output",
                    args.adapter,
                    "--steps",
                    str(args.steps),
                    "--rank",
                    str(args.rank),
                    "--lr",
                    str(args.lr),
                    "--max-positions",
                    str(args.max_positions),
                    "--device",
                    args.train_device,
                    "--query-chunk-size",
                    str(args.query_chunk_size),
                ],
            )
        if not args.profiles:
            args.profiles = str(root / "profiles")
            stage(
                root,
                "calibrate",
                [
                    "calibrate",
                    "--model",
                    args.model,
                    "--adapter",
                    args.adapter,
                    "--input",
                    args.calibration_input or args.input,
                    "--output",
                    args.profiles,
                    "--device",
                    args.calibration_device,
                    "--max-elements",
                    str(args.max_profile_elements),
                ],
            )
        settings = settings_for(args, root)
        if args.smoke:
            from .smoke import run_smoke

            records, summary = run_smoke(args, settings)
        else:
            from .benchmark import run_pairs
            from .integration import create_engine

            with create_engine(
                args.model,
                settings,
                receipt_dir=root / "receipts",
                dtype=args.dtype,
                context_length=args.context_length,
                max_total_tokens=args.max_total_tokens,
                mem_fraction_static=args.mem_fraction_static,
            ) as engine:
                records, summary = run_pairs(
                    engine,
                    cases,
                    seeds,
                    max_tokens=args.max_tokens,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
        with (root / "raw_outputs.jsonl").open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        save(root / "summary.json", summary)
        if (
            summary.get("validation_passed", summary.get("correctness_passed"))
            is not True
        ):
            raise BenchmarkValidationError(
                "benchmark validation failed; inspect summary.json "
                "and raw_outputs.jsonl"
            )
        manifest["status"] = "completed"
        return summary
    except BaseException as exc:
        manifest.update(
            status="validation_failed"
            if isinstance(exc, BenchmarkValidationError)
            else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        save(root / "workflow.json", manifest)


def main(argv=None):
    original = list(sys.argv[1:] if argv is None else argv)
    if not original:
        parser().print_help()
        return 0
    args = parser().parse_args(original)
    try:
        normalize(args)
        if args.install:
            return install_and_reexec(args, original)
        result = run(args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
