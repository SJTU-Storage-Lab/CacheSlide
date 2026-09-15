#!/usr/bin/env python3
"""One-key verified HotpotQA continued pretraining, not a native benchmark.

No dependency installation, GPU-keeper changes, remote code or implicit
downloads. The complete pinned model is verified before any GPU subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_MODULE = "cacheslide_core.training_pipeline"
MODEL_NAME = "Mistral-7B-Instruct-v0.2"
MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"


def _sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _downloader():
    spec = importlib.util.spec_from_file_location(
        "cacheslide_pretraining_asset_verifier",
        ROOT / "scripts/download_paper_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_complete_model(
    asset_root: Path, spec_path: Path, trusted_path: Path | None = None
) -> dict:
    """Read-only official-manifest verification, including every indexed shard.

    A config/tokenizer-only snapshot is never enough. This deliberately imports
    no torch, transformers, tokenizer or GPU library. Integrity primitives come
    from the same local downloader used to publish the pinned assets.
    """
    downloader = _downloader()
    trusted_path = trusted_path or ROOT / "configs/paper_model_integrity.json"
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise ValueError("asset root must be an existing non-symlink directory")
    if asset_root.stat().st_uid != os.getuid():
        raise ValueError("asset root must belong to the current user")
    manifest_path = asset_root / "download_manifest.json"
    owner_path = asset_root / ".owner.json"
    for path in (manifest_path, owner_path, spec_path, trusted_path):
        downloader.regular(path)
    manifest = json.loads(manifest_path.read_text())
    spec = json.loads(spec_path.read_text())
    identity = manifest.get("identity", {})
    if (
        identity.get("owner") != downloader.OWNER
        or identity.get("spec_sha256") != _sha256(spec_path)
        or identity.get("selection") not in {"all", "models"}
        or json.loads(owner_path.read_text()) != identity
        or manifest.get("spec") != spec
    ):
        raise ValueError("asset manifest owner/spec identity mismatch")
    definitions = [
        item for item in spec.get("models", []) if item.get("name") == MODEL_NAME
    ]
    if len(definitions) != 1 or (
        definitions[0].get("repo") != MODEL_REPO
        or definitions[0].get("revision") != MODEL_REVISION
    ):
        raise ValueError("launcher requires the pinned official Mistral model")
    trusted = json.loads(trusted_path.read_text())
    if trusted.get("repo") != MODEL_REPO or trusted.get("revision") != MODEL_REVISION:
        raise ValueError("trusted model integrity anchor has the wrong repo/revision")
    prefix = f"models/{MODEL_NAME}/"
    entries, seen = [], set()
    for item in manifest.get("files", []):
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("manifest paths must be distinct strings")
        seen.add(relative)
        if relative.startswith(prefix):
            name = relative.removeprefix(prefix)
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ValueError("model assets must have simple file names")
            if (
                type(item.get("size")) is not int
                or item["size"] <= 0
                or not (
                    isinstance(item.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    or isinstance(item.get("git_blob_sha1"), str)
                    and re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha1"])
                )
            ):
                raise ValueError("model file lacks an official size/hash")
            expected_url = (
                f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
            )
            if item.get("url") != expected_url:
                raise ValueError("model asset URL is not the pinned official source")
            entries.append(item)

    def identities(items):
        result = {}
        for item in items:
            path = item.get("path")
            if not isinstance(path, str) or path in result:
                raise ValueError("trusted model integrity paths must be distinct")
            result[path] = {
                key: item.get(key)
                for key in ("path", "size", "sha256", "git_blob_sha1")
            }
        return result

    if not trusted.get("files") or identities(entries) != identities(trusted["files"]):
        raise ValueError(
            "local model manifest differs from trusted official integrity anchor"
        )
    names = {entry["path"].removeprefix(prefix) for entry in entries}
    if not {"config.json", "tokenizer.json"} <= names:
        raise ValueError("complete model config and tokenizer are required")
    if not ({"model.safetensors", "model.safetensors.index.json"} & names):
        raise ValueError("model weights are incomplete; no weights/index in manifest")
    verified = []
    for item in entries:
        path = downloader.safe_child(asset_root, item["path"], create=False)
        if not path.exists():
            raise FileNotFoundError(
                f"model incomplete before GPU training: {item['path']}"
            )
        verified.append(downloader.verify(path, item))
    model_dir = asset_root / "models" / MODEL_NAME
    if "model.safetensors.index.json" in names:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("model safetensors index has no weight map")
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in mapping.values()
        ):
            raise ValueError("model index contains invalid shard names")
        shards = set(mapping.values())
        if not shards <= names:
            raise ValueError("model index references shards missing from the manifest")
    else:
        shards = {"model.safetensors"}
    if not shards:
        raise ValueError("a complete model weight set is required before training")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "mistral" or config.get("sliding_window"):
        raise ValueError(
            "the candidate must be full-attention Mistral, not altered SWA"
        )
    return {
        "model_dir": str(model_dir),
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "manifest_sha256": _sha256(manifest_path),
        "trusted_manifest_sha256": _sha256(trusted_path),
        "weight_shards": sorted(shards),
        "verified_files": verified,
        "official_integrity_verified": True,
    }


def _write_state(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stage(state, output, name, *, command=None, action=None):
    record = {"name": name, "status": "running", "started_at": time.time()}
    if command is not None:
        record["command"] = command
    state["stages"].append(record)
    _write_state(output / "status.json", state)
    logs = output / "logs"
    try:
        with (
            (logs / f"{name}.stdout.log").open("x") as stdout,
            (logs / f"{name}.stderr.log").open("x") as stderr,
        ):
            if action is not None:
                result = action()
                json.dump(result, stdout, indent=2, allow_nan=False)
                stdout.write("\n")
            else:
                env = os.environ.copy()
                old_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(ROOT / "src") + (
                    os.pathsep + old_path if old_path else ""
                )
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
                record["exit_code"] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"stage {name} exited {completed.returncode}; see logs"
                    )
                result = None
        record["status"] = "complete"
        return result
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        with (logs / f"{name}.stderr.log").open("a") as stderr:
            stderr.write(record["error"] + "\n")
        raise
    finally:
        record["finished_at"] = time.time()
        _write_state(output / "status.json", state)


def _completed_adapter(output, expected_model, expected_steps):
    training = output / "training"
    report = json.loads((training / "report.json").read_text())
    artifact = training / "adapter"
    if report.get("status") != "complete" or report.get("adapter") != str(artifact):
        raise ValueError("training did not report the expected completed adapter")
    manifest = json.loads((artifact / "manifest.json").read_text())
    if (
        manifest.get("format") != "cacheslide-adapter-v1"
        or manifest.get("trained") is not True
        or type(manifest.get("training_steps")) is not int
        or manifest["training_steps"] < 1
        or manifest.get("adapter_sha256") != _sha256(artifact / "adapter.safetensors")
    ):
        raise ValueError("completed adapter manifest/checksum mismatch")
    if (
        type(report.get("optimizer_steps")) is not int
        or report["optimizer_steps"] != expected_steps
        or manifest["training_steps"] != expected_steps
        or type(report.get("training_tokens")) is not int
        or report["training_tokens"] < 1
        or manifest.get("training_tokens") != report["training_tokens"]
    ):
        raise ValueError(
            "adapter/report optimizer progress differs from requested training"
        )
    hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in expected_model["verified_files"]
    }
    expected_files = {"config.json", *expected_model["weight_shards"]}
    if "model.safetensors.index.json" in hashes:
        expected_files.add("model.safetensors.index.json")
    if manifest.get("base_files") != {name: hashes[name] for name in expected_files}:
        raise ValueError("adapter is not tied to the verified official backbone")
    # Canonical native loader, CPU only: hashes alone do not establish valid
    # safetensors, adapter geometry/finiteness or a native-mountable bundle.
    source = str(ROOT / "src")
    sys.path.insert(0, source)
    try:
        artifacts = importlib.import_module(
            TRAINING_MODULE.rpartition(".")[0] + ".artifacts"
        )
        bundle = artifacts.AdapterBundle(artifact, expected_model["model_dir"])
    finally:
        sys.path.remove(source)
    return {
        "adapter": str(artifact),
        "training_steps": manifest["training_steps"],
        "training_report": str(training / "report.json"),
        "adapter_identity": bundle.identity,
        "canonical_adapter_load_verified": True,
        "next_step": (
            f"Mount this combined CoPE+LoRA artifact with --adapter {artifact}"
        ),
        "quality_validated": False,
        "paper_results_reproduced": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True, help="explicit cpu or cuda:N")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pretraining_engineering_pilot.json",
    )
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/paper_assets.json")
    parser.add_argument(
        "--trusted-model-manifest",
        type=Path,
        default=ROOT / "configs/paper_model_integrity.json",
        help="trusted official per-file hash anchor; never derived from local receipts",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="explicitly download the pinned spec first; never implicit",
    )
    parser.add_argument("--training-count", type=int, default=128)
    parser.add_argument("--nll-validation-count", type=int, default=16)
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--evaluation-count", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=9, help="data selection seed")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda:[0-9]+", args.device):
        parser.error("--device must explicitly name cpu or cuda:N")
    if (
        any(
            getattr(args, name) < 1
            for name in (
                "training_count",
                "nll_validation_count",
                "calibration_count",
                "evaluation_count",
                "max_prompt_tokens",
            )
        )
        or args.seed < 0
    ):
        parser.error("positive case/prompt counts and nonnegative seed are required")
    # Check a symlink before resolve: a dangling symlink is not a new directory.
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new directory")
    output, asset_root = args.output.absolute(), args.asset_root.absolute()
    if output.resolve().is_relative_to(
        asset_root.resolve()
    ) or asset_root.resolve().is_relative_to(output.resolve()):
        parser.error("output and asset directories must not overlap")
    config_path, spec_path = (
        args.config.resolve(strict=True),
        args.spec.resolve(strict=True),
    )
    config = json.loads(config_path.read_text())
    length = config.get("sequence_length", 2048)
    expected_steps = config.get("max_steps", 1000)
    if type(length) is not int or length < 1:
        parser.error("training config sequence_length must be a positive integer")
    if type(expected_steps) is not int or expected_steps < 1:
        parser.error("training config max_steps must be a positive integer")
    trusted_path = args.trusted_model_manifest.absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    state = {
        "format": "cacheslide-paper-pretraining-run-v1",
        "status": "running",
        "started_at": time.time(),
        "device": args.device,
        "training_config": str(config_path),
        "config_sha256": _sha256(config_path),
        "asset_root": str(asset_root),
        "trusted_model_manifest": str(trusted_path),
        "stages": [],
        "manages_gpu_keeper": False,
        "starts_native_benchmark": False,
        "paper_results_reproduced": False,
    }
    _write_state(output / "status.json", state)
    try:
        if args.download:
            _stage(
                state,
                output,
                "download",
                command=[
                    sys.executable,
                    str(ROOT / "scripts/download_paper_assets.py"),
                    "--root",
                    str(asset_root),
                    "--spec",
                    str(spec_path),
                    "--only",
                    "all",
                ],
            )
        verified = _stage(
            state,
            output,
            "verify_model",
            action=lambda: verify_complete_model(asset_root, spec_path, trusted_path),
        )
        data = output / "data"
        _stage(
            state,
            output,
            "prepare_data",
            command=[
                sys.executable,
                str(ROOT / "scripts/prepare_paper_data.py"),
                "--asset-root",
                str(asset_root),
                "--output",
                str(data),
                "--training-count",
                str(args.training_count),
                "--nll-validation-count",
                str(args.nll_validation_count),
                "--calibration-count",
                str(args.calibration_count),
                "--evaluation-count",
                str(args.evaluation_count),
                "--max-prompt-tokens",
                str(args.max_prompt_tokens),
                "--seed",
                str(args.seed),
            ],
        )
        for stage, source, directory in (
            ("prepare_train", "train.corpus.jsonl", "train_tokens"),
            ("prepare_validation", "nll_validation.corpus.jsonl", "validation_tokens"),
        ):
            _stage(
                state,
                output,
                stage,
                command=[
                    sys.executable,
                    "-m",
                    TRAINING_MODULE,
                    "prepare",
                    "--input",
                    str(data / source),
                    "--output",
                    str(output / directory),
                    "--sequence-length",
                    str(length),
                ],
            )
        before_training = _stage(
            state,
            output,
            "verify_before_training",
            action=lambda: verify_complete_model(asset_root, spec_path, trusted_path),
        )
        if (
            before_training != verified
            or _sha256(config_path) != state["config_sha256"]
        ):
            raise ValueError(
                "model/config changed after preparation; GPU training refused"
            )
        _stage(
            state,
            output,
            "train",
            command=[
                sys.executable,
                "-m",
                TRAINING_MODULE,
                "train",
                "--model",
                verified["model_dir"],
                "--train",
                str(output / "train_tokens/tokens.jsonl"),
                "--validation",
                str(output / "validation_tokens/tokens.jsonl"),
                "--output",
                str(output / "training"),
                "--config",
                str(config_path),
                "--device",
                args.device,
            ],
        )
        result = _stage(
            state,
            output,
            "verify_adapter",
            action=lambda: _completed_adapter(output, verified, expected_steps),
        )
        state["status"] = "complete"
        state["result"] = result
        _write_state(output / "result.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        print(
            f"Pretraining stopped: {error}. Inspect {output / 'status.json'}",
            file=sys.stderr,
        )
        return 1
    finally:
        state["finished_at"] = time.time()
        _write_state(output / "status.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
