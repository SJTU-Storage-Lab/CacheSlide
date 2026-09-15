"""Real local-corpus CoPE/attention-LoRA continued pretraining and validation.

Run ``python -m cacheslide_core.training_pipeline --help``. The backbone is
loaded from a local Hugging Face safetensors checkpoint, not initialized from
synthetic weights. This is a single-device training reference, not a fused or
distributed training engine. All defaults below are engineering choices: the
CacheSlide paper does not disclose its corpus, rank, optimizer or schedule.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

from .artifacts import backbone_manifest, file_sha256, read_json, save_adapter
from .position import cope_attention
from .reference import ReferenceLlama
from .training_data import (
    TokenCorpus,
    assert_disjoint,
    load_local_tokenizer,
    prepare_corpus,
)


@dataclass(frozen=True)
class PretrainingConfig:
    """Explicit engineering settings, deliberately not a claimed paper preset."""

    max_steps: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    sequence_length: int = 2048
    rank: int = 8
    max_positions: int = 256
    query_chunk_size: int = 64
    loss_chunk_size: int = 128
    gradient_checkpointing: bool = True
    precision: str = "float32"
    seed: int = 42
    eval_every: int = 100
    eval_max_sequences: int = 0
    checkpoint_every: int = 100

    def validate(self) -> None:
        for name in (
            "max_steps",
            "gradient_accumulation_steps",
            "sequence_length",
            "rank",
            "max_positions",
            "query_chunk_size",
            "loss_chunk_size",
            "eval_every",
            "checkpoint_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "eval_max_sequences", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps >= self.max_steps or self.max_positions < 2:
            raise ValueError("warmup must precede the last step; max_positions >= 2")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if (
                type(value) not in {int, float}
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.weight_decay) not in {int, float}
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or type(self.min_lr_ratio) not in {int, float}
            or not math.isfinite(self.min_lr_ratio)
            or not 0 < self.min_lr_ratio <= 1
        ):
            raise ValueError("invalid weight decay or minimum learning-rate ratio")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be bool")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


def learning_rate_at(config: PretrainingConfig, step: int) -> float:
    """One-based optimizer step: linear warmup, then cosine decay."""
    if not 1 <= step <= config.max_steps:
        raise ValueError("learning-rate step outside configured schedule")
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    remaining = config.max_steps - config.warmup_steps
    progress = (step - config.warmup_steps - 1) / max(1, remaining - 1)
    ratio = (
        config.min_lr_ratio
        + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )
    return config.learning_rate * ratio


class TrainingForward:
    """Checkpoint residual layers and CE chunks, retaining exact causal targets.

    Query-chunked CoPE alone still retains each chunk's autograd graph. Layer
    checkpointing rematerializes that graph on backward and saves frozen-MLP
    activations. The attention remains quadratic, not FlashAttention speed.
    FP32 master adapters are optimized even with BF16 backbone matmuls; CoPE
    gates/cumulative sums/softmax explicitly run outside BF16 autocast.
    """

    def __init__(self, model: ReferenceLlama, config: PretrainingConfig):
        self.model, self.config = model, config
        self.device = model.embed_tokens.weight.device

        def attention(layer_index, positions, query, key, value, adapter):
            with torch.autocast(self.device.type, enabled=False):
                return cope_attention(
                    query,
                    key,
                    value,
                    adapter.cope,
                    positions,
                    query_chunk_size=config.query_chunk_size,
                )

        for layer in model.layers:
            layer.self_attn.attention_handler = attention

    def autocast(self):
        if self.config.precision == "bfloat16":
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def loss_sum(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
            raise ValueError("training requires a one-dimensional next-token sequence")
        positions = torch.arange(ids.numel() - 1, device=self.device)
        hidden, residual = self.model.embed_tokens(ids[:-1]), None
        rematerialize = self.config.gradient_checkpointing and self.model.training
        with self.autocast():
            for layer in self.model.layers:
                if rematerialize:
                    # Bind the current layer, not the loop variable at backward time.
                    def forward(h, r, current_layer=layer):
                        return current_layer(positions, h, r)

                    hidden, residual = checkpoint(
                        forward, hidden, residual, use_reentrant=False
                    )
                else:
                    hidden, residual = layer(positions, hidden, residual)
            hidden, _ = self.model.norm(hidden, residual)
            losses = []
            for start in range(0, hidden.shape[0], self.config.loss_chunk_size):
                end = start + self.config.loss_chunk_size

                def head_loss(rows, targets):
                    logits, _ = self.model.lm_head(rows)
                    logits = logits.float() * self.model.logit_scale
                    return F.cross_entropy(logits, targets, reduction="sum")

                args = (hidden[start:end], ids[1:][start:end])
                losses.append(
                    checkpoint(head_loss, *args, use_reentrant=False)
                    if rematerialize
                    else head_loss(*args)
                )
        return torch.stack(losses).sum()


def evaluate(
    forward: TrainingForward, corpus: TokenCorpus, *, max_sequences: int = 0
) -> dict:
    """Token-weighted held-out causal NLL/perplexity, never answer F1."""
    if type(max_sequences) is not int or max_sequences < 0:
        raise ValueError("max_sequences must be a nonnegative integer")
    was_training = forward.model.training
    forward.model.eval()
    count = min(len(corpus), max_sequences or len(corpus))
    total_nll, targets = 0.0, 0
    try:
        with torch.inference_mode():
            for index in range(count):
                ids = torch.tensor(corpus[index], device=forward.device)
                loss = float(forward.loss_sum(ids).cpu())
                if not math.isfinite(loss):
                    raise ValueError("held-out loss is nonfinite")
                total_nll += loss
                targets += ids.numel() - 1
    finally:
        forward.model.train(was_training)
    nll = total_nll / targets
    return {
        "nll": nll,
        "perplexity": math.exp(nll) if nll < 700 else None,
        "perplexity_overflow": nll >= 700,
        "supervised_tokens": targets,
        "sequences": count,
        "full_validation_split": count == len(corpus),
    }


def _sample_indices(
    corpus: TokenCorpus, cursor: int, count: int, seed: int
) -> list[int]:
    result, current_epoch, order = [], None, []
    for absolute in range(cursor, cursor + count):
        epoch, offset = divmod(absolute, len(corpus))
        if epoch != current_epoch:
            order = list(range(len(corpus)))
            random.Random(seed + epoch).shuffle(order)
            current_epoch = epoch
        result.append(order[offset])
    return result


def _save_checkpoint(
    output: Path, model, optimizer, identity: dict, progress: dict, device
) -> None:
    """New immutable directory, safetensors optimizer state, no pickle load path."""
    output.mkdir(parents=True, exist_ok=False)
    tensors = {
        "adapter." + name: value.detach().cpu().contiguous()
        for name, value in model.adapters.state_dict().items()
    }
    for name, parameter in model.adapters.named_parameters():
        state = optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("unexpected AdamW state; checkpoint not published")
        for key, value in state.items():
            tensors[f"optimizer.{name}.{key}"] = value.detach().cpu().contiguous()
    tensors["rng.cpu"] = torch.get_rng_state()
    if device.type == "cuda":
        tensors["rng.cuda"] = torch.cuda.get_rng_state(device)
    save_file(tensors, str(output / "state.safetensors"))
    (output / "state.json").write_text(
        json.dumps(
            {
                "format": "cacheslide-pretraining-checkpoint-v1",
                "identity": identity,
                "progress": progress,
                "state_sha256": file_sha256(output / "state.safetensors"),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _load_checkpoint(path: Path, model, optimizer, identity: dict, device) -> dict:
    metadata = read_json(path / "state.json")
    if (
        metadata.get("format") != "cacheslide-pretraining-checkpoint-v1"
        or metadata.get("identity") != identity
        or metadata.get("state_sha256") != file_sha256(path / "state.safetensors")
    ):
        raise ValueError("resume checkpoint identity/configuration/checksum mismatch")
    tensors = load_file(str(path / "state.safetensors"))
    expected = {"adapter." + name for name in model.adapters.state_dict()}
    expected |= {
        f"optimizer.{name}.{key}"
        for name, _ in model.adapters.named_parameters()
        for key in ("step", "exp_avg", "exp_avg_sq")
    }
    expected |= {"rng.cpu"} | ({"rng.cuda"} if device.type == "cuda" else set())
    if set(tensors) != expected:
        raise ValueError("resume checkpoint tensor layout mismatch")
    if any(
        not torch.isfinite(t).all() for t in tensors.values() if t.is_floating_point()
    ):
        raise ValueError("resume checkpoint contains nonfinite tensors")
    model.adapters.load_state_dict(
        {
            key.removeprefix("adapter."): value
            for key, value in tensors.items()
            if key.startswith("adapter.")
        },
        strict=True,
    )
    progress = metadata["progress"]
    step = progress.get("step")
    config = identity["config"]
    if (
        type(step) is not int
        or not 1 <= step <= config["max_steps"]
        or progress.get("cursor") != step * config["gradient_accumulation_steps"]
        or len(progress.get("losses", [])) != step
        or type(progress.get("training_tokens")) is not int
        or progress["training_tokens"] < 1
        or any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in progress.get("losses", [])
        )
    ):
        raise ValueError("resume checkpoint has invalid progress")
    for name, parameter in model.adapters.named_parameters():
        state = {}
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = tensors[f"optimizer.{name}.{key}"]
            if key == "step":
                if value.numel() != 1 or float(value) != step:
                    raise ValueError(
                        "optimizer step does not match checkpoint progress"
                    )
                state[key] = value.cpu()
            else:
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("optimizer moment shape/dtype mismatch")
                state[key] = value.to(device)
        optimizer.state[parameter] = state
    torch.set_rng_state(tensors["rng.cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(tensors["rng.cuda"], device)
    return progress


def run_pretraining(
    model_dir: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    output: str | Path,
    *,
    config: PretrainingConfig,
    device: str = "cpu",
    resume: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> dict:
    """Train attention LoRA+CoPE and export the existing native adapter format.

    Resume writes to a *new* run directory and checks the entire original
    schedule, backbone and corpus identities. ``stop_after_steps`` is an
    explicit partial-run/debug facility: it saves a resumable checkpoint but
    never exports a final trained adapter or claims experiment completion.
    """
    config.validate()
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this training pipeline supports CPU or one CUDA device")
    if config.precision == "bfloat16":
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("bfloat16 training requires a BF16-capable CUDA device")
    if stop_after_steps is not None and (
        type(stop_after_steps) is not int
        or not 1 <= stop_after_steps <= config.max_steps
    ):
        raise ValueError("stop_after_steps outside requested training schedule")
    output, model_dir = Path(output), Path(model_dir).resolve(strict=True)
    if output.exists():
        raise FileExistsError("training output must be a new directory")
    base_identity = backbone_manifest(model_dir)
    torch.manual_seed(config.seed)
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float32
    model = ReferenceLlama.from_checkpoint(
        model_dir,
        rank=config.rank,
        max_positions=config.max_positions,
        query_chunk_size=config.query_chunk_size,
        device=device,
        dtype=dtype,
    )
    # Keep FP32 master weights and AdamW moments when backbone uses BF16.
    model.adapters.float()
    train = TokenCorpus(
        train_path,
        max_sequence_length=config.sequence_length,
        vocab_size=model.config["vocab_size"],
    )
    validation = TokenCorpus(
        validation_path,
        max_sequence_length=config.sequence_length,
        vocab_size=model.config["vocab_size"],
    )
    assert_disjoint(train, validation)
    identity = {
        "config": asdict(config),
        "base_files": base_identity,
        "train_sha256": train.sha256,
        "validation_sha256": validation.sha256,
        "torch_version": str(torch.__version__),
        "device_type": device.type,
        "adapter_contract": "cacheslide-adapter-v1",
    }
    trainable = list(model.adapters.parameters())
    if {id(p) for p in model.parameters() if p.requires_grad} != {
        id(p) for p in trainable
    }:
        raise RuntimeError("only explicitly registered CoPE/attention LoRA may train")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    forward = TrainingForward(model, config)
    progress = {"step": 0, "cursor": 0, "training_tokens": 0, "losses": []}
    if resume is not None:
        progress = _load_checkpoint(
            Path(resume).resolve(strict=True), model, optimizer, identity, device
        )
    final_step = stop_after_steps or config.max_steps
    if final_step <= progress["step"]:
        raise ValueError("resume requires at least one further optimizer update")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(
        json.dumps(
            {
                "format": "cacheslide-continued-pretraining-v1",
                "identity": identity,
                "train": train.manifest(),
                "validation": validation.manifest(),
                "resume": str(Path(resume).resolve()) if resume else None,
                "paper_hyperparameter_claim": False,
                "lora_scaling": (
                    "unit scale; shared concatenated QKV A and output adapter"
                ),
                "backbone_frozen": True,
                "original_rope_quality_comparison": False,
                "attention": "CoPE, full causal; quadratic unfused training reference",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    validation_before = evaluate(
        forward, validation, max_sequences=config.eval_max_sequences
    )
    evaluations = [{"step": progress["step"], **validation_before}]
    model.train()
    started = time.monotonic()
    with (output / "metrics.jsonl").open("x", buffering=1) as log:
        for step in range(progress["step"] + 1, final_step + 1):
            indices = _sample_indices(
                train,
                progress["cursor"],
                config.gradient_accumulation_steps,
                config.seed,
            )
            targets = sum(train.lengths[index] for index in indices)
            lr = learning_rate_at(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for index in indices:
                ids = torch.tensor(train[index], dtype=torch.long, device=device)
                loss = forward.loss_sum(ids) / targets
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite continued-pretraining loss")
                loss.backward()
                total_loss += float(loss.detach().cpu())
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable, config.max_grad_norm, error_if_nonfinite=True
                )
            )
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in trainable):
                raise ValueError("nonfinite adapter after optimizer update")
            progress["step"] = step
            progress["cursor"] += len(indices)
            progress["training_tokens"] += targets
            progress["losses"].append(total_loss)
            record = {
                "step": step,
                "train_nll": total_loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "supervised_tokens": targets,
                "cumulative_training_tokens": progress["training_tokens"],
                "elapsed_seconds": time.monotonic() - started,
            }
            if step % config.eval_every == 0 or step == final_step:
                result = evaluate(
                    forward, validation, max_sequences=config.eval_max_sequences
                )
                evaluations.append({"step": step, **result})
                record["validation"] = result
            log.write(json.dumps(record, allow_nan=False) + "\n")
            if step % config.checkpoint_every == 0 or step == final_step:
                _save_checkpoint(
                    output / "checkpoints" / f"step-{step:08d}",
                    model,
                    optimizer,
                    identity,
                    progress,
                    device,
                )
    if backbone_manifest(model_dir) != base_identity:
        raise ValueError("local backbone changed during training; no final export")
    if (
        file_sha256(train.path) != train.sha256
        or file_sha256(validation.path) != validation.sha256
    ):
        raise ValueError("corpus changed during training; no final export")
    complete = progress["step"] == config.max_steps
    artifact = None
    if complete:
        artifact = save_adapter(
            output / "adapter",
            model_dir,
            list(model.adapters),
            training_steps=progress["step"],
            training_tokens=progress["training_tokens"],
            losses=progress["losses"],
        )
    report = {
        "status": "complete" if complete else "partial",
        "adapter": str(artifact) if artifact else None,
        "optimizer_steps": progress["step"],
        "training_tokens": progress["training_tokens"],
        "validation": evaluations,
        "validation_metric": "token-weighted causal NLL/perplexity; not answer F1",
        "baseline": "CoPE adapter at run start, not original RoPE model",
        "paper_results_reproduced": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="tokenize local text/token JSONL")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tokenizer", help="local HF snapshot; never downloads")
    prepare.add_argument("--sequence-length", type=int, default=2048)
    prepare.add_argument("--no-special-tokens", action="store_true")
    train = sub.add_parser("train", help="continued pretraining + held-out validation")
    train.add_argument("--model", required=True)
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config", required=True, help="JSON PretrainingConfig fields")
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume", help="a step checkpoint; output must remain new")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        tokenizer, provenance = (
            load_local_tokenizer(args.tokenizer) if args.tokenizer else (None, None)
        )
        result = prepare_corpus(
            args.input,
            args.output,
            sequence_length=args.sequence_length,
            tokenizer=tokenizer,
            tokenizer_provenance=provenance,
            add_special_tokens=not args.no_special_tokens,
        )
    else:
        result = run_pretraining(
            args.model,
            args.train,
            args.validation,
            args.output,
            config=PretrainingConfig(**read_json(Path(args.config))),
            device=args.device,
            resume=args.resume,
        )
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
