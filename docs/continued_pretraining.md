# Continued CoPE and attention-LoRA pretraining

This pipeline trains on real tokenized documents using causal next-token
cross-entropy. It loads a complete local Hugging Face safetensors backbone,
freezes every backbone parameter, and learns the existing CacheSlide attention
adapters: CoPE embeddings, a shared low-rank A matrix for concatenated Q/K/V,
and a separate attention-output adapter. Adapter scaling is one. These choices
retain the native adapter contract; they are not an assertion that the authors
used these precise LoRA settings.

The paper describes continued pretraining but does **not** disclose its exact
corpus, train split, rank, learning rate, optimizer, schedule, context length or
seed. Every setting in `configs/pretraining_engineering_pilot.json` is an
engineering choice. Its 20 updates are a pipeline pilot, not sufficient evidence
of quality recovery or reproduction of the paper's gains.

## Supported checkpoints and execution

- Local, complete, unquantized, bias-free Llama and full-attention Mistral
  safetensors checkpoints are supported.
- Mistral-7B-Instruct-v0.2 is an explicitly labeled engineering candidate, not
  a confirmed identity for the paper's unspecified Mistral-7B checkpoint.
  Mistral-v0.1's sliding-window configuration is rejected, not silently changed.
- MPT-30B is not supported by this trainer. Its attention/backbone architecture
  requires a separate implementation before any MPT result may be reported.
- Training uses one explicitly selected CPU or CUDA device. It is not a
  distributed trainer. BF16 CUDA matmuls retain FP32 trainable master parameters
  and FP32 AdamW moments; CoPE gates, cumulative sums and softmax avoid BF16
  autocast. FP16 training is not enabled.
- Layer and loss-chunk gradient checkpointing limit saved activations. CoPE is
  still a quadratic, unfused PyTorch reference, not a fast attention kernel.

Use the repository's isolated environment with PyTorch and safetensors. The
optional raw-text preparation command additionally requires Transformers and
its local tokenizer dependencies. Pretokenized corpora do not require
Transformers, a network connection, model remote code, or a native engine.
Do not modify a shared serving environment to install training dependencies.

## One-key verified pretraining

After installing the isolated training/data-preparation dependencies, run this
from the repository root with an explicitly allocated GPU. The command works on
both branches and selects that branch's training module:

```bash
python scripts/run_paper_pretraining.py \
  --asset-root /workspace/CacheSlide-paper-assets \
  --output runs/hotpotqa-continued-pretraining \
  --device cuda:0
```

There is no implicit download. Add `--download` only to explicitly run the pinned
asset downloader first. Otherwise an incomplete model or dataset stops the run.
The launcher verifies owner/spec identity and compares the entire local model
file list/size/hash records against the trusted repository-provided
`configs/paper_model_integrity.json`, sourced independently from the pinned
official Hugging Face revision. Changing a local payload together with its
download receipt cannot satisfy this anchor. The launcher then checks every
model file and every safetensors index shard **before data
preparation and again before the GPU training stage**. A config/tokenizer-only
snapshot cannot reach training. The data preparer independently verifies its
official dataset/tokenizer files.

Defaults are 128 training cases, 16 held-out NLL cases, 8 calibration cases,
16 untouched generation-evaluation cases, and an 8,192-token prompt capacity.
The counts, prompt capacity and data-selection seed are explicit CLI options;
overlength cases fail rather than being truncated or replaced. The default
`--config configs/pretraining_engineering_pilot.json` makes 512-target windows
and performs only 20 optimizer updates. These remain engineering pilot settings,
not a paper training recipe or a quality guarantee.

The launcher runs data preparation, train/NLL windowing and continued pretraining
as separate subprocesses. Every stage has stdout/stderr logs under `logs/`, and
`status.json` records running/completed/failed stages and subprocess exit codes.
On success, `result.json` reports the verified combined artifact at
`training/adapter/` and its next-step `--adapter` mount instruction. Training
metrics and held-out NLL remain under `training/`; success does not claim that
model quality or paper speedups have been achieved. Before success, the exact
native `AdapterBundle` loader independently reloads the result on CPU, checking
tensor names/shapes/finiteness and backbone binding. Artifact steps must match
both the requested configuration and the training report; malformed tensor
files or shorter runs labeled complete cannot report success.

This launcher never installs dependencies, manages GPU holders, starts a native
engine or runs a benchmark. Pause only the allocated GPU's task-owned idle
holder before launching, and restore it on success, failure or interruption
using the existing external keeper workflow. Output directories must be new
and cannot overlap the asset root. Interrupted training checkpoints can be
resumed with the lower-level command documented below into a new run directory.

## Real HotpotQA data to a native-compatible adapter

Run from the repository root after the pinned assets have been verified. The
asset root below is an example location; it must contain the layout produced by
`scripts/download_paper_assets.py`. Preparation and run outputs must be new
directories. Training data and held-out NLL documents are disjoint; the separate
generation evaluation inputs never include gold answers.

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_paper_data.py \
  --asset-root /workspace/CacheSlide-paper-assets \
  --output runs/hotpotqa-pilot-data \
  --training-count 64 --nll-validation-count 16 \
  --calibration-count 8 --evaluation-count 16 \
  --max-prompt-tokens 4096 --seed 9

python -m cacheslide_vllm.training_pipeline prepare \
  --input runs/hotpotqa-pilot-data/train.corpus.jsonl \
  --output runs/hotpotqa-pilot-train --sequence-length 512

python -m cacheslide_vllm.training_pipeline prepare \
  --input runs/hotpotqa-pilot-data/nll_validation.corpus.jsonl \
  --output runs/hotpotqa-pilot-validation --sequence-length 512

python -m cacheslide_vllm.training_pipeline train \
  --model /workspace/CacheSlide-paper-assets/models/Mistral-7B-Instruct-v0.2 \
  --train runs/hotpotqa-pilot-train/tokens.jsonl \
  --validation runs/hotpotqa-pilot-validation/tokens.jsonl \
  --config configs/pretraining_engineering_pilot.json \
  --device cuda:0 --output runs/hotpotqa-pilot-training
```

Check the downloaded model path in the asset manifest before running; the
`--model` argument must name the directory that actually contains `config.json`
and the complete safetensors index/shards. If the asset downloader uses a
different root layout, substitute that exact verified directory.

Windows retain all next-token transitions without cross-document packing:
adjacent windows share one token. Each window supervises at most 512 targets in
this pilot, and each optimizer update accumulates two windows. Document
identities survive windowing so windows from the same document cannot silently
appear in both the train and validation corpora. Training uses all document
tokens, including context, rather than answer-only masking. Validation NLL can
therefore be dominated by context; generation-task quality is a separate test.

For a different local text corpus, each source JSONL row is `{"text": "..."}`.
Pass `--tokenizer /path/to/local/model` to the `prepare` command. Alternatively,
each row can be `{"token_ids": [1, 2, 3]}` with no tokenizer. Make document-level
splits before preparation. User-supplied split quality remains the caller's
responsibility; exact document and token-sequence overlap is rejected.

## Checkpoints, reports and resume

A successful run produces:

- `adapter/`: the existing SHA-bound native CacheSlide adapter artifact; use
  this directory as the native runtime's `--adapter` value.
- `run.json`: the full training configuration, frozen-backbone SHA manifest,
  corpus hashes, precision, adapter semantics and explicit non-paper claims.
- `metrics.jsonl`: every optimizer update, target-weighted training loss,
  gradient norm, learning rate, token count and scheduled held-out validation.
- `checkpoints/step-00000010/`: immutable safetensors/JSON checkpoints with
  adapter weights, AdamW moments, RNG state, data cursor and schedule identity.
  No optimizer checkpoint is loaded through pickle.
- `report.json`: final/partial status and held-out token-weighted NLL/perplexity.

Resume into a new output directory with the identical configuration, corpus and
backbone. The scheduler and deterministic epoch ordering continue from the
saved optimizer step; changed inputs/configurations fail closed.

```bash
python -m cacheslide_vllm.training_pipeline train \
  --model /workspace/CacheSlide-paper-assets/models/Mistral-7B-Instruct-v0.2 \
  --train runs/hotpotqa-pilot-train/tokens.jsonl \
  --validation runs/hotpotqa-pilot-validation/tokens.jsonl \
  --config configs/pretraining_engineering_pilot.json \
  --device cuda:0 --output runs/hotpotqa-pilot-resumed \
  --resume runs/hotpotqa-pilot-training/checkpoints/step-00000010
```

The run-start validation baseline is the current CoPE/LoRA model, **not** the
original RoPE checkpoint. NLL/perplexity is **not** HotpotQA F1, MSC quality,
SWE-bench resolution, TTFT, QPS or memory saved. `paper_results_reproduced` remains
false in training reports: exporting a valid adapter is not a paper result.
Train longer and select hyperparameters using a dedicated validation split,
then run the paired native-engine evaluation on an untouched evaluation split.
Original-backbone quality, CacheSlide quality, true reuse receipts and measured
latency must all be retained independently.

## Mounting the trained artifact: not a plain PEFT LoRA

Pass the complete `adapter/` directory, containing `manifest.json` and
`adapter.safetensors`, to CacheSlide's `--adapter` option. Do not pass the step
checkpoint directory or just the safetensors file. The production vLLM
`CacheSlideModel` constructor verifies `AdapterBundle` against the backbone
before native allocation, loads each layer on the native parameter device/dtype,
and binds both LoRA deltas and CoPE to the attention wrapper.

For example, native generation mounts a verified adapter as follows. This
mounting check runs the adapted model; it does not claim cache-reuse acceleration
without separately calibrated profiles and verified reuse receipts.

```bash
python -m cacheslide_vllm.cli generate \
  --model /workspace/CacheSlide-paper-assets/models/Mistral-7B-Instruct-v0.2 \
  --adapter runs/hotpotqa-pilot-training/adapter \
  --input runs/hotpotqa-pilot-data/evaluation.inputs.jsonl \
  --cache-root runs/hotpotqa-native-cache \
  --output runs/hotpotqa-native-generation \
  --max-tokens 128 --dtype bfloat16 --run
```

For a paired reuse benchmark, use `bench` with `--profiles` naming profiles that
bind this exact adapter and the prepared layouts. Training does not itself
create CCPE profiles. Wrong-backbone adapters, incomplete artifacts and
mismatched profiles are rejected. `--adapter` does not bypass source-attestation
or supported-engine gates.

This is a **custom combined CoPE + attention-LoRA artifact**, not a standard PEFT
adapter. The shared QKV factors could be split algebraically into ordinary Q/K/V
LoRA matrices (unit scaling corresponds to `lora_alpha = rank`), but that would
not export CoPE's learned embeddings, contextual gates or the removal of native
RoPE. Loading only those linear factors with a generic `--lora-modules` flag
would run a different model and is not supported. No incomplete PEFT export is
provided or represented as equivalent to this trained model.

## Verification

```bash
PYTHONPATH=src python -m pytest -q tests/test_training.py tests/test_training_pipeline.py
```

The tests verify causal-loss/gradient agreement, frozen backbone weights,
token-weighted validation, no split overlap, actual optimizer updates and
bitwise-equivalent interrupted/resumed CPU adapters. The optional
`test_cuda_bfloat16_checkpointed_training` runs a tiny BF16 mechanics check only
when CUDA is available; it is not a model-quality or speed experiment. Allocate
its visible GPU explicitly and pause only this task's idle holder before a GPU
run, restoring the holder after the run or on failure.

The CPU integration test
`test_trained_artifact_mounts_through_actual_native_adapter_path` additionally
trains nonzero LoRA/CoPE parameters, saves and reloads them through the actual
vLLM CacheSlide model constructor and `CacheSlideRuntime`, and checks logit
agreement with the trained reference. Disabling the mounted LoRA or CoPE changes
outputs; a changed backbone is rejected before native allocation. Only
unavailable engine-native allocation/dispatch shells are replaced by tiny frozen
checkpoint projections in this test. It is **not** a native engine launch,
GPU/kernel validation, real Mistral training result or KV-reuse benchmark.
