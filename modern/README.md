# CacheSlide for the pinned vLLM V1 runtime

This directory implements the CacheSlide paper's contextual positions, weighted cache adaptation, and SLIDE slot indirection as a separate package. It includes actual CoPE/attention-adapter training, contextual profile calibration, a CPU reference model, and an opt-in native vLLM integration. The original repository's vLLM 0.8.5 code remains separate and unchanged.

The supported engine is **stable vLLM 0.29.0**, with source fingerprints tied to commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. This is not support for arbitrary `main` revisions. The observed server development build, `0.29.1rc1.dev5` with an `e52…` commit prefix, intentionally fails this compatibility check.

CPU numerical, training, storage, request-lifecycle, and integration-contract tests are provided. **Native GPU execution and large-model speedups have not been verified.** No author-provided trained CoPE weights or CCPE profiles are included. Newly initialized CoPE embeddings cannot enable native inference: a verified trained adapter is required.

## Supported scope

- Local, unquantized Llama or full-attention Mistral safetensors checkpoints with SwiGLU and bias-free attention/MLP projections.
- Explicit `CacheSlideLlamaForCausalLM` architecture and `CacheSlideWorker`; the plugin's registration alone does not change other models.
- One request at a time, TP/PP/DP and context parallelism all equal to one, eager V1 execution, and the audited `FLASH_ATTN` native KV layout.
- No automatic prefix caching, chunked prefill, asynchronous scheduling, CUDA graphs, compilation, sliding windows, speculative decoding, multimodal inputs, external LoRA requests, or prompt logprobs.
- Float16 or bfloat16 native inference. The portable CPU training/reference path uses floating-point Torch weights without a Transformers dependency.

The native model retains vLLM projections, normalization, MLPs, checkpoint loading, sampler, scheduler, and KV block ownership. Contextual attention currently gathers logical K/V through the slot map and runs query-chunked Torch attention. There is no fused FlashAttention CoPE kernel in this implementation.

## Install separately from the legacy tree

Use Python 3.12 or newer and a separate environment. Replace the example paths with local paths:

```bash
python3.12 -m venv /path/to/cacheslide-env
source /path/to/cacheslide-env/bin/activate
python -m pip install "/path/to/CacheSlide/modern[test]"
mkdir -p /path/to/cacheslide-work
cd /path/to/cacheslide-work
```

Run application commands from outside the legacy `CacheSlide/` root, so its `vllm/` package cannot shadow the installed engine. Installing `modern` does not require installing the legacy root package.

On the intended native GPU environment, install the pinned engine extra and inspect its source contract before creating an engine:

```bash
python -m pip install "/path/to/CacheSlide/modern[engine,test]"
cacheslide check-engine
```

An already downloaded source tree can also be checked without starting a GPU engine:

```bash
cacheslide check-engine --source-root /path/to/vllm-source --version 0.29.0
```

Version or source-hash differences fail closed. Do not bypass this check by relabeling a development build.

## Train the contextual attention adapter

Training input is JSONL with pretokenized `prompt_token_ids`. Tokenization must use the exact backbone tokenizer; the integers below only illustrate the schema:

```jsonl
{"id":"train-a","prompt_token_ids":[1,41,42,7,81,82,9,2]}
{"id":"train-b","prompt_token_ids":[1,41,42,17,18,81,82,19,2]}
```

```bash
cacheslide train \
  --model /models/llama \
  --input train.jsonl \
  --output artifacts/adapter \
  --steps 100 --lr 0.001 --rank 8 --max-positions 256 \
  --device cpu
```

This runs causal next-token cross-entropy and real optimizer updates. The backbone stays frozen; learned CoPE embeddings and low-rank QKV/output updates are trainable. `training_tokens` counts supervised next-token targets. The output records the completed training steps, loss endpoints, adapter checksum, model geometry, and exact local backbone-file hashes. Zero-step exports are rejected.

CPU is useful for small correctness runs. Choose hardware, data, and training duration appropriate to the model, and evaluate the trained model separately. The example step count is not a claim of recovered model quality. `--device cuda:0` selects GPU training when explicitly run on an appropriate environment.

## Calibrate real contextual profiles

Each request needs exact chunk boundaries over token IDs. Chunks must partition the prompt without gaps or overlaps. `reuse` chunks retain their relative order, IDs, and exact token content; `recompute` chunks can change in content and length. The last prompt token is always mandatory for sampling, even if its chunk is marked reusable.

For example, save these lines as `calibration.jsonl`:

```jsonl
{"id":"calibration-short","prompt_token_ids":[1,41,42,7,81,82,9,2],"cacheslide":{"version":1,"operation":"calibrate","namespace":"demo","task_id":"qa","chunks":[{"id":"document-a","role":"reuse","start":0,"end":3},{"id":"dynamic","role":"recompute","start":3,"end":4},{"id":"document-b","role":"reuse","start":4,"end":6},{"id":"query","role":"recompute","start":6,"end":8}]}}
{"id":"calibration-long","prompt_token_ids":[1,41,42,17,18,81,82,19,2],"cacheslide":{"version":1,"operation":"calibrate","namespace":"demo","task_id":"qa","chunks":[{"id":"document-a","role":"reuse","start":0,"end":3},{"id":"dynamic","role":"recompute","start":3,"end":5},{"id":"document-b","role":"reuse","start":5,"end":7},{"id":"query","role":"recompute","start":7,"end":9}]}}
```

```bash
cacheslide calibrate \
  --model /models/llama --adapter artifacts/adapter \
  --input calibration.jsonl --output artifacts/profiles \
  --profile-version qa-v1 --max-elements 1000000 --device cpu

cacheslide inspect \
  --model /models/llama --adapter artifacts/adapter \
  --profiles artifacts/profiles
```

Calibration runs the trained, frozen model on full contexts. It computes CoPE gates including dynamic tokens, then projects fixed-query/fixed-key positions into compact fixed-token ordinals. A deterministic joint-pattern histogram selects an actual observed encoding. Profiles are tied to the adapter, namespace/task, ordered fixed chunks, layer, and exact fixed-token lengths. Dynamic tokens can move the same fixed chunks to different absolute prompt positions.

Calibration source traces and canonical profiles have quadratic query/key dimensions. Aggregate element budgets are checked before source-trace allocation. This explicit, bounded profile representation is **not a demonstrated scalable long-context CCPE implementation**. Saved artifacts contain canonical positions and fixed-chunk identities, not raw Q/K source traces or input token sequences.

## Populate, reuse, and measure

Save these lines as `requests.jsonl`. The first populates fixed-chunk snapshots; the second reuses the same ordered fixed context with a longer dynamic span:

```jsonl
{"id":"seed","prompt_token_ids":[1,41,42,7,81,82,9,2],"cacheslide":{"version":1,"operation":"populate","namespace":"demo","task_id":"qa","chunks":[{"id":"document-a","role":"reuse","start":0,"end":3},{"id":"dynamic","role":"recompute","start":3,"end":4},{"id":"document-b","role":"reuse","start":4,"end":6},{"id":"query","role":"recompute","start":6,"end":8}]}}
{"id":"query-long","prompt_token_ids":[1,41,42,17,18,81,82,19,2],"cacheslide":{"version":1,"operation":"reuse","namespace":"demo","task_id":"qa","chunks":[{"id":"document-a","role":"reuse","start":0,"end":3},{"id":"dynamic","role":"recompute","start":3,"end":5},{"id":"document-b","role":"reuse","start":5,"end":7},{"id":"query","role":"recompute","start":7,"end":9}]}}
```

First review the native configuration without constructing an engine:

```bash
cacheslide generate \
  --model /models/llama --adapter artifacts/adapter --profiles artifacts/profiles \
  --input requests.jsonl --cache-root cache/generation --output results/generation \
  --max-model-len 4096 --max-tokens 1
```

Add `--run` to construct and execute the native GPU engine. It sets `VLLM_USE_V1=1` and `VLLM_USE_V2_MODEL_RUNNER=0`, explicitly selects the CacheSlide architecture/worker, and disables incompatible scheduler/cache modes. The native engine is never launched by the default configuration-review invocation.

For a matched offline benchmark:

```bash
cacheslide bench \
  --model /models/llama --adapter artifacts/adapter --profiles artifacts/profiles \
  --input requests.jsonl --cache-root cache/benchmark --output results/benchmark \
  --max-model-len 4096 --max-tokens 1 --warmup 1 --repeats 3 --run
```

The benchmark requires `--profiles` when `--run` is supplied. It populates one seed per fixed layout, then compares full `recompute` and `reuse` operations on the same measured queries using one warm engine. `--seed-input seed.jsonl` supplies separate populate seeds. Seed population and warmup calls are saved but excluded from measured means. Each case retains its input length, output length, generated token IDs, and latency. The reported ratio is `mean(recompute latency) / mean(reuse latency)`, not the mean of per-request ratios.

After each timed call, a worker metrics receipt is collected outside the timed region and matched to the request ID, operation, and prompt length. `reuse_validation_passed` requires every measured reuse to report an actual cache hit without fallback. Missing/stale receipts, fallback, invalid means, or mismatched output lengths suppress the latency ratio (`null`); raw results remain available. The command drains its cache store through the worker before exiting.

With one output token, timing is **offline prefill-plus-one-token latency**, including the blocking generation call's overhead. It is not streaming TTFT. With more output tokens it is offline generation latency. Output checks report exact generated-token agreement; optional `--token-f1` is token-ID multiset overlap, ignores order, and is not dataset answer F1 or semantic accuracy. The baseline uses the same trained CoPE/LoRA model and profiles, not an untouched RoPE checkpoint.

Results are written to a new directory as `raw_outputs.jsonl` and `summary.json`; existing output directories are rejected. The standalone `benchmarks/benchmark_reuse.py` script forwards to the same command. `--backend reference` currently fails explicitly: a CPU reuse benchmark command is not connected.

## Algorithm choices and current limits

WCA defaults to a correction budget of `ceil(0.26 * reusable_tokens)`, capped by positive-error candidates. Dynamic rows and the final prompt row are always computed. The native runtime fully computes two shallow layers by default and initializes selection from zero-based layer `1`: first-layer pre-attention K is token-local under CoPE and can otherwise give zero reused-token deviation.

The paper's literal CKSim condition, mean head cosine `< 0.12` every four layers, is the default despite its tension with the prose's convergence description. `distance_lt` is a separate experiment. The raw adaptation ratio can exceed one; previous-layer weighting is the default, with same-layer weighting exposed separately. See [the design notes](docs/design.md) for the equations and conventions.

SLIDE uses an actual native block-table-backed slot arena plus bounded device sidecars for selected tokens. Decode can reuse safely vacated physical slots, but all later attention must continue through the logical mapping. This **does not free blocks or shrink the native KV pool reserved by vLLM**. Host caches include K/V and per-layer hidden/residual snapshots needed when WCA promotes previously skipped tokens. Storage counters measure managed payload/file bytes, not SSD hardware write amplification or total process RSS.

Profiles and complete cache snapshots are required for reuse. A profile/cache miss or a recoverable reuse failure falls back to full prompt computation. The benchmark records cache-hit/fallback receipts so a `reuse` operation alone is never treated as proof of acceleration.

## Verify locally

From the `modern/` directory, using the isolated environment:

```bash
python -m pytest -q
python -m ruff check src tests benchmarks
```

These checks exercise portable numerical paths, real small-model training/calibration, state restoration, request cleanup, storage ordering, and pinned engine contracts. They do not substitute for native GPU correctness tests, representative task-quality evaluation, or large-model performance measurements.

Source map: [training](src/cacheslide_vllm/training.py), [reference model](src/cacheslide_vllm/reference.py), [CoPE/CCPE](src/cacheslide_vllm/position.py), [profile artifacts](src/cacheslide_vllm/profiles.py), [WCA](src/cacheslide_vllm/wca.py), [runtime](src/cacheslide_vllm/runtime.py), [native model](src/cacheslide_vllm/model.py), [paged KV arena](src/cacheslide_vllm/paged.py), [slot mapping](src/cacheslide_vllm/slide.py), [host storage](src/cacheslide_vllm/storage.py), and [CLI](src/cacheslide_vllm/cli.py).
