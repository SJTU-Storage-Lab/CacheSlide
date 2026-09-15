<p align="center">
  <img src="CacheSlide.png" alt="CacheSlide — contextual KV cache reuse" width="560">
</p>

<p align="center"><strong>SGlang-CacheSlide: contextual KV reuse with an audited native SGLang adapter.</strong></p>

This **SGlang-CacheSlide branch** adapts [CacheSlide (FAST '26)](https://www.usenix.org/conference/fast26/presentation/liu-yang) to **SGLang 0.5.19**. It shares CCPE, WCA and SLIDE algorithms through `cacheslide_core`, with separate SGLang and vLLM adapters. No engine source is vendored. The audited vLLM main baseline at `3f220b8` is unchanged by this branch.

SGLang is pinned to commit `0bcd822377da7b5718e674eaf9c870d349424dd1`; all **30 audited source fingerprints** match the official 0.5.19 wheel. Other versions or modified audited files are rejected. The SGLang adapter does not import vLLM.

**Experimental implementation, not a reproduced speedup.** CPU training, numerical, lifecycle and native-layout checks pass. A native SGLang GPU engine run, representative model quality and the paper's TTFT/SSD gains are not yet established. See [SGLang design](docs/sglang_design.md) and [validation](docs/sglang_validation.md).

## Real model training and paper datasets

The [pinned open assets](docs/open_assets.md) include an explicitly identified
Mistral-7B engineering checkpoint, official HotpotQA, MSC and SWE-bench sources.
They are acquired with official checksum verification; model weights and raw
datasets are not silently embedded in Git. See the [HotpotQA preparation
protocol](docs/hotpot_data_preparation.md) for separate training, NLL validation,
calibration and held-out generation inputs.

[Continued pretraining](docs/continued_pretraining.md) loads the real frozen
safetensors backbone and optimizes CoPE plus attention LoRA using causal
next-token loss. It supports BF16, gradient checkpointing, resumable optimizer
state and held-out NLL. Mount the resulting combined artifact with `--adapter`;
a generic runtime LoRA alone cannot enable CoPE. The checked-in 20-step pilot
is a pipeline check, **not** a quality-qualified model or reproduced speedup.
[Answer reporting](docs/reproduction_reporting.md) preserves actual generations
and supports official HotpotQA answer F1/EM. Training loss, token agreement and
blocking generation latency are not relabeled as task accuracy or TTFT.

With the selected environment and its `paper` extra installed, a real-data pilot
can be launched in one command:

```bash
python scripts/run_paper_pretraining.py \
  --asset-root /workspace/Models/CacheSlide-paper-assets \
  --output ./runs/mistral-hotpot-pilot --device cuda:0
```

This verifies every model shard before GPU work, prepares disjoint data, trains,
and validates the saved adapter. Add `--download` to explicitly acquire missing
assets first. It does not install dependencies, manage GPU reservations, or
claim native-engine results. Stage logs and failures are retained.

## One-command workflow

Use Python 3.12 or newer. The launcher preserves caller-relative paths and never downloads model weights.

### CPU correctness

```bash
./run_cacheslide_benchmark.sh --smoke --install --output ./results/sglang-cpu
```

This creates a new environment and performs real adapter training, CCPE calibration, population and multi-token decoding using a tiny random checkpoint and the actual split-K/V pool adapter on CPU. It imports neither native SGLang nor vLLM. Strict defaults verify same-context reuse and safe plain-CoPE fallback for an incompatible shifted context; fallback is **not** a cache hit. The fixture does not establish language quality or GPU speed.

With dependencies installed, omit `--install`; select an interpreter through `CACHESLIDE_PYTHON=/path/to/env/bin/python`. To exercise the explicit approximate engineering variant:

```bash
./run_cacheslide_benchmark.sh --smoke \
  --ccpe-position-policy mixed_bias_override --calibration-layer 1 \
  --max-tokens 4 --output ./results/sglang-cpu-variant
```

The variant accepts mixed positional biases without contextual-path validation and initializes WCA after two shallow layers. It is not the literal strict default or proof of the paper's accuracy.

### Native SGLang experiment

```bash
CUDA_VISIBLE_DEVICES=0 ./run_cacheslide_benchmark.sh \
  --install --model /models/llama \
  --train-input /data/train.jsonl \
  --calibration-input /data/calibration.jsonl \
  --input /data/evaluation.jsonl --output ./results/sglang-native \
  --train-device cuda:0 --calibration-device cuda:0 \
  --steps 100 --warmup 3 --repeats 10 --max-tokens 50 --run
```

This is an experimental entry point, not a claim that native GPU validation is complete. It checks the pinned installation, trains, calibrates, populates, then measures paired recompute/reuse calls in **one warm engine**. Training and calibration run in separate processes. `--install` creates only a new isolated environment, installs the pinned SGLang extra, then executes the same request; it never modifies an existing environment. Omit both `--run` and `--install` for a read-only plan. With no arguments, the launcher prints help.

Use `--adapter /artifacts/adapter` to skip training; add `--profiles /artifacts/profiles` to skip calibration. Profiles require their verified adapter. `--seed-input` supplies separate population cases. Otherwise training/calibration default to `--input`, which is convenient but **not held-out quality evaluation**.

`--output` is required and must be new. The default cache is `<output>/cache`; `--cache-root /new/ssd-cache` selects another new directory. Results include `workflow.json`, `logs/`, `raw_outputs.jsonl`, `summary.json`, and native request receipts. Stage failure stops execution without switching backends. Invalid benchmark receipts/fallbacks or failed output validation suppress the ratio, record `validation_failed`, and return exit code 2.

The launcher does not pause GPU holders or manage allocations. Assign an available GPU; never stop unrelated jobs. Any separately authorized temporary GPU reservation must be restored after experiments, including failure.

## Input and strict defaults

Inputs are JSONL using exact tokenizer token IDs, partitioned completely into ordered chunks:

```json
{"id":"query-1","prompt_token_ids":[1,41,42,7,81,82,9,2],"cacheslide":{"version":1,"operation":"reuse","namespace":"demo","task_id":"qa","chunks":[{"id":"document-a","role":"reuse","start":0,"end":3},{"id":"dynamic","role":"recompute","start":3,"end":4},{"id":"document-b","role":"reuse","start":4,"end":6},{"id":"query","role":"recompute","start":6,"end":8}]}}
```

These IDs illustrate the schema, not model text. Training rows need only `id` and `prompt_token_ids`. Calibration/evaluation require chunk metadata. Cache identity binds every ordered chunk ID/role and fixed content/length; dynamic content and length may vary within that template. Identity matching does not guarantee a valid strict positional path.

Defaults are `strict_contextual` and `calibration_layer=0` (Algorithm 2's first layer). Invalid canonical/current paths clear every layer profile and restart the full prefill with plain trained CoPE, recording a fallback rather than a hit. First-layer token-local K can have zero deviation, legitimately selecting no correction tokens. WCA retains the literal `cosine < 0.12` gate every four layers and correction fraction 0.26.

## Architecture and supported scope

```text
src/cacheslide_core/     Shared contracts, CoPE/CCPE, WCA, SLIDE, storage,
                        reference training and selective runtime
src/cacheslide_sglang/   Pinned plugin/hooks, model/backend, split K/V pool,
                        request receipts, CLI and workflow
src/cacheslide_vllm/     Retained independent vLLM adapter and commands
```

The initial native scope is one GPU/rank, one request, eager execution, local unquantized bias-free Llama or full-attention Mistral, and trained adapters. Radix/prefix caching, HiCache, overlap scheduling, chunked prefill, CUDA graphs, compilation, parallelism, speculative decoding, sliding-window attention, multimodal inputs and external runtime LoRA are unsupported. Source/configuration checks and parent/worker hook attestations fail closed.

SLIDE supports load-ready in-place writes, pending-load relocation and safe physical token-hole reuse through per-layer mappings. It does **not** release SGLang's reserved native pool. Physical selected-token page ordering is advisory; mutable GPU-page SSD eviction/coalesced writeback is not integrated. Dense gathers, sidecars, snapshots and quadratic profile budgets remain costs; no fused/scalable contextual-attention kernel is claimed.

## Measurement and retained vLLM baseline

Both measured paths use the same trained CoPE/adapter configuration, not an untouched RoPE baseline. Setup, population and warmups are excluded; only verified request-bound hits without fallback can produce `mean(recompute latency) / mean(reuse latency)`. The measured interval is blocking whole generation, not streaming TTFT or concurrent QPS. Token-ID multiset F1 is consistency, not dataset answer F1. Host file-byte counters are not SSD hardware write amplification.

The explicit vLLM entry points remain `run_cacheslide_vllm_benchmark.sh`, `cacheslide-vllm` and `cacheslide-vllm-benchmark`. The existing [design](docs/design.md), [validation](docs/validation.md) and [paper audit](docs/paper_conformance.md) are **historical vLLM baseline documents**; their engine details and counts do not certify SGLang. Shared algorithm interpretations remain applicable where stated.

## Development

```bash
python3.12 -m venv .sglang-venv
.sglang-venv/bin/python -m pip install -c constraints/cpu-tests.txt '.[test]'
CACHESLIDE_SGLANG_SOURCE=/path/to/sglang-v0.5.19 \
CACHESLIDE_VLLM_SOURCE=/path/to/vllm-v0.29.0 \
  .sglang-venv/bin/python -m pytest -q
.sglang-venv/bin/python -m ruff check src tests benchmarks
```

Source-dependent tests explicitly skip without their matching source checkouts. Real CUDA pool tests require a separate opt-in and assigned GPU; they are not native engine tests. Commands: `cacheslide train`, `calibrate`, `check-engine`, `bench`; `cacheslide-offline` provides shared offline preparation.

## Citation

```bibtex
@inproceedings{liu2026CacheSlide,
  title={CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving},
  author={Yang Liu and Yunfei Gu and Liqiang Zhang and Chentao Wu and Guangtao Xue and Jie Li and Minyi Guo and Junhao Hu and Jie Meng},
  booktitle={24th USENIX Conference on File and Storage Technologies (FAST 26)},
  year={2026}
}
```

Apache-2.0. Contact: [liuyang370@sjtu.edu.cn](mailto:liuyang370@sjtu.edu.cn).
