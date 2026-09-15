<p align="center">
  <img src="CacheSlide.png" alt="CacheSlide — contextual KV cache reuse" width="560">
</p>

<p align="center"><strong>Contextual, cross-position KV cache reuse for modern vLLM.</strong></p>

This repository implements an audited, experimental adaptation of [CacheSlide (FAST '26)](https://www.usenix.org/conference/fast26/presentation/liu-yang): Chunked Contextual Position Encoding (CCPE), Weighted Correction Attention (WCA), and SLIDE-style KV slot management. The paper's complete mutable-page SSD path and serving-performance results are not reproduced.

CacheSlide is now a standalone package. It **does not vendor the old vLLM 0.8.5 source tree**. The old implementation remains in Git history, including commit `2d4a44a`. vLLM is an external, pinned dependency; importing CacheSlide never globally patches an existing model.

## Engine and validation status

The baseline is [vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0), the latest stable release checked on September 15, 2026. Model Runner **V2 is the default**; `--model-runner v1` selects the compatibility path. Source fingerprints bind the integration to upstream commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. Other development builds are rejected, not relabeled as compatible.

**This is experimental code, not a reproduced performance result.** CPU numerical, storage, request-lifecycle and tiny-model training tests are available. Native CUDA execution, large-model accuracy, and the paper's TTFT/SSD gains have not been established. See [validation](docs/validation.md), [design](docs/design.md), and the [paper-conformance audit](docs/paper_conformance.md).

## One-command benchmark

Use Python 3.12 or newer. The launcher preserves caller-relative paths and never downloads model weights.

### Self-contained CPU correctness run

```bash
./run_cacheslide_benchmark.sh --smoke --install --output ./results/cpu-smoke
```

This creates a new isolated environment and runs a tiny random safetensors model through **real adapter training, CCPE calibration, population and multi-token decoding**. The strict default checks same-context reuse and verifies that an incompatible shifted non-prefix request safely falls back to full plain CoPE. A verified fallback is not a cache hit or a shifted-reuse success. No vLLM import, pretrained download, language-quality claim or GPU timing is involved. With dependencies already installed, omit `--install`; select an interpreter with `CACHESLIDE_PYTHON=/path/to/env/bin/python` if needed.

To exercise the explicitly non-literal, approximate non-prefix reuse variant:

```bash
./run_cacheslide_benchmark.sh --smoke \
  --ccpe-position-policy mixed_bias_override --calibration-layer 1 \
  --max-tokens 5 --output ./results/cpu-engineering-variant
```

This permits mixed positional biases without a valid contextual-path requirement and initializes WCA after two shallow layers. It tests that variant's execution, not equivalence to strict CCPE or the paper's accuracy.

### Native vLLM benchmark

```bash
CUDA_VISIBLE_DEVICES=0 ./run_cacheslide_benchmark.sh \
  --install \
  --model /models/llama \
  --train-input /data/train.jsonl \
  --calibration-input /data/calibration.jsonl \
  --input /data/evaluation.jsonl \
  --output ./results/native-run \
  --train-device cuda:0 --calibration-device cuda:0 \
  --steps 100 --warmup 3 --repeats 10 --max-tokens 50 --run
```

The workflow verifies the pinned engine, trains the attention adapter, calibrates profiles, populates caches, warms one engine, and measures matched recompute/reuse pairs. Training and calibration use separate processes so their GPU allocations are released before native benchmarking. `--install` only creates a **new** environment, installs `vllm==0.29.0` for native mode, then executes the same request; it never modifies the current environment. Omit `--run` and `--install` to inspect the plan without training, GPU allocation, or output writes. Running with no model prints usage, not a model download.

Native mode also defaults to `--ccpe-position-policy strict_contextual --calibration-layer 0`. An incompatible canonical/current positional path causes full recomputation and invalidates a reuse benchmark. The two engineering flags above can be chosen explicitly for an approximate-reuse experiment; they are not silently enabled to obtain a latency ratio.

Use `--adapter /artifacts/adapter` to skip training; add `--profiles /artifacts/profiles` to skip calibration as well, using verified artifacts. `--profiles` requires `--adapter`. Rebuild pre-v0.3 profiles and caches: the v2 identity now binds the complete ordered chunk-role template. `--seed-input /data/seeds.jsonl` supplies separate population requests. Without explicit training/calibration inputs, those stages use `--input` for convenience; that is **not held-out accuracy evaluation**. Use representative, separate datasets for quality claims.

Output directories must be new. Cache files default to `<output>/cache`; use `--cache-root /new/ssd-cache` to select a new directory on an SSD. Results include `workflow.json`, stage logs, `benchmark/raw_outputs.jsonl`, and `benchmark/summary.json`. A failed stage preserves diagnostics and stops; it never silently switches from native vLLM to CPU. Invalid cache-hit receipts or mismatched output lengths invalidate acceleration claims.

The launcher does not kill GPU processes or manage cluster allocations. Choose an available GPU. If an idle-GPU keeper is in use, pause only the selected GPU before native execution and restore it after the process exits, including on error.

## Input format

Inputs are JSONL with **exact tokenizer token IDs**, not character offsets. Each request partitions its entire prompt into ordered `reuse` and `recompute` chunks:

```json
{"id":"query-1","prompt_token_ids":[1,41,42,7,81,82,9,2],"cacheslide":{"version":1,"operation":"reuse","namespace":"demo","task_id":"qa","chunks":[{"id":"document-a","role":"reuse","start":0,"end":3},{"id":"dynamic","role":"recompute","start":3,"end":4},{"id":"document-b","role":"reuse","start":4,"end":6},{"id":"query","role":"recompute","start":6,"end":8}]}}
```

These numbers illustrate the schema, not meaningful model text. Training rows need only `id` and `prompt_token_ids`. Calibration/evaluation rows require chunk metadata. The complete ordered sequence of chunk IDs and roles must match, as must reusable content and lengths. Dynamic spans may change content and length within that template; moving, renaming or relabeling a dynamic chunk creates a different identity. Matching identity alone does not guarantee a valid strict CCPE path. All dynamic rows and the last prompt row are computed.

## Decoupled source layout

```text
src/cacheslide_vllm/
  contracts.py, policy.py, config.py    Data and policies (stdlib only)
  position.py, profiles.py, attention.py  CoPE, profiles and shared attention policies
  wca.py                              Request-local selection and KV fusion
  slide.py, paged.py, storage.py        Slot mapping, native arena, RAM/SSD pages
  artifacts.py, reference.py, training.py  Verified adapters and training
  runtime.py                          Selective execution and dense fallback
  integration.py, worker.py, model.py  Instance-scoped native V1/V2 adapters
  compat.py, compatibility.json        Exact upstream source/config contract
  cli.py, workflow.py, workflow_smoke.py  Commands and orchestration
benchmarks/                           Direct paired benchmark entry
tests/                                Numerical, lifecycle and source contracts
docs/                                 Design and validation boundaries
run_cacheslide_benchmark.sh            One-command workflow
```

Policy, token-layout and planning modules do not import Torch or vLLM. Numerical/cache modules do not import native vLLM. Only the explicit worker/model integration depends on the engine. The same selective runtime can be checked with a CPU reference model and the actual packed-page adapter before CUDA deployment.

## Supported native scope

- Local, unquantized, bias-free **Llama or full-attention Mistral** safetensors checkpoints with SwiGLU.
- One request, one rank, eager execution, float16/bfloat16 and explicit `FLASH_ATTN` KV layout.
- No APC, chunked prefill, asynchronous scheduling, CUDA graphs, compilation, tensor/pipeline parallelism, speculative decoding, sliding-window attention, multimodal input, runtime LoRA or prompt logprobs.
- Verified **trained** CoPE/attention adapters are mandatory. No author-trained adapters are bundled; random embeddings cannot enable native reuse.
- V1/V2 cleanup invalidates old slot maps. Preemption replays the full prompt plus generated suffix and resumes consecutive decoding; this dense recovery is not an accelerated cache hit.

WCA defaults to Algorithm 2's first-layer initialization (`--calibration-layer 0`), the `0.26` correction fraction, and literal `CKSim < 0.12` gate every four layers. In a CoPE model, first-layer pre-attention K can be token-local and identical for reused content: zero deviation legitimately selects no correction tokens. The optional two-shallow-layer initialization, convergence ambiguity, raw fusion weights and selected-query policy are explicit in [design.md](docs/design.md).

## Measurement boundaries

Both requested paths use the **same trained CoPE/LoRA model and configured profiles**, not an untouched RoPE baseline. Strict positional failures clear every layer profile for that request and retry plain CoPE; such runs cannot establish a reuse speedup. Population, training, calibration and warmups are excluded from measured means. The ratio is `mean(recompute latency) / mean(reuse latency)`, valid only for request-bound cache hits without fallback.

One output token measures offline prefill-plus-one-token latency; longer output measures offline generation latency. Neither is streaming TTFT. Generated-token agreement and token-ID multiset F1 are consistency checks, not dataset answer F1. Serial measurements are not concurrent QPS.

SLIDE selects in-place writes only after actual baseline device writes complete; a pending host load uses relocation and later decode can reuse its safe old slots. This **does not release vLLM's reserved block pool**. Device sidecars, dense gathers and hidden/residual snapshots add memory costs. Physical-page selected-token counts and clean-first/dirty-count ordering are advisory metadata, not an integrated mutable GPU-page SSD eviction/writeback pipeline. Immutable runtime snapshots remain separate. Storage counters are managed payload/file bytes, not SSD hardware write amplification or process RSS. CCPE profiles have explicit quadratic element budgets; a fused/scalable long-context CCPE kernel remains future work.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints/cpu-tests.txt '.[test]'
CACHESLIDE_VLLM_SOURCE=/path/to/vllm-v0.29.0 \
  .venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests benchmarks
```

`CACHESLIDE_VLLM_SOURCE` enables pinned fingerprint/AST tests; without it source-dependent tests are explicitly skipped. CI checks out that commit separately and runs CPU smoke. Individual commands remain available: `cacheslide train`, `calibrate`, `inspect`, `check-engine`, `generate`, and `bench`.

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
