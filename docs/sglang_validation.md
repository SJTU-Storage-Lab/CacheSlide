# SGLang validation snapshot — 2026-09-15

This documents the **SGlang-CacheSlide** branch, not the retained historical
vLLM [validation](validation.md). Package/source verification and CPU numerical
tests are separate from native engine execution.

## Confirmed

- **539 CPU tests passed**, with one explicitly skipped opt-in CUDA module.
  Both pinned upstream source checkouts were enabled. The suite covers
  shared core, retained vLLM adapters and the new SGLang integration; this count
  is not a count of native GPU engine tests.
- SGLang 0.5.19 at `0bcd822377da7b5718e674eaf9c870d349424dd1` is pinned.
  The official CPython 3.12 wheel was checked against all **30** audited source
  fingerprints. Version-only or import-shadowed installations are rejected.
- The **0.4.0 wheel** was built and installed into a separate target directory;
  its strict CPU training/calibration/reuse workflow passed from outside the
  repository. This used existing numerical dependencies, not a complete native
  SGLang installation. Ruff and formatting checks passed for all 72 Python files.
- Strict CPU smoke: five requests, two reuse attempts, one hit and two guarded
  positional fallbacks (shifted recompute and shifted reuse). Fallback logits
  match plain CoPE; fallback is not shifted-reuse success.
- Explicit `mixed_bias_override` plus `calibration_layer=1`: two of two reuse
  attempts hit with no fallback. CPU tests compare logits and per-layer K/V
  against the dense shared runtime through three decode steps. This validates
  execution/consistency, not language quality or exact shifted recomputation.
- Smoke performs actual positive-step adapter training and CCPE calibration on
  a tiny random checkpoint, using the real split-K/V pool adapter on CPU without
  importing native SGLang or vLLM. It reports no GPU timing ratio.
- Official dependency resolution succeeded for 203 packages. This is resolver
  evidence, not proof that the complete wheelhouse was downloaded or installed.
- **6 opt-in CUDA pool tests passed** on the server (7.53 seconds), covering
  float32 and bfloat16, actual CUDA events, nondefault streams, in-place and
  relocated updates, pinned holes, canonical decode mirrors and retirement.
  This is GPU storage-path evidence, not native SGLang engine execution.

## Coverage and reproducibility

| Tests | Boundary exercised |
| --- | --- |
| `test_sglang_compat.py` | Stable source fingerprints, import shadowing and unsupported modes |
| `test_sglang_plugin.py`, `test_sglang_integration.py` | Applied parent/child hooks, worker attestation, real request binding, upstream alias propagation, retirement before free, replay and shutdown |
| `test_sglang_model.py`, `test_sglang_backend.py` | Native projections/loading/logits interfaces and guarded attention backend |
| `test_sglang_pool.py` | Split native buffers, ready/pending paths, ownership/generation, pins, holes and dense per-layer parity |
| `test_sglang_receipts.py` | Atomic immutable receipt publication, path guards and exact output identity |
| `test_sglang_workflow.py` | Real CPU training/calibration, isolated installation, caller-cwd safety, path/capacity checks and invalid-ratio failure |
| Shared core tests | CoPE gradients/path validation, WCA/fallback semantics, artifact/cache identity, storage faults and asynchronous slot lifecycle |

Run the [README](../README.md) commands with exact upstream checkouts provided
through `CACHESLIDE_SGLANG_SOURCE` and `CACHESLIDE_VLLM_SOURCE` for source-dependent
tests; absent checkouts cause explicit skips. The standard CPU suite does not
implicitly reserve a GPU.

## GPU and deployment status

**No native SGLang GPU engine run has been completed.** Separate opt-in tests in
`test_sglang_cuda_pool.py` passed all six cases using tiny CUDA tensors, actual
event completion, stream ordering, pins and retirement. The test inserts a
short real device dependency at the completion boundary to expose pending
publication; the host-load guard is synthetic, not an SSD transfer. The first
attempt's pre-write delay was consumed by implicit index-copy synchronization;
only the test timing was corrected, not production code. This does not validate
SGLang engine startup, model loading, scheduling, sampling or native generation.

The assigned device reports `NVIDIA H20G`, compute capability **10.3**, driver
595.58.03, with Torch 2.13.0+cu130 / Python 3.12.3. Only its existing keeper was
temporarily paused; the other seven GPUs stayed occupied. The test wrapper
restored that UUID after both attempts, including the failed first run. The
test-only Python wheels were extracted to a fresh isolated directory, without
changing the existing server environment. CPU runs skip this opt-in module.

The inspected server's current dependency versions do not match the complete
pinned SGLang environment, and direct PyPI access was unavailable. No existing
environment or proxy was modified, and no complete offline SGLang installation
has been established. Resolver success does not remove this deployment boundary.

## Not established or not implemented

- Native SGLang GPU generation, representative trained-model accuracy and
  large-model TTFT/concurrent-QPS improvements.
- Mutable physical GPU-page SSD eviction/coalesced writeback, native pool
  release or greater admission capacity.
- A fused/scalable contextual-attention kernel or long-context profile scaling.
- Sustained SSD device write amplification; file-byte counters are not WAF.

The historical vLLM [design](design.md), [validation](validation.md) and
[paper audit](paper_conformance.md) remain preserved for comparison. Their
engine-specific evidence does not certify this backend. See the
[SGLang design](sglang_design.md) for the implemented lifecycle and policy scope.
