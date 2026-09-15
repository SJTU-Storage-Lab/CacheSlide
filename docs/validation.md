# Validation snapshot: 2026-09-15

Environment: Python 3.12.14, Torch 2.13.0, macOS ARM64 CPU.
CUDA was unavailable in this test environment. These are correctness and
integration-contract results, not GPU performance measurements.

## Executed checks

- **311 tests passed**, with no skips, after setting `CACHESLIDE_VLLM_SOURCE`
  to a checkout of vLLM tag `v0.29.0`.
- Ruff passed for `src`, `tests`, and `benchmarks`; `git diff --check` passed.
- Built and installed the standalone `cacheslide-vllm` 0.2.0 wheel; the current
  tree no longer ships the legacy engine, CUDA sources or build system.
- Loaded the installed package from outside the repository, independently of
  `src/`, and ran its complete CPU workflow successfully.
- The installed `cacheslide check-engine` command verified all **26** audited
  source fingerprints at `98dff2a81d747d1dba01a47f939f48c3526d4206`.
- Ran the root `run_cacheslide_benchmark.sh` from outside the repository:
  two real optimizer steps, CCPE calibration, 15 requests including seven reuse
  requests, and four decode steps per request (`--max-tokens 5`).
- Verified configuration/policy/CLI/workflow imports with Torch, safetensors and
  vLLM explicitly blocked. Planning and help do not allocate GPU memory.

The native runner selection defaults to V2. CPU lifecycle tests execute selected
actual upstream admission/removal methods and inspect the audited model call
boundary, in addition to integration mocks. They cover original-prompt identity,
full generated-history replay after preemption, context cleanup on errors/dummy
passes, retirement, and subsequent decode. These checks are not a CUDA engine
startup test.

The six-layer synthetic GQA fixture performs real next-token CE adapter updates
and profile calibration. Tests cover unchanged and shifted non-prefix contexts,
selected-row execution, WCA promotion/input-state restoration, and multiple decode
steps. The actual packed-page CPU adapter and dense reference agree bit-for-bit
on logits and gathered per-layer K/V in the paired tests. These fixtures are not
trained language models and do not establish task quality.

Fault tests cover corrupt/checksummed malformed snapshots, disk-write failures,
interrupted population, dtype identity changes, in-place residual mutation during
a failed reuse attempt, incomplete cache cleanup, asynchronous selected writes,
load completion, native slot ownership, and request retirement. Benchmark tests
reject stale receipts and suppress ratios for cache misses or fallback.

Selected-write tests additionally check pinned canonical mirrors, atomic
pre-mutation rejection, pin/write races, retirement during late completion, and
event completion without holding the map mutex. Device-event ordering is tested
using controlled CPU event doubles; that does not establish CUDA throughput.

CoPE regression tests retain gate gradients at integer contextual positions;
reference training also preserves the native checkpoint's output logit scale.
Native HF overrides must preserve trained embedding/LM-head tying and reject
unsupported projection biases, activations and model families before loading.
The one-command workflow preserves stage logs on failure and rejects invalid
reuse receipts; it does not substitute a reference backend for native failures.

## Not yet established

- Native CUDA engine startup and end-to-end GPU generation.
- Representative model quality after CoPE training or large-model TTFT/QPS gains.
- A fused contextual-attention kernel or scalable long-context CCPE profiles.
- Reduced native vLLM block-pool reservations/admission capacity.
- Sustained device-level SSD write amplification or concurrent-serving gains.

The available GPU server's vLLM development build is outside the pinned stable
source contract; it was not relabeled or modified to bypass validation. The
official stable Linux wheel was downloaded locally and its publisher SHA-256
and all 26 source fingerprints passed. Transfer to the GPU host was interrupted;
no stable environment or CUDA run was completed. The existing environment also
has different exact versions of FlashInfer, CUTLASS DSL and QuACK; these must be
resolved in a separate environment rather than ignored. See the
[design boundaries](design.md) before interpreting the implementation as a
reproduction of the paper's reported performance.
