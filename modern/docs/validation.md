# Validation snapshot: 2026-09-14

Environment: Python 3.12.14, Torch 2.13.0, macOS ARM64 CPU.
CUDA was unavailable in this test environment. These are correctness and
integration-contract results, not GPU performance measurements.

## Executed checks

- **215 tests passed**, with no skips, after setting `CACHESLIDE_VLLM_SOURCE`
  to a checkout of vLLM tag `v0.29.0`.
- Ruff passed for `src`, `tests`, and `benchmarks`; `git diff --check` passed.
- Built and installed the standalone wheel without installing the legacy tree.
- From outside the repository, the installed `cacheslide --help` command worked.
- The installed `cacheslide check-engine` command verified all **18** audited
  source fingerprints at `98dff2a81d747d1dba01a47f939f48c3526d4206`.

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

## Not yet established

- Native CUDA engine startup and end-to-end GPU generation.
- Representative model quality after CoPE training or large-model TTFT/QPS gains.
- A fused contextual-attention kernel or scalable long-context CCPE profiles.
- Reduced native vLLM block-pool reservations/admission capacity.
- Sustained device-level SSD write amplification or concurrent-serving gains.

The available GPU server's vLLM development build is outside the pinned stable
source contract; it was not relabeled or modified to bypass validation. See the
[design boundaries](design.md) before interpreting the implementation as a
reproduction of the paper's reported performance.
