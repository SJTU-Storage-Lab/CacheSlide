# Validation: paper-conformance audit, 2026-09-15

> This is the historical vLLM 0.3.0 validation record for `main` at `3f220b8`.
> It does not describe the SGLang branch's current test count or GPU checks;
> see [SGLang validation](sglang_validation.md) for those results.

Environment: Python 3.12.14, Torch 2.13.0, macOS ARM64 CPU. CUDA was
unavailable. These are correctness and integration-contract checks, not GPU
performance measurements. See the [paper-conformance audit](paper_conformance.md)
for literal defaults, explicit variants and missing system functionality.

## Current audit results

- **423 tests passed, with no skips**, using the exact vLLM source checkout.
  Ruff passed for source/tests/benchmarks; `git diff --check` passed.
- Built and installed the **0.3.0 wheel** into an independent target directory.
  Its strict CPU workflow ran successfully from outside the repository,
  independently of the source-tree import path.
- Strict one-command CPU smoke completed: five requests, two reuse attempts,
  one cache hit, and two guarded positional fallbacks (shifted recompute and
  shifted reuse). Same-context logits agree; shifted fallback logits were
  checked against full plain CoPE. The installed wheel reproduced those counts.
  This does **not** report shifted reuse success.
- The explicit `mixed_bias_override` / `calibration_layer=1` smoke completed:
  five requests, two reuse hits, no fallback, three decode steps per request,
  and same-context logits agreement. This verifies that engineering variant's
  execution, not shifted-request semantic equivalence or model quality.
- Both smoke modes performed real optimizer updates, calibrated real contextual
  traces, used the actual packed-page CPU adapter, and reported no timing ratio,
  no native vLLM execution and no GPU-performance claim.

These results supersede the earlier 311-test / 0.2.0-wheel snapshot.

## Regression coverage

| Boundary | Evidence |
| --- | --- |
| CoPE causal sigmoid gates, GQA, interpolation and gradients | `test_position.py`, including integer-knot gradients and cumulative-sum roundoff |
| Strict canonical/current path validity and post-gate selected visibility | `test_attention_policies.py`; shared `cope_attention` transforms, not duplicated runtime math |
| Genuine full-context calibration, joint histograms, artifact identity and budgets | `test_profiles.py`, `test_position.py` |
| Ordered chunk ID/role template with variable dynamic content/length | `test_contracts.py` |
| Literal first-layer WCA, raw/previous-layer weights, literal cosine gate and transactional failures | `test_wca.py`, `test_runtime_failures.py` |
| Whole-prefill invalid-position recovery; clear every profile; no partial publication or false cache hit | `test_runtime_failures.py::test_invalid_hybrid_positions_retry_plain_cope_and_do_not_publish` |
| Load-first native in-place writes and pending-load relocation; sidecars, source holes and selected-page counts | `test_paged.py`, `test_runtime_paged.py` |
| Pins, shared slots, async writes, retirement and device-completion publication, including decode | `test_slide.py`, `test_paged.py`; CUDA event behavior uses controlled CPU doubles |
| Request isolation, dense/paged parity, promotion input-state restoration and decode continuity | `test_runtime.py`, `test_runtime_paged.py`, `test_runtime_replay.py` |
| Snapshot corruption, interrupted population, disk failure, restart and capacity accounting | `test_storage.py`, `test_runtime_failures.py` |
| V1/V2 source/lifecycle contracts, preemption replay and setup rejection | Fingerprint/AST and integration tests with the exact upstream checkout |
| Standard-library-only planning/help, caller-cwd safety, isolated explicit installation and stage failures | `test_package_boundaries.py`, `test_workflow.py` |
| Request-bound hit receipts and suppression of invalid ratios | CLI/benchmark tests and native workflow validation |

The native contract is vLLM **0.29.0**, commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`, with 26 audited source fingerprints.
V2 is default; V1 is opt-in. Set `CACHESLIDE_VLLM_SOURCE` to that checkout for
source-dependent tests; otherwise they explicitly skip. CPU lifecycle tests
exercise selected actual upstream methods as well as mocks. Neither source
verification nor a CPU event double is a CUDA engine startup test.

The tiny six-layer GQA checkpoint is random and the adapter receives only a few
updates. Generated-token agreement is a consistency test, not dataset answer F1.
Some integration fixtures explicitly choose engineering variants to exercise
nonempty WCA selection; they must not be read as default strict-mode results.

## Not established or not implemented

- Native CUDA startup/end-to-end generation and representative trained-model
  accuracy, TTFT, concurrent QPS or the paper's multi-GPU/beam results.
- A fused contextual-attention kernel or scalable long-context CCPE profiles.
- A physically integrated mutable dirty-GPU-page SSD spill/writeback path,
  combined sequential dirty-page overwrites, or a fully decoupled H2D pipeline.
- Native vLLM pool release, smaller reserved pool or increased admission capacity.
- Sustained device-level SSD write amplification. Managed page/file byte counts
  and advisory selected-count eviction order are not hardware WAF measurements.

The available server's development vLLM build is outside the stable source
contract. Prior local verification of the official Linux wheel did not establish
an installed stable GPU environment or a successful CUDA run. No existing server
environment is relabeled compatible, and native failure never switches to a CPU
benchmark. [Design](design.md) documents the remaining implementation boundaries.
