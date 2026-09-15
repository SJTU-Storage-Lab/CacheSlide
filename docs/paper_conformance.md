# Paper-conformance audit

Scope: the standalone vLLM adaptation, version 0.3.0, audited 2026-09-15 against
[CacheSlide, FAST '26](https://www.usenix.org/conference/fast26/presentation/liu-yang).
Page references below give **printed proceedings page / PDF page including the
cover**. This is a source-and-correctness audit, not a reproduction of the
paper's quality or performance results. Current execution evidence belongs in
[validation.md](validation.md); operational details are in [design.md](design.md).

## What matches, what is interpreted, what is missing

| Paper anchor | Current implementation and classification | Code / regression evidence |
| --- | --- | --- |
| §4.2, Algorithm 1, Figure 7; p89 / PDF8: task histogram and fixed reuse chunks | **Interpreted representation.** Real trained CoPE traces retain dynamic gates before fixed/fixed projection; a joint quantized histogram keeps an observed representative. The paper does not specify encoding shape, quantization, head/layer granularity, or a complete histogram update/insertion rule. | `profiles.py`, `position.py`; genuine-trace, joint-histogram, identity/budget tests in `test_profiles.py` and `test_position.py` |
| Algorithm 1 lines 8–16: the same chunk template | **Enforced.** v2 KV/profile keys bind every ordered chunk ID/role plus fixed content/length. Dynamic content and length may vary, but changing the template misses. Old artifacts are not silently migrated. | `contracts.py::RequestPlan.cache_key`; ordered-template and dynamic-variation tests in `test_contracts.py` |
| Algorithm 1 fixed encodings plus current dynamic encodings, built on CoPE contextual sums | **Safety boundary, not a paper-completeness claim.** Strict default validates the complete hybrid key path. An invalid path causes full-prefill plain-CoPE recovery with all profiles cleared and no cache hit. Necessary path constraints do not prove equality to current gates or quality. | `attention.py::CanonicalPositionPolicy`, `position.py::validate_contextual_path`, `runtime.py`; strict-hybrid and full-fallback tests |
| §4.3, Algorithm 2 lines 1–6, Figure 8; p90 / PDF9: full first-layer initialization | **Literal default:** `calibration_layer=0`. Positive squared K deviation, fraction 0.26, deterministic top-k. With unrotated token-local first-layer K, repeated fixed tokens can have zero error; no correction token is invented. | `config.py`, `wca.py::WCAState.initialize`; `test_literal_first_layer_wca_does_not_invent_nonzero_error` |
| Algorithm 2 lines 10–14: fusion, weight update, four-layer CKSim gate | **Literal order/comparator with explicit numerics.** Prior alpha fuses K/V before current alpha is stored; raw alpha may exceed one. Default removes when cosine < 0.12 every four layers despite the prose's convergence interpretation. Epsilon, ceil budget, tie-breaking and zero-norm handling are explicit choices. | `policy.py`, `wca.py`; weight-order, raw-ratio, literal-gate, request-isolation and overflow tests |
| §4.3 updated-segment association and promotion | **Interpreted policy.** Selected queries see causal dynamic keys plus self, after full causal gates are counted. Promoted rows restore cached layer-input hidden/residual states; this remains approximate. Mandatory dynamic/final rows stay fresh. | `attention.py::SelectedAssociationPolicy`, `runtime.py`; `test_wca_association_is_applied_after_all_causal_gates`, runtime promotion tests |
| §4.4, Figure 9; p91 / PDF10: load-first versus write-first relocation | **Both slot branches implemented.** Completed baseline device writes permit exclusive/unpinned in-place updates; pending host reads retain relocation. Publication waits for device completion and source holes await loads/pins. Raw staging does not remove WCA's old-KV fusion dependency. | `paged.py::write_selected_ready`, `promote_selected`; load-first and pending-load tests in `test_paged.py`, `test_runtime_paged.py` |
| Figure 9 decode overwrites vacated original slots before using new slots | **Request-local mapping implemented.** Only eligible native holes are reused; shared/pinned/loading slots are excluded, and attention follows the logical mapping. Device writes complete before new decode mappings publish. This is not native pool release. | `slide.py`, `paged.py`; hole/pin/retirement and asynchronous decode-publication tests |
| §4.4: clean first, dirty pages in descending selected-token count | **Physical policy metadata implemented; spill integration missing.** Page counts follow actual slot relocation; guarded pages are excluded. `spill_order()` is advisory, not an eviction reservation or backing-version protocol. | `LayerSlotMap.page_metadata`, `NativePagedKV.page_metadata` / `spill_order`; physical-page count/order tests |
| §4.4: dirty writeback aggregation, parallel load/write and SSD pressure | **Incomplete system path.** Runtime saves immutable layer snapshots with `selected_count=0`; its actual store eviction is clean/LRU. Individual file spills, store-wide I/O locking and host prefetch do not implement mutable physical-page sequential overwrite, GPU residency eviction or a dedicated asynchronous H2D pipeline. | `runtime.py::_put` / `_prefetch` / `_read`, `storage.py`; store budget/checksum/fault tests validate only these implemented APIs |
| §5.1–5.5, Figures 10–14; p91–94 / PDF10–13 | **Not reproduced.** Paper uses vLLM 0.8.5 and A100 hardware, task quality, batch/beam concurrency and storage workloads. This package pins vLLM 0.29.0 and currently establishes CPU correctness/source contracts only. | `compatibility.json`; [validation](validation.md), benchmark receipt tests |

## Why strict CCPE may decline a reuse request

For dense causal keys, a CoPE reverse sum has adjacent decreases in `[0, 1]`
and a self count at most one. A genuine three-token trace can give fixed-token
counts `[1.5, 0.5]`; inserting those into a genuine five-token current path
`[2.5, 2, 1.5, 1, 0.5]` yields `[1.5, 2, 1.5, 1, 0.5]`. The increase from
1.5 to 2 cannot arise from nonnegative sigmoid gates. Both traces being genuine
does not make their mixture valid. `test_strict_policy_rejects_real_three_to_five_token_hybrid`
reproduces this boundary with real gates.

The implementation neither rescales canonical values nor silently accepts this
mixture in strict mode. `CCPEPositionError` clears every layer profile, abandons
partial population/arenas, and restarts the whole prompt before sampling.
The receipt records `plain_cope_fallback`, `position_policy_fallback=true`,
`fallback=true`, and `cache_hit=false`. A successful safety fallback is not
successful cross-position reuse. Structural validity alone is also not proof
of exact recomputation or preserved task accuracy.

## Explicit engineering variants and numerical choices

- `--ccpe-position-policy mixed_bias_override`: accepts the historical hybrid
  bias without the contextual-path check. It is not equivalent to strict CCPE.
- `--calibration-layer 1`: two shallow full layers before WCA initialization,
  instead of Algorithm 2's first layer. It often exposes contextual K deviation
  in CoPE models, but is not the literal default.
- `--convergence-mode distance_lt`: uses `1 - cosine < threshold`; the default
  preserves the printed comparator. Low-cosine-as-convergence prose does not
  justify silently reversing that comparator.
- `--weight-update same_layer` and `--selected-attention full_causal` are separate
  experiments. `clamp_alpha=true` is a non-default settings/API experiment.
- CoPE interpolation uses `floor + 1` to retain the right-hand gradient at
  integer knots. Forward values match floor/ceil interpolation; the derivative
  at knots is an explicit choice, not literal appendix-code identity.
- `WCANumericalError` permits safe full-prefill recovery for numerical failure.
  Shape, policy and layer-order errors remain ordinary errors, not hidden by a
  broad fallback catch.

## Claims this repository does not make

Figure 12(c), p94 / PDF13, plots **total write size in TiB** and describes
write-amplification reductions. Host payload/page-file byte counters do not
measure SSD flash writes or hardware WAF. Device slot-record coalescing does
not implement coalesced SSD dirty-page overwrite. Immutable snapshots are not
labeled dirty to claim that the physical-page policy is integrated.

Physical hole reuse and lazy sidecars do not release the native vLLM reserved
block pool or establish greater request admission capacity. Dense gathers,
sidecars and input-state snapshots remain costs. Serving support is currently
one request/rank with eager full-attention Llama/Mistral; the paper's batch,
beam and multi-GPU experiments are outside that scope.

The paired benchmark compares the same trained CoPE/adapter configuration,
not the paper's unchanged positional-encoding baselines. Offline one-token
latency is not client-observed streaming TTFT; serial latency is not concurrent
QPS; token-ID consistency is not dataset answer F1. Only genuine request-bound
reuse hits without fallback can produce a benchmark latency ratio. No quality,
SSD, pool-memory or GPU speedup claim follows from fewer computed token layers.
