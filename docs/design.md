# Implementation design and paper interpretation

> Historical vLLM 0.3.0 design, published on `main` at `3f220b8`. On this
> branch, engine-independent modules have moved to `src/cacheslide_core/`;
> vLLM adapters remain in `src/cacheslide_vllm/`. For the current SGLang
> integration and commands, see [SGLang design](sglang_design.md) and the
> [branch README](../README.md).

This document describes executable behavior in `src/cacheslide_vllm/`, including choices where the [CacheSlide paper](https://www.usenix.org/conference/fast26/presentation/liu-yang) leaves implementation details ambiguous. Contextual position encoding follows [CoPE, arXiv:2405.18719, section 4](https://arxiv.org/abs/2405.18719). The implementation includes operational interpretations and bounded reference paths; it does not claim access to unpublished author weights or reproduction of the paper's performance results. The [paper-conformance audit](paper_conformance.md) maps individual paper claims, defaults, variants and missing integrations to executable tests.

## Package and engine boundary

The original vLLM 0.8.5 vendor tree has been removed from the current repository; Git history retains it. The standalone `cacheslide-vllm` package pins stable `vllm==0.29.0`; `compatibility.json` also fingerprints 26 audited source files at commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. A different version or modified audited file is a compatibility failure. The observed `0.29.1rc1.dev5` server build is therefore outside this contract, even though its version is newer. Planning, configuration, token contracts and policy validation use only the standard library; numerical modules never import native vLLM.

`plugin.py` registers one new architecture. `model.py` subclasses native Llama attention/layer/model classes and retains their projections, fused RMSNorm/residual conventions, SwiGLU, checkpoint loader, and sampler. Trained adapter parameters are registered under `model.cacheslide_adapters`; the native loader accounts for these independently loaded parameters without expecting them in backbone shards. No existing native model class is globally patched.

`worker.py` validates the configuration, then `integration.py` wraps only that runner instance. Model Runner V2 is the default; V1 remains an explicit compatibility path. V2 hooks request admission, input preparation, model execution and native request removal, retaining the original prompt separately from generated history. A `ContextVar` carries an immutable `StepContext` into the model call and is restored even after exceptions or dummy passes. Context includes request ID, prompt IDs, absolute query positions, and frozen request metadata. Request cleanup retires the associated CacheSlide state.

Preemption invalidates the old arena and replays the full original prompt plus its generated suffix, checked against actual runner input IDs. This recovery uses dense computation, preserves the original request/cache identity, and neither publishes a new snapshot nor claims an accelerated hit. Its allocation budget includes the generated suffix. Subsequent decode positions advance consecutively from the full replay length. Decode metadata must remain identical.

The gate requires one scheduled request, one rank in every parallel dimension, eager execution in Model Runner V2 or V1, no CUDA graphs/compilation, no APC/chunked prefill/async scheduling, and the audited `FLASH_ATTN` native cache geometry. Native inference accepts float16/bfloat16, bias-free Llama or full-attention Mistral, without quantization, sliding windows, heterogeneous attention, multimodal input, speculative decoding, prompt logprobs, or external LoRA requests.

## Token identity and artifacts

`RequestPlan` is an exact partition of integer prompt token IDs. The `CacheSlide-KV-v2` and `CacheSlide-CCPE-v2` identities incorporate model/adapter identity, namespace, task, the complete ordered fixed layout, the ordered ID/role pair of **every** chunk, and layer. Each fixed-layout entry includes chunk ID, content digest, and exact token length. The runtime also binds its profiles and dtype to persistent cache identity. Changing fixed content/context, or moving/renaming/relabeling any dynamic chunk, creates a miss. Dynamic content and length may vary only within the same ordered template. This restores Algorithm 1's same-template requirement; v1 caches/profiles must be regenerated, not silently reinterpreted.

All dynamic tokens are mandatory. The final prompt token is also mandatory, because native sampling indexes its original scheduled row. Selective execution returns the original output-row count; uncomputed prompt rows are placeholders and are never exposed as prompt logprobs.

Adapter artifacts bind learned tensors to exact local backbone-file SHA-256 hashes and model geometry. Loading verifies positive completed training steps, checksums, expected layer tensor keys/shapes, and finite weights. Native HF overrides must also preserve embedding/LM-head weight tying, geometry and output numerics; unsupported activations, projection biases and model families are rejected. No remote model code or pickle checkpoint is executed. Profile artifacts separately bind canonical contextual positions to the trained adapter, layer, namespace/task key, and fixed layout. They use strict JSON plus safetensors and reject incompatible bounds, ordinals, layouts, or checksums.

## Trained contextual attention

`reference.py` implements a frozen Llama/Mistral backbone using local safetensors. It exposes the native projection tuple convention and fused residual layer interface so CPU integration tests can exercise the same selective runtime. Only attention adapter parameters are trainable: learned CoPE embeddings and low-rank updates to concatenated Q/K/V and attention output projections.

`training.py` minimizes causal next-token cross-entropy on explicit token sequences. It cycles through the supplied sequences, performs actual optimizer steps, checks finite loss/gradients, and exports only after the requested updates finish. The reference applies the checkpoint's `logit_scale` after its LM head, consistently with native logits; head dimension and RMSNorm epsilon are also identity-checked. A positive-step artifact is an operational training result, not proof of model quality or equivalence to the original RoPE backbone. No pretrained author CoPE weights are bundled.

For Q shaped `[query, query_head, head_dim]` and K/V shaped `[key, kv_head, head_dim]`, GQA repeats KV heads to match query heads. For each query/head:

```text
L[i,j] = dot(Q[i], K[j]) / sqrt(head_dim)
g[i,j] = sigmoid(L[i,j]) for allowed causal keys, else 0
p[i,j] = clamp(sum(g[i,k] for k >= j), 0, max_positions - 1)
E = learned matrix [head_dim, max_positions]
b[i,j] = interpolate(Q[i] @ E, floor(p[i,j]), min(floor(p[i,j]) + 1, max_positions - 1))
A = softmax(mask_causal(L + b))
output = A @ V
```

Future keys are masked before sigmoid and reverse cumulative summation. Position bias remains query- and head-dependent; the learned embedding matrix is shared across heads within a layer. It is not a RoPE rotation and cannot be represented by a one-dimensional vector of shifted token offsets. Gradients pass through the gates, interpolation fractions, and learned embeddings. Using the next interpolation bin (`floor + 1`), rather than `ceil`, preserves the right-hand gate gradient at integer positions; the final saturated bin remains clamped. This has the same forward interpolation but is an explicit derivative choice at integer knots, not a verbatim reproduction of CoPE's appendix code.

Reference and native runtime attention both call `position.py::cope_attention`; `attention.py` supplies canonical-position and selected-association policies, without reimplementing logits, gates, interpolation or softmax. The shared kernel chunks the query axis during ordinary inference. Returning full traces is an explicit calibration operation. Native attention still materializes logical K/V gathered from the native arena; this is not an optimized fused CoPE kernel.

## CCPE calibration convention

The paper describes a task-conditioned encoding histogram but does not fully specify the artifact construction and lookup procedure. This implementation makes that procedure explicit:

1. Run the verified trained model over each complete calibration prompt.
2. At every layer, capture contextual positions from real Q/K logits with all causal dynamic-token gates present.
3. Extract fixed-query × fixed-key positions only after the full contextual computation. Retain source logits in memory long enough to verify the gate equation.
4. Replace absolute position metadata with compact ordinals in the ordered fixed-token sequence. This permits the same fixed chunks to appear after dynamic spans of different lengths.
5. Group by adapter, namespace/task, complete ordered template and fixed layout, and layer. Quantize each complete encoding pattern for histogram membership, select the most frequent joint pattern, break pattern ties lexicographically, and retain an actual representative trace's values.

Independent per-cell modes are deliberately not assembled into a never-observed encoding. Within the winning quantized pattern, the first observed sample supplies the stored continuous values. The default histogram bin width is an explicit implementation parameter, not a recovered author setting.

At runtime, canonical values replace only the fixed-query/fixed-key cells. Other cells use current CoPE positions. Lookup selects the requested fixed query ordinals and rejects unknown keys or shapes. Decode queries are new contextual queries and are not extrapolated from fixed profiles. This fixed/fixed representation is an engineering interpretation of Algorithm 1, which does not specify the shape of its encoding, head/layer granularity, histogram quantization or a complete insertion rule.

The default `ccpe_position_policy=strict_contextual` checks each resulting hybrid path in the **actual complete key coordinate frame**, after substitution and before attention. A causal sigmoid reverse-sum path must be nonnegative and nonincreasing, with an adjacent decrease at most one and a self count at most one; gaps in projected coordinates allow the corresponding number of intervening gates. The validator tolerates floating-point cumulative-sum roundoff and ignores masked/future entries. These are necessary structural conditions, not proof that the path equals the current Q/K gate sums or a guarantee of semantic accuracy.

Canonical fixed cells plus current dynamic cells can violate those conditions even when each source trace is genuine. If any layer/query tile fails, `CCPEPositionError` aborts the entire prefill before sampling. The runtime drains pending I/O, retires arenas, discards uncommitted population, clears **all** layer profiles and WCA state, then recomputes every layer using plain trained CoPE. Receipts report `cache_hit=false`, `fallback=true`, `position_policy_fallback=true`, and `ccpe_position_policy=plain_cope_fallback`; decode continues without canonical substitution. Population, recompute and reuse all use the same validation boundary.

The explicit `mixed_bias_override` variant performs the historical cell substitution without validating a contextual path. It can therefore supply a positional bias no sigmoid-gate path could produce. It is an approximate engineering experiment, not a silent strict-mode repair. Profiles are not projected or rescaled to make invalid paths appear valid.

The full source trace is `heads × prompt_length²` per layer/sample; each canonical profile is `heads × fixed_length²`. Calibration checks the aggregate retained trace/profile budget before allocation, and profile loading checks tensor shapes and element totals before materialization. These budgets prevent accidental oversized artifacts; they do not turn this representation into a scalable long-context scheme. Exported profile artifacts omit raw source logits, Q/K tensors, request token IDs and raw dynamic content; their lookup identity still binds the ordered template.

## Weighted Correction Attention

`WCAState` belongs to one request. It clones identity masks and maintains initial deviations, positive-error candidates, the selected set, prior weights, and permanently removed tokens. Token IDs resolve deterministic ties.

The per-token initialization error is the squared K difference summed across all heads and dimensions:

```text
error[i] = sum((K_new[i] - K_cache[i])²)
candidate = reusable tokens with error > 0
budget = min(ceil(0.26 * reusable_token_count), candidate_count)
selected = highest-error candidates, with token-index tie breaking
active = selected union mandatory
```

Dynamic tokens never enter the reusable candidate set. Mandatory reused rows, such as a final prompt token marked reusable, still receive fresh K/V.

The default zero-based `calibration_layer=0` follows Algorithm 2's first-layer initialization: fully compute layer 0, initialize WCA there, and use active rows from layer 1 onward. CoPE adds a score bias without rotating K. First-block pre-attention K can therefore be token-local and identical for repeated content despite changed preceding context. Zero deviation legitimately produces an empty correction set; the runtime does not manufacture an error to select tokens.

`calibration_layer=1` is an explicit two-shallow-layer engineering variant: fully compute layers 0 and 1, initialize WCA from contextual K in layer 1, and compute sparsely from layer 2 onward. It is not identical to Algorithm 2's literal initialization. Later selected rows run the actual projection, attention, MLP, and residual computation. The CLI option names the WCA observation layer, not the number of layers captured by offline CCPE calibration.

For each selected token, with epsilon `1e-8` by default:

```text
alpha_new = ||K_raw_new - K_cache||² / (||K_cache||² + epsilon)
K_fused = alpha_used * K_raw_new + (1 - alpha_used) * K_cache
V_fused = alpha_used * V_raw_new + (1 - alpha_used) * V_cache
```

One per-token alpha is shared by K and V. The default `weight_update=previous_layer` uses the prior stored alpha for fusion, then stores the current raw-K ratio for the next layer, matching the paper pseudocode's operation order. `same_layer` computes the current ratio before fusion and is exposed as a separate option. Alpha is not silently clamped: it can exceed one. `clamp_alpha` is an explicit non-default configuration experiment. Epsilon stabilizes the denominator; Algorithm 2 lists epsilon but does not place it in the printed ratio. Float64 accumulation avoids half/float32 squaring overflow. `WCANumericalError` alone identifies recoverable nonfinite WCA arithmetic/data; the runtime does not catch arbitrary WCA `ValueError` and conceal shape, policy or call-order bugs as a dense fallback.

Every four one-based layers, CKSim averages cosine similarity across heads. The paper pseudocode's literal removal condition is `cosine < 0.12`, while its prose describes convergence. The default `paper_cosine_lt` preserves that condition. The separate `distance_lt` experiment uses `1 - cosine < threshold`. Removed tokens leave both candidate and selected sets and cannot re-enter the promotion loop. Available slots promote the highest remaining initial-deviation candidates.

Promotion requires more than K/V. A newly selected row may not have run the previous layer, so `runtime.py` restores its cached layer-input hidden state and fused residual from immutable snapshots. Continuing selected/dynamic rows carry their newly computed states. Restoring old context is part of this approximate reuse algorithm; it is not exact recomputation of omitted intervening layers.

For selected reusable queries, the default `updated_and_self` association policy permits causally preceding dynamic keys plus the query's own key. Other query rows retain full causal visibility. Contextual gates are calculated over the full causal key set before this final visibility mask; the mask does not retrospectively remove dynamic/fixed gate contributions. The paper's association prose does not define this exact self-key/gate-mask convention. `full_causal` is an explicit alternative association policy. Tests exercise the same shared position and visibility code used by native inference, not a second runtime attention formula.

## SLIDE and the actual native KV arena

`NativePagedKV` wraps the native tensor shape `[block, kv_head, block_offset, 2 * head_dim]` and the caller-owned block-table row. It uses actual strides and cross-checks native slot mappings. It never treats arbitrary global blocks as request-owned.

Figure 9 has two branches, both represented explicitly:

- **Write first / host load pending:** `promote_selected` stages raw selected K/V into a bounded, lazily allocated device sidecar. It publishes the new logical-token mapping only after the selected device write completes. Later baseline writes populate the original slots; fused selected updates retain the relocated mapping. Old slots become decode holes only after source loads and pins finish.
- **Load first / actual device baseline ready:** after `write_prefill` completes the baseline device writes, `write_selected_ready` updates exclusive, unpinned original slots. It allocates no sidecar, creates no hole, and rejects incomplete baseline writes, pending loads, duplicate selection, prior relocation or decode. A completed host `Future[bytes]` is **not** evidence of completed GPU baseline writes.

On CUDA, device completion means a recorded event has completed, not merely that a host enqueue returned. Shared or still-live slots cannot be overwritten; retired request generations reject late completions. Actual overlap is limited: WCA fusion needs the old cached K/V, and the runtime waits for that dependency before fusion. Staging raw selected K/V does not eliminate it.

Baseline writes, ready in-place selection and selected-row updates reserve writable slots atomically against concurrent attention pins. Generic `LayerSlotMap.promote_selected` also holds destination write reservations across asynchronous in-place completion, preventing new pins from observing partial data. Required writes to pinned slots are rejected before mutation; optional canonical mirrors are skipped when pinned. Reservations outlive device completion without holding the mapping mutex while waiting, so load callbacks can progress. This is a guarded request-local arena, not permission for arbitrary uncoordinated external tensor writes.

After prefill, decode first tries an eligible vacated physical native slot, otherwise binds the new slot supplied by vLLM. Decode records are also mirrored to their own canonical slots. Mirroring a new row does not restore old selected rows displaced by hole reuse: subsequent attention must always use the logical mapping. The implementation gathers through that map for every attention read.

The scheduler's native block table and reserved pool remain authoritative. This sidecar mapping **does not release native pool blocks, reduce reserved pool capacity, or demonstrate additional native admission capacity**. Sidecar allocations, dense gather buffers, and hidden/residual snapshots are additional costs that must be included in future memory/performance evaluation.

`NativePagedKV.page_metadata()` derives selected-token counts from the actual logical-to-physical mapping, aggregating shared/pinned/inflight guards page-wide. Here `dirty` means the paper's **selected-token presence**, not a claim that an SSD-backed version is stale. `spill_order()` offers eligible clean pages first, then dirty pages in descending selected-token count. This is a snapshot of policy candidates, **not** an eviction reservation: no mutable page backing/version tracking, GPU-page writeback, residency removal or pool freeing occurs. Immutable reusable snapshots are not relabeled dirty to simulate this missing integration.

`coalesce_slot_writes` combines adjacent device slot records before submitting a selected write; this is not coalesced dirty-page SSD writeback. `storage.py` separately provides immutable, checksummed RAM/SSD pages, bounded resident/inflight accounting, guarded eviction, atomic file publication, and restart validation. Filenames hash keys rather than embedding user-supplied request paths. A directory has one writer; stale locks and unfinished writes require explicit recovery.

The runtime persists K/V and layer-input state snapshots, and publishes a commit marker only when population is complete. It prefetches the next layer's K/V into host memory while the current layer runs. Deserialize/device transfer and the fusion dependency still wait; no dedicated pinned-memory asynchronous H2D pipeline is implemented. Population waits for individual immutable page spills. Store spill I/O holds its store mutex and can serialize other operations. The executor therefore does not establish complete load/write independence or multi-request throughput.

Host write counters represent payload bytes and bytes written to page files, including headers. They do not measure SSD flash writes, hardware write amplification, allocator overhead, or process RSS. The store has selected-count-aware eviction metadata, but immutable runtime snapshots use `selected_count=0`, making that actual snapshot path clean/LRU rather than physical dirty-page SLIDE eviction. The physical candidate ordering and immutable store are separate APIs. Mutable dirty-page spill, combined sequential overwrite, scalable prefetch and the paper's sustained SSD-budget behavior remain unimplemented or unvalidated.

## Fallback, measurement, and verification

A cache/profile/layout miss uses full prompt computation. Recoverable reuse failures, including unavailable capacity, corrupt snapshots or typed WCA numerical failure, retry a full prefill before sampling. Pending I/O is drained/retired and partial population is discarded. Invalid canonical paths additionally clear all layer profiles and retry plain CoPE, as described above. Invalid engine modes and untrained/incompatible artifacts fail at setup rather than enabling an implicit positional substitution. Ordinary programming/schema errors remain errors.

The benchmark requires profiles for execution and configures the same trained CoPE/LoRA model and profiles for both operations. It populates seeds, warms the engine, and records matched recompute/reuse queries. Setup/warmup records remain in raw output but are excluded from measured means. After each timed call, `collective_rpc("cacheslide_metrics")` obtains a receipt outside the timed region. Request ID, operation, and prompt length must match. Every measured reuse must report `cache_hit=true` and `fallback=false`; otherwise `reuse_validation_passed` is false and no latency ratio is reported. Stale/missing baseline receipts, invalid means, or unequal output lengths also suppress the ratio. A strict positional fallback is a correctness outcome, never evidence of acceleration. Valid runs report `mean(baseline_seconds) / mean(reuse_seconds)`; they do not average per-request ratios. A final worker RPC drains the CacheSlide store and releases its writer lock.

Blocking one-token generation measures offline prefill-plus-one-token latency. It includes call, scheduling, sampling, and result-return overhead and is not streaming TTFT. Longer generations measure offline generation latency. Exact generated-token agreement and optional token-ID multiset F1 are consistency checks, not task answer accuracy or dataset F1. Runtime receipts and logs establish whether reuse hit or fell back.

Portable tests include independent dense numerical references, causal/GQA gradients, actual small-model optimizer updates, full-context profile calibration, request-isolated selection, state restoration after promotion, decode continuity, cache corruption/restart behavior, asynchronous publication ordering, and native-source signature/fingerprint checks. GPU execution, representative trained-model quality, sustained SSD behavior, native pool savings, and large-model speedups remain unverified. Reduced computed token-layer counts alone do not establish any of those outcomes.
