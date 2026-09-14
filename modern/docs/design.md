# Implementation design and paper interpretation

This document describes executable behavior in `modern/`, including choices where the [CacheSlide paper](../../CacheSlide.pdf) leaves implementation details ambiguous. Contextual position encoding follows [CoPE, arXiv:2405.18719, section 4](https://arxiv.org/abs/2405.18719). The implementation includes operational interpretations and bounded reference paths; it does not claim access to unpublished author weights or reproduction of the paper's performance results.

## Package and engine boundary

The original vLLM 0.8.5 tree and the `cacheslide-vllm` package are separate. The package pins stable `vllm==0.29.0`; `compatibility.json` also fingerprints the audited source files at commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. A different version or modified audited file is a compatibility failure. The observed `0.29.1rc1.dev5` server build is therefore outside this contract, even though its version is newer.

`plugin.py` registers one new architecture. `model.py` subclasses native Llama attention/layer/model classes and retains their projections, fused RMSNorm/residual conventions, SwiGLU, checkpoint loader, and sampler. Trained adapter parameters are registered under `model.cacheslide_adapters`; the native loader accounts for these independently loaded parameters without expecting them in backbone shards. No existing native model class is globally patched.

`worker.py` validates the configuration, then `integration.py` wraps only that runner instance. A `ContextVar` carries an immutable `StepContext` into the model call and is restored afterward. Context includes request ID, prompt IDs, absolute query positions, and frozen request metadata. Request cleanup retires the associated CacheSlide state; preemption starts from a fresh full prompt. Decode metadata must remain identical and positions must advance consecutively.

The gate requires one scheduled request, one rank in every parallel dimension, eager V1, no CUDA graphs/compilation, no APC/chunked prefill/async scheduling, and the audited `FLASH_ATTN` native cache geometry. Native inference accepts float16/bfloat16, bias-free Llama or full-attention Mistral, without quantization, sliding windows, heterogeneous attention, multimodal input, speculative decoding, prompt logprobs, or external LoRA requests.

## Token identity and artifacts

`RequestPlan` is an exact partition of integer prompt token IDs. Reusable chunks are ordered by their original prompt order. Cache identity incorporates adapter/profile identity, namespace, task, the complete ordered fixed layout, and layer. Each fixed-layout entry includes chunk ID, content digest, and exact token length. Changing a fixed chunk or its preceding fixed context creates a miss. Changes to dynamic chunks alone do not change the fixed identity.

All dynamic tokens are mandatory. The final prompt token is also mandatory, because native V1 sampling indexes its original scheduled row. Selective execution returns the original output-row count; uncomputed prompt rows are placeholders and are never exposed as prompt logprobs.

Adapter artifacts bind learned tensors to exact local backbone-file SHA-256 hashes and model geometry. Loading verifies positive completed training steps, checksums, expected layer tensor keys/shapes, and finite weights. No remote model code or pickle checkpoint is executed. Profile artifacts separately bind canonical contextual positions to the trained adapter, layer, namespace/task key, and fixed layout. They use strict JSON plus safetensors and reject incompatible bounds, ordinals, layouts, or checksums.

## Trained contextual attention

`reference.py` implements a frozen Llama/Mistral backbone using local safetensors. It exposes the native projection tuple convention and fused residual layer interface so CPU integration tests can exercise the same selective runtime. Only attention adapter parameters are trainable: learned CoPE embeddings and low-rank updates to concatenated Q/K/V and attention output projections.

`training.py` minimizes causal next-token cross-entropy on explicit token sequences. It cycles through the supplied sequences, performs actual optimizer steps, checks finite loss/gradients, and exports only after the requested updates finish. A positive-step artifact is an operational training result, not proof of model quality or equivalence to the original RoPE backbone. No pretrained author CoPE weights are bundled.

For Q shaped `[query, query_head, head_dim]` and K/V shaped `[key, kv_head, head_dim]`, GQA repeats KV heads to match query heads. For each query/head:

```text
L[i,j] = dot(Q[i], K[j]) / sqrt(head_dim)
g[i,j] = sigmoid(L[i,j]) for allowed causal keys, else 0
p[i,j] = clamp(sum(g[i,k] for k >= j), 0, max_positions - 1)
E = learned matrix [head_dim, max_positions]
b[i,j] = interpolate(Q[i] @ E, floor(p[i,j]), ceil(p[i,j]))
A = softmax(mask_causal(L + b))
output = A @ V
```

Future keys are masked before sigmoid and reverse cumulative summation. Position bias remains query- and head-dependent. It is not a RoPE rotation and cannot be represented by a one-dimensional vector of shifted token offsets. Gradients pass through the gates, interpolation fractions, and learned embeddings.

The reference attention chunks the query axis to avoid allocating a full prompt attention matrix during ordinary inference. Returning full traces is an explicit calibration operation. Native contextual attention also uses query chunks, but materializes logical K/V gathered from the native arena; it is not an optimized fused CoPE kernel.

## CCPE calibration convention

The paper describes a task-conditioned encoding histogram but does not fully specify the artifact construction and lookup procedure. This implementation makes that procedure explicit:

1. Run the verified trained model over each complete calibration prompt.
2. At every layer, capture contextual positions from real Q/K logits with all causal dynamic-token gates present.
3. Extract fixed-query × fixed-key positions only after the full contextual computation. Retain source logits in memory long enough to verify the gate equation.
4. Replace absolute position metadata with compact ordinals in the ordered fixed-token sequence. This permits the same fixed chunks to appear after dynamic spans of different lengths.
5. Group by adapter, namespace/task, ordered fixed layout, and layer. Quantize each complete encoding pattern for histogram membership, select the most frequent joint pattern, break pattern ties lexicographically, and retain an actual representative trace's values.

Independent per-cell modes are deliberately not assembled into a never-observed encoding. Within the winning quantized pattern, the first observed sample supplies the stored continuous values. The default histogram bin width is an explicit implementation parameter, not a recovered author setting.

At runtime, canonical values replace only the fixed-query/fixed-key cells. Other cells use current CoPE positions. Lookup selects the requested fixed query ordinals and rejects unknown keys or shapes. Decode queries are new contextual queries and are not extrapolated from fixed profiles.

The full source trace is `heads × prompt_length²` per layer/sample; each canonical profile is `heads × fixed_length²`. Calibration checks the aggregate retained trace/profile budget before allocation, and profile loading checks tensor shapes and element totals before materialization. These budgets prevent accidental oversized artifacts; they do not turn this representation into a scalable long-context scheme. Exported profile artifacts omit raw source logits, Q/K tensors, request token IDs, and dynamic chunk metadata.

## Weighted cache adaptation

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

CoPE adds a score bias without rotating K. Therefore first-block pre-attention K is token-local and can be identical for repeated content despite changed preceding context. The runtime defaults to zero-based `calibration_layer=1`: it fully computes layers 0 and 1, initializes WCA from contextual K in layer 1, and performs sparse computation from layer 2 onward. This two-shallow-layer convention is an explicit engineering interpretation; it is not presented as identical to a literal one-layer initialization. Later selected rows run the actual projection, attention, MLP, and residual computation.

For each selected token, with epsilon `1e-8` by default:

```text
alpha_new = ||K_raw_new - K_cache||² / (||K_cache||² + epsilon)
K_fused = alpha_used * K_raw_new + (1 - alpha_used) * K_cache
V_fused = alpha_used * V_raw_new + (1 - alpha_used) * V_cache
```

One per-token alpha is shared by K and V. The default `weight_update=previous_layer` uses the prior stored alpha for fusion, then stores the current raw-K ratio for the next layer, matching the paper pseudocode's operation order. `same_layer` computes the current ratio before fusion and is exposed as a separate option. Alpha is not silently clamped: it can exceed one. `clamp_alpha` is an explicit non-default configuration experiment. Float64 weight/error accumulation avoids half/float32 squaring overflow; nonfinite fused results trigger validation/fallback instead of silent corruption.

Every four one-based layers, CKSim averages cosine similarity across heads. The paper pseudocode's literal removal condition is `cosine < 0.12`, while its prose describes convergence. The default `paper_cosine_lt` preserves that condition. The separate `distance_lt` experiment uses `1 - cosine < threshold`. Removed tokens leave both candidate and selected sets and cannot re-enter the promotion loop. Available slots promote the highest remaining initial-deviation candidates.

Promotion requires more than K/V. A newly selected row may not have run the previous layer, so `runtime.py` restores its cached layer-input hidden state and fused residual from immutable snapshots. Continuing selected/dynamic rows carry their newly computed states. Restoring old context is part of this approximate reuse algorithm; it is not exact recomputation of omitted intervening layers.

For selected reusable queries, the default `updated_and_self` association policy permits causally preceding dynamic keys plus the query's own key. Other query rows retain full causal visibility. Contextual gates are calculated over the full causal key set before this final visibility mask; the mask does not retrospectively remove dynamic/fixed gate contributions. `full_causal` is an explicit alternative association policy.

## SLIDE and the actual native KV arena

`NativePagedKV` wraps the native tensor shape `[block, kv_head, block_offset, 2 * head_dim]` and the caller-owned block-table row. It uses actual strides and cross-checks native slot mappings. It never treats arbitrary global blocks as request-owned.

Selected K/V may be staged into a bounded device sidecar while baseline host loading is pending. `LayerSlotMap` publishes the logical-token mapping only after the selected write completes. On CUDA, completion means a recorded device event has completed, not merely that a host enqueue returned. Old slots are reclaimable only after source loads and pins finish. Shared or still-live slots cannot be overwritten; retired request generations reject late completions.

After prefill, decode first tries an eligible vacated physical native slot, otherwise binds the new slot supplied by vLLM. Decode records are also mirrored to their own canonical slots. Mirroring a new row does not restore old selected rows displaced by hole reuse: subsequent attention must always use the logical mapping. The implementation gathers through that map for every attention read.

The scheduler's native block table and reserved pool remain authoritative. This sidecar mapping **does not release native pool blocks, reduce reserved pool capacity, or demonstrate additional native admission capacity**. Sidecar allocations, dense gather buffers, and hidden/residual snapshots are additional costs that must be included in future memory/performance evaluation.

`coalesce_slot_writes` combines adjacent physical slot records before submitting a selected write. `storage.py` provides immutable, checksummed RAM/SSD pages, bounded resident/inflight accounting, guarded eviction, atomic file publication, and restart validation. Filenames hash keys rather than embedding user-supplied request paths. A directory has one writer; stale locks and unfinished writes require explicit recovery.

The runtime persists K/V and layer-input state snapshots, and publishes a commit marker only when population is complete. It prefetches the next layer's K/V while the current layer runs. Host write counters represent payload bytes and bytes written to page files, including headers. They do not measure SSD flash writes, hardware write amplification, allocator overhead, or process RSS. The page store has selection-count-aware eviction metadata; current immutable runtime snapshots use `selected_count=0`, so selection-priority SSD policy performance is not established by these paths.

## Fallback, measurement, and verification

A cache/profile/layout miss uses full prompt computation. Recoverable reuse failures, including unavailable capacity or corrupt snapshots, retry a full prefill before sampling. Pending I/O is drained/retired and partial population is discarded. Invalid engine modes and untrained/incompatible artifacts fail at setup rather than enabling an implicit positional substitution.

The benchmark requires profiles for execution and uses the same trained CoPE/LoRA model and canonical profiles for both operations. It populates seeds, warms the engine, and records matched recompute/reuse queries. Setup/warmup records remain in raw output but are excluded from measured means. After each timed call, `collective_rpc("cacheslide_metrics")` obtains a receipt outside the timed region. Request ID, operation, and prompt length must match. Every measured reuse must report `cache_hit=true` and `fallback=false`; otherwise `reuse_validation_passed` is false and no latency ratio is reported. Stale/missing baseline receipts, invalid means, or unequal output lengths also suppress the ratio. Valid runs report `mean(baseline_seconds) / mean(reuse_seconds)`; they do not average per-request ratios. A final worker RPC drains the CacheSlide store and releases its writer lock.

Blocking one-token generation measures offline prefill-plus-one-token latency. It includes call, scheduling, sampling, and result-return overhead and is not streaming TTFT. Longer generations measure offline generation latency. Exact generated-token agreement and optional token-ID multiset F1 are consistency checks, not task answer accuracy or dataset F1. Runtime receipts and logs establish whether reuse hit or fell back.

Portable tests include independent dense numerical references, causal/GQA gradients, actual small-model optimizer updates, full-context profile calibration, request-isolated selection, state restoration after promotion, decode continuity, cache corruption/restart behavior, asynchronous publication ordering, and native-source signature/fingerprint checks. GPU execution, representative trained-model quality, sustained SSD behavior, native pool savings, and large-model speedups remain unverified. Reduced computed token-layer counts alone do not establish any of those outcomes.
