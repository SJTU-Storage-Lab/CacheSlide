# SGLang branch design

This branch targets SGLang **0.5.19**, commit
`0bcd822377da7b5718e674eaf9c870d349424dd1`. It keeps the audited vLLM main
baseline (`3f220b8`) separate. Engine sources are external dependencies, not
vendored trees. [Validation](sglang_validation.md) distinguishes source/CPU
evidence from pending native GPU execution.

## Shared algorithms, independent adapters

`cacheslide_core` owns contracts, trained CoPE/CCPE, WCA, slot-map state,
immutable storage, reference training and selective execution. The SGLang
adapter imports this core, not `cacheslide_vllm`. The vLLM adapter remains for
explicit cross-backend comparison; neither backend is a fallback for the other.

| SGLang module | Responsibility |
| --- | --- |
| `compat.py`, `compatibility.json` | Exact version, 30 source fingerprints, import identity and supported configuration |
| `plugin.py`, `integration.py` | Official hooks, parent/worker attestations, actual request/forward context, retirement and launch |
| `models/llama.py`, `backend.py` | Native projection/loading/logits conventions with shared contextual attention |
| `pool.py` | Actual split K/V NHD token buffers, request-generation ownership and per-layer SLIDE indirection |
| `receipts.py`, `benchmark.py` | Atomic terminal receipts, output/identity checks and paired warm-engine measurements |
| `workflow.py`, `smoke.py` | Isolated installation, real training/calibration stages and CPU-only fixture checks |

## Native lifecycle boundary

The official `sglang.srt.plugins` loader can log and continue after hook errors.
Registration is therefore insufficient: the adapter checks actual applied hooks
in the parent and child and validates worker attestations before generation.
The manifest binds the upstream hook machinery as well as model, pool, scheduler
and output paths. A source override used for inspection cannot bypass installed
source checks during native launch.

The worker boundary reads the actual `ScheduleBatch.reqs`; a request-local
context carries IDs, token plan, pool row and generation. The model checks the
final `ForwardBatch`, including copies made by the eager runner. Dummy profiling
has an explicit nonpersisting path. Contexts reset on errors and normal returns.
Retraction uses a fresh epoch and dense prompt-plus-generated-history replay,
not a new accelerated hit or reusable snapshot.

CacheSlide drains/retires its request state before native `release_kv_cache`
recycles tokens. Native allocation ownership stays with SGLang. Terminal atomic
JSON receipts bind run/request nonce, actual input, configuration and visible
output IDs. Missing, malformed, failed, stale or mismatched receipts cannot
authorize a benchmark ratio. Publication/validation failures are errors, not
silent success. Shutdown has bounded cleanup and does not depend on `atexit`
running after a worker is killed.

## Positions and correction

Defaults match the audited shared policy: `strict_contextual`, first-layer WCA
initialization (`calibration_layer=0`), correction fraction 0.26, prior-layer
raw fusion weight and the printed cosine-below-0.12 gate every four layers.
Zero first-layer deviation does not manufacture selected tokens.

Canonical fixed/fixed cells are substituted only after real full-context CoPE
gates are computed. Strict mode validates the complete hybrid key path; it does
not prove equality to current gates or semantic accuracy. An invalid path clears
all profiles and retries the entire prefill using plain trained CoPE before
sampling. Receipts explicitly report fallback and no cache hit.

`mixed_bias_override` permits the historical hybrid positional bias without
that validation; `calibration_layer=1` computes two shallow layers before WCA.
These are explicit approximate engineering variants, not literal defaults.
The shared [historical paper audit](paper_conformance.md) explains the remaining
histogram, selected-association, fusion and comparator interpretations; its
vLLM-specific validation claims do not transfer to SGLang.

## Native pool and actual I/O

`SGLangTokenKV` uses separate native K/V tensors in token/head/dimension layout.
It validates the scheduler's request row and allocation generation and never
rewrites the global request-to-token row to install layer-specific mappings.

After actual baseline device writes finish, selected updates can remain in
exclusive unpinned native slots. Pending loads instead relocate selected rows
into bounded sidecars; publication waits for device completion. Decode may
reuse safe vacated original slots before its new native slot. Loads, pins,
inflight writes and stale generations prevent unsafe reuse; decode publication
also waits for completed writes. CPU tests compare gathered per-layer K/V and
logits against the dense shared runtime through multiple decode steps.

These mappings do **not** release the native reserved token pool or increase
admission capacity. Physical page selected counts and clean-first/descending
dirty-count order are advisory metadata, not an eviction reservation. Runtime
snapshots remain immutable with selected count zero. Mutable GPU-page SSD
writeback, aggregated dirty-page overwrites and a fully asynchronous H2D pipeline
are not integrated. Host prefetch still has deserialization/device-transfer and
old-KV fusion dependencies. Sidecars, dense gathers and snapshots remain costs.

## Supported experiment and measurement

Initial support is one request/GPU/rank, eager local unquantized bias-free
Llama/full-attention Mistral with trained adapters. Native prefix/radix/HiCache,
overlap, chunked prefill, graphs/compilation, parallelism, speculative decoding,
sliding windows, multimodal input and external LoRA are rejected.

The workflow trains and calibrates in separate processes, populates seeds and
warms one native engine, then records matched recompute/reuse pairs. It measures
blocking whole-generation latency, not streaming TTFT or concurrent QPS.
Both paths use the same trained CoPE/adapter configuration. Setup/warmup is
excluded, and fallback or invalid receipts/output suppress the ratio and produce
`validation_failed` with exit code 2. Token consistency is not task answer F1;
managed file bytes are not hardware SSD WAF. No fused contextual kernel or
paper-level performance reproduction is claimed.
