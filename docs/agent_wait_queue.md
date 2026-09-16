# Agent-wait preparation: an opt-in scheduling contract

Status: **CPU-tested control-plane prototype, not native compression or a
measured speedup.** Nothing imports or enables this coordinator in the model
runtime. CCPE, WCA, SLIDE, cache storage and the original benchmark are unchanged.

The first version has three mechanisms:

1. Admit existing immutable artifacts when an agent actually starts waiting.
2. Deduplicate compatible demand in one FIFO queue, with one work unit in flight.
3. At wakeup, pin an already-ready artifact or return a miss for the original path.

There is no deadline predictor, weighted priority formula, speculative model
forward, extra cache writer or automatic GPU allocation. An agent waiting does
not imply that a shared GPU is idle.

## Ownership and execution

`wait_queue.py` uses only the Python standard library. All coordinator events
belong to its creating thread: use the serving orchestrator's existing event
loop to deliver wait, wake, cancellation and completion events. Cross-thread
calls are rejected. This serializes controller state without a contended wake
lock; it does **not** provide native concurrent model execution.

An external, explicitly assigned worker performs preparation. `begin_next`
requires the caller's resource-grant assertion and returns at most one issued
work ticket. The assertion is not a GPU reservation: resource ownership,
memory/I/O budgets and foreground interference remain the caller's responsibility.
Run blocking preparation outside the event-loop thread, then report completion
back to that thread. Never synchronously run the compressor from `on_wake`.

A work unit is one complete, independently publishable artifact. Cancellation
removes a consumer, not another consumer's shared work. Already executing work
drains to its safe boundary; the coordinator cannot cancel a running CUDA kernel.
Count work completed after all consumers wake as wasted preparation. Shutdown
stops new submissions and explicitly reports whether the last unit has drained;
it neither waits for nor kills external workers.

## Exact identity, not text similarity

Use the existing runtime's full source key, including its snapshot kind, without
rebuilding or weakening it. That identity includes the actual runtime/model,
adapter, profile, position policy, task, namespace and ordered token layout.
Add immutable source-page generation, codec/config identity and representation
version to `ArtifactKey`. Different tasks containing identical text are not
automatically eligible for sharing.

Each consumer binds its engine instance, request ID and request generation.
The coordinator additionally issues a distinct ticket for each wait. Old wake,
cancel and completion events must not alter a replacement wait or a different
source generation. Queue metadata is bounded; overload is explicit and must
use the original foreground path rather than growing without limit.
Each engine/request has one artifact demand at a time; another `on_wait`
replaces that demand. This is not a per-layer batch/prefetch API. A future
multi-page adapter must define its complete publishable unit deliberately.

Completed/failed scheduling history is bounded and is not proof that data is
still resident. After backend eviction or a failed attempt, `forget(key)` can
discard terminal history only when no consumer remains; a later explicit
`on_wait` may retry. There is no automatic retry loop or hidden data deletion.

## HOST_READY is deliberately the only initial tier

This version rejects `PERSISTED` and `DEVICE_READY`. A file on disk is not a
ready host artifact, and host readiness does not hide transfer or model prefill.
Request-owned device slots require an additional target arena/device generation
and a completed device-write record; no such native ready-handoff integration
is provided by this module.

The backend's `try_pin_ready(key, consumer)` contract must perform only a
nonblocking lookup and atomic lifetime pin. Metadata contention, eviction,
incomplete preparation or incompatible identity is a miss. It must not call
`read`, `submit_load`, unfinished `Future.result`, checksumming, decompression,
device synchronization or model execution. The backend validates all bytes and
geometry before ready publication. A returned lease remains valid until the
consumer closes it, including if the controller is cancelled or closed.
Lease identity fields are plain metadata; lease close must release only a pin
without blocking. Any expensive destruction belongs to the backend worker.

The existing `TieredPageStore.pin()` calls blocking `read()` and cannot be used
as this fast path. The controller does not own stored bytes, eviction or leases,
and does not open a second writer on an existing cache directory.

## What this does not implement

The current runtime stores uncompressed safetensors snapshots. Neither that
serializer nor cache population is a compression codec. This change supplies
the scheduling/lifecycle contract and a test-only backend, not an invented
replacement codec or a production-ready storage adapter.

The actual budget-window baseline and intended compression backend still need
to be identified before a matched performance comparison. The prototype is
opt-in and is not wired into either engine's benchmark launcher. No memory
savings, native QPS, task-quality preservation or B300 acceleration is claimed.

## Run the correctness checks

With the repository's test dependencies installed, from its root:

```bash
python -m pytest -q tests/test_wait_queue.py tests/test_package_boundaries.py
```

These checks need no weights, GPU, model download or serving engine. They cover
FIFO/deduplication, independent consumers, stale tickets, failure/eviction,
capacity, explicit resource grants, shutdown and ready-pin lifetime. Test fake
bytes and readiness events are not a compression or GPU benchmark.

For a later performance experiment, compare no preparation, the actual unchanged
budget-window implementation, this FIFO policy, and the same policy without
deduplication. Use one codec and quality setting, the same resource limits and
observable agent events. Include misses, cancellations, source recovery, retained
originals, temporary buffers, transfer and foreground interference in task
p50/p95 and memory accounting. A ready-only hit is not the whole workload.
