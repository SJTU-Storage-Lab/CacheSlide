"""Opt-in, single-owner wait-work coordination; not a codec or native adapter.

No threads, I/O, model forwards, or resource allocation are started here. An
external worker executes an issued immutable work unit and reports completion
on the owner thread. HOST_READY is the only supported consumer tier. Backend
publication must precede successful completion; publication and lease lifetime
remain backend responsibilities, including validation of the consumer epoch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol


def _text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be a nonempty bounded identifier")


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """One existing immutable publishable unit, not merely identical text.

    For CacheSlide pages, source_key is runtime._key(plan, layer, kind), which
    includes the complete runtime/task/ordered-layout identity; source_generation
    is the store page generation. codec_identity includes its version/settings;
    representation includes its format version. No identity is inferred here.
    """

    source_key: str
    source_generation: str
    codec_identity: str
    representation: str
    tier: str = "HOST_READY"

    def __post_init__(self):
        for name in (
            "source_key",
            "source_generation",
            "codec_identity",
            "representation",
        ):
            _text(getattr(self, name), name)
        if type(self.tier) is not str or self.tier != "HOST_READY":
            raise ValueError("v1 supports HOST_READY only, not persisted/device KV")


@dataclass(frozen=True, slots=True)
class ConsumerEpoch:
    """An owner-supplied real request generation, not a predicted wake time."""

    engine_id: str
    request_id: str
    generation: int

    def __post_init__(self):
        _text(self.engine_id, "engine_id")
        _text(self.request_id, "request_id")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a nonnegative integer")


@dataclass(frozen=True, slots=True, eq=False)
class WaitTicket:
    """Only the exact instance issued by on_wait authorizes a live operation."""

    key: ArtifactKey
    consumer: ConsumerEpoch


@dataclass(frozen=True, slots=True, eq=False)
class WorkTicket:
    """Only the exact currently issued instance can drain the in-flight unit."""

    key: ArtifactKey


class ReadyLease(Protocol):
    """Bindings are plain immutable metadata, never blocking properties."""

    key: ArtifactKey
    consumer: ConsumerEpoch

    def close(self) -> None:
        """Idempotent nonblocking pin release; defer resource destruction.

        No I/O, eviction, waiting, or device synchronization is allowed here,
        including when the controller closes an incompatible returned lease.
        """


class ReadyBackend(Protocol):
    def try_pin_ready(
        self, key: ArtifactKey, consumer: ConsumerEpoch
    ) -> ReadyLease | None:
        """Atomically validate exact identity/epoch and pin, or return a miss.

        Must not load, checksum, evict, synchronize devices, run models, or wait
        for locks/futures. An unavailable metadata lock is a miss. A returned
        lease exposes its actual key and consumer binding, not merely an echo
        of unvalidated arguments. It remains valid until its caller closes it,
        including after queue cancellation/shutdown. On exceptions, the backend
        retains no new pin. The controller cannot inspect native ownership.
        """


@dataclass
class _Job:
    key: ArtifactKey
    state: str = "queued"
    consumers: set[WaitTicket] = field(default_factory=set)


class WaitQueue:
    """One bounded insertion-ordered queue and at most one external work unit.

    All methods require the creating thread. A newer wait for the same engine
    and request supersedes its previous ticket. The owner supplies authoritative
    generations; this object is not a global request-generation registry.
    Terminal job metadata is retained for deduplication while bounded by
    max_jobs; only terminal entries without consumers may be evicted. A failed
    job is not automatically retried. Evicted history may be admitted by a later
    explicit on_wait. Capacity rejection leaves existing demands unchanged.
    """

    def __init__(self, backend: ReadyBackend, *, max_jobs=128, max_consumers=512):
        for limit in (max_jobs, max_consumers):
            if type(limit) is not int or limit < 1:
                raise ValueError("metadata limits must be positive integers")
        if not callable(getattr(backend, "try_pin_ready", None)):
            raise TypeError("backend must implement try_pin_ready")
        self._backend = backend
        self._owner = threading.current_thread()
        self._max_jobs, self._max_consumers = max_jobs, max_consumers
        self._jobs: dict[ArtifactKey, _Job] = {}
        self._waits: dict[tuple[str, str], WaitTicket] = {}
        self._work: WorkTicket | None = None
        self._closed = False
        self._backend_errors = 0

    def _check_owner(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("wait queue operations require its owner thread")

    def _check_open(self):
        self._check_owner()
        if self._closed:
            raise RuntimeError("wait queue is closed")

    @staticmethod
    def _consumer_id(consumer):
        return consumer.engine_id, consumer.request_id

    def on_wait(self, consumer: ConsumerEpoch, key: ArtifactKey) -> WaitTicket:
        self._check_open()
        if type(consumer) is not ConsumerEpoch or type(key) is not ArtifactKey:
            raise TypeError("on_wait requires ConsumerEpoch and ArtifactKey")
        identity = self._consumer_id(consumer)
        previous = self._waits.get(identity)
        if previous and consumer.generation < previous.consumer.generation:
            raise ValueError("cannot replace a wait with an older request generation")
        if previous is None and len(self._waits) >= self._max_consumers:
            raise OverflowError("waiting-consumer metadata capacity exhausted")
        if key not in self._jobs and len(self._jobs) >= self._max_jobs:
            victim = next(
                (
                    k
                    for k, j in self._jobs.items()
                    if j.state in {"complete", "failed"} and not j.consumers
                ),
                None,
            )
            if victim is None:
                raise OverflowError("job metadata capacity exhausted")
            del self._jobs[victim]
        if previous is not None:
            if previous.key == key:
                self._jobs[key].consumers.remove(previous)
            else:
                self.cancel(previous)
        job = self._jobs.setdefault(key, _Job(key))
        ticket = WaitTicket(key, consumer)
        job.consumers.add(ticket)
        self._waits[identity] = ticket
        return ticket

    def cancel(self, ticket: WaitTicket) -> bool:
        self._check_owner()
        if type(ticket) is not WaitTicket:
            raise TypeError("expected an issued WaitTicket")
        identity = self._consumer_id(ticket.consumer)
        if self._waits.get(identity) is not ticket:
            return False
        del self._waits[identity]
        job = self._jobs[ticket.key]
        job.consumers.remove(ticket)
        if not job.consumers and job.state == "queued":
            del self._jobs[ticket.key]
        return True

    def on_wake(self, ticket: WaitTicket) -> ReadyLease | None:
        """Consume this wait once, then attempt only a nonblocking ready pin."""
        if not self.cancel(ticket):
            return None
        lease = None
        try:
            lease = self._backend.try_pin_ready(ticket.key, ticket.consumer)
            if lease is not None and (
                type(lease.key) is not ArtifactKey
                or type(lease.consumer) is not ConsumerEpoch
                or lease.key != ticket.key
                or lease.consumer != ticket.consumer
                or not callable(lease.close)
            ):
                raise ValueError("backend returned an incompatible ready lease")
            return lease
        except Exception:
            # Backend failure is a miss, never a reason to wait or retry here.
            self._backend_errors += 1
            if lease is not None:
                try:
                    lease.close()
                except Exception:
                    self._backend_errors += 1
            return None

    def begin_next(self, *, resource_grant: bool) -> WorkTicket | None:
        """Caller asserts permission; this bool does not reserve GPU resources."""
        self._check_open()
        if type(resource_grant) is not bool:
            raise TypeError("resource_grant must be an explicit bool")
        if not resource_grant or self._work is not None:
            return None
        for job in self._jobs.values():
            if job.state == "queued" and job.consumers:
                job.state = "running"
                self._work = WorkTicket(job.key)
                return self._work
        return None

    def complete(self, work: WorkTicket, *, success: bool) -> None:
        """Drain exactly the issued unit, even after cancellation or close.

        Success does not make data ready: the backend must already have safely
        published it. Late/duplicate/foreign completions cannot mutate any job.
        """
        self._check_owner()
        if type(success) is not bool:
            raise TypeError("success must be bool")
        if type(work) is not WorkTicket or self._work is not work:
            raise ValueError("completion is not the current issued work ticket")
        self._jobs[work.key].state = "complete" if success else "failed"
        self._work = None
        if self._closed:
            self._jobs.clear()

    def forget(self, key: ArtifactKey) -> bool:
        """Forget terminal history, e.g. after eviction, to permit explicit retry.

        This only drops scheduling metadata, never backend data or a live lease.
        Success history does not assert that the backend still has ready data.
        """
        self._check_open()
        if type(key) is not ArtifactKey:
            raise TypeError("expected ArtifactKey")
        job = self._jobs.get(key)
        if job is None:
            return False
        if job.consumers or job.state not in {"complete", "failed"}:
            raise ValueError("cannot forget active work or waiting consumers")
        del self._jobs[key]
        return True

    def close(self) -> bool:
        """Stop admission and detach waiters; return whether work has drained.

        Does not kill/wait for an external worker or close the shared backend.
        If False, the owner must still report complete before backend teardown.
        Already handed-off leases remain the consumer/backend's responsibility.
        """
        self._check_owner()
        self._closed = True
        for ticket in tuple(self._waits.values()):
            self.cancel(ticket)
        self._jobs = {
            key: job for key, job in self._jobs.items() if job.state == "running"
        }
        return self._work is None

    def stats(self) -> dict[str, int | bool]:
        self._check_owner()
        return {
            "retained_jobs": len(self._jobs),
            "waiting_consumers": len(self._waits),
            "queued_jobs": sum(j.state == "queued" for j in self._jobs.values()),
            "in_flight": self._work is not None,
            "closed": self._closed,
            "drained": self._work is None,
            "backend_errors": self._backend_errors,
        }
