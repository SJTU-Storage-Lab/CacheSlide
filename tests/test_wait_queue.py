"""CPU state-machine contracts only: no codec, native engine, or speed claim."""

from dataclasses import FrozenInstanceError, replace
from threading import Thread

import pytest

from cacheslide_core.contracts import RequestPlan
from cacheslide_core.wait_queue import (
    ArtifactKey,
    ConsumerEpoch,
    WaitQueue,
    WaitTicket,
    WorkTicket,
)


def key(task="task", generation="page-1", **changes):
    plan = RequestPlan.parse(
        {
            "version": 1,
            "operation": "reuse",
            "namespace": "test",
            "task_id": task,
            "chunks": [{"id": "chunk", "role": "reuse", "start": 0, "end": 2}],
        },
        [10, 20],
    )
    return replace(
        ArtifactKey(
            plan.cache_key("actual-runtime-identity", 0) + "/kv",
            generation,
            "test-only-codec-v1",
            "test-only-host-bytes-v1",
        ),
        **changes,
    )


class Lease:
    def __init__(self, backend, artifact, consumer):
        self.backend, self.key, self.consumer = backend, artifact, consumer
        self.closed = False
        backend.pins += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.pins -= 1
            self.backend.releases += 1


class Backend:
    """Explicit test publication and authoritative host consumer generations."""

    def __init__(self):
        self.ready, self.epochs, self.calls = set(), {}, []
        self.pins = self.releases = 0
        self.bad_key = self.bad_consumer = None
        self.error = False
        self.metadata_busy = False

    def try_pin_ready(self, artifact, consumer):
        self.calls.append((artifact, consumer))
        if self.error:
            raise RuntimeError("controlled backend failure")
        if (
            self.metadata_busy
            or artifact not in self.ready
            or self.epochs.get((consumer.engine_id, consumer.request_id))
            != consumer.generation
        ):
            return None
        return Lease(self, self.bad_key or artifact, self.bad_consumer or consumer)


def consumer(backend, name="request", generation=0, engine="engine"):
    result = ConsumerEpoch(engine, name, generation)
    backend.epochs[(engine, name)] = generation
    return result


def test_fifo_dedup_pending_inflight_and_completed():
    backend = Backend()
    queue = WaitQueue(backend)
    a, b = key("A"), key("B")
    first = queue.on_wait(consumer(backend, "a1"), a)
    queue.on_wait(consumer(backend, "b"), b)
    queue.on_wait(consumer(backend, "a2"), a)
    work = queue.begin_next(resource_grant=True)
    assert work.key == a
    queue.on_wait(consumer(backend, "a3"), a)
    assert queue.begin_next(resource_grant=True) is None
    backend.ready.add(a)
    queue.complete(work, success=True)
    queue.on_wait(consumer(backend, "a4"), a)
    assert queue.stats()["retained_jobs"] == 2
    assert queue.begin_next(resource_grant=True).key == b
    lease = queue.on_wake(first)
    assert lease.key == a and backend.pins == 1
    lease.close()
    lease.close()
    assert backend.pins == 0 and backend.releases == 1


def test_same_consumer_same_key_replacement_keeps_fifo_position():
    backend = Backend()
    queue = WaitQueue(backend)
    epoch = consumer(backend)
    old = queue.on_wait(epoch, key("A"))
    queue.on_wait(consumer(backend, "second"), key("B"))
    current = queue.on_wait(epoch, key("A"))
    assert current is not old
    assert queue.on_wake(old) is None
    assert not queue.cancel(old)
    assert backend.calls == []
    assert queue.begin_next(resource_grant=True).key == key("A")


def test_cancel_one_shared_consumer_and_all_queued_consumers():
    backend = Backend()
    queue = WaitQueue(backend)
    first = queue.on_wait(consumer(backend, "one"), key())
    second = queue.on_wait(consumer(backend, "two"), key())
    assert queue.cancel(first)
    assert queue.stats()["queued_jobs"] == 1
    assert queue.cancel(second)
    assert not queue.cancel(second)
    assert queue.stats()["retained_jobs"] == 0
    assert queue.begin_next(resource_grant=True) is None


def test_cancel_inflight_drains_before_next_job():
    backend = Backend()
    queue = WaitQueue(backend)
    first = queue.on_wait(consumer(backend), key("A"))
    work = queue.begin_next(resource_grant=True)
    queue.cancel(first)
    queue.on_wait(consumer(backend, "second"), key("B"))
    assert queue.begin_next(resource_grant=True) is None
    queue.complete(work, success=False)
    assert queue.begin_next(resource_grant=True).key == key("B")


def test_revised_demand_and_late_completion_do_not_pollute_new_wait():
    backend = Backend()
    queue = WaitQueue(backend)
    epoch = consumer(backend)
    old = queue.on_wait(epoch, key(generation="old-page"))
    work = queue.begin_next(resource_grant=True)
    new = queue.on_wait(epoch, key(generation="new-page"))
    backend.ready.add(work.key)
    queue.complete(work, success=True)
    assert queue.on_wake(old) is None
    assert queue.on_wake(new) is None
    assert backend.calls == [(new.key, epoch)]
    assert backend.pins == 0


def test_generation_replacement_rejects_old_waits_and_old_native_epoch():
    backend = Backend()
    queue = WaitQueue(backend)
    old_epoch = consumer(backend, generation=1)
    old = queue.on_wait(old_epoch, key())
    new_epoch = consumer(backend, generation=2)
    current = queue.on_wait(new_epoch, key())
    with pytest.raises(ValueError, match="older"):
        queue.on_wait(old_epoch, key())
    assert queue.on_wake(old) is None
    backend.ready.add(key())
    backend.epochs[(new_epoch.engine_id, new_epoch.request_id)] = 3
    assert queue.on_wake(current) is None
    assert backend.pins == 0


def test_tickets_are_immutable_and_authorized_by_instance_not_equality():
    backend = Backend()
    queue = WaitQueue(backend)
    ticket = queue.on_wait(consumer(backend), key())
    forged = WaitTicket(ticket.key, ticket.consumer)
    assert ticket != forged
    assert not queue.cancel(forged)
    assert queue.on_wake(forged) is None
    with pytest.raises(FrozenInstanceError):
        ticket.key = key("different")
    work = queue.begin_next(resource_grant=True)
    with pytest.raises(ValueError, match="issued"):
        queue.complete(WorkTicket(work.key), success=True)
    foreign = WaitQueue(backend)
    assert foreign.on_wake(ticket) is None
    with pytest.raises(ValueError, match="issued"):
        foreign.complete(work, success=True)
    queue.complete(work, success=True)
    with pytest.raises(ValueError, match="issued"):
        queue.complete(work, success=True)


def test_success_is_not_backend_readiness_and_failed_work_does_not_retry():
    backend = Backend()
    queue = WaitQueue(backend)
    first = queue.on_wait(consumer(backend), key())
    work = queue.begin_next(resource_grant=True)
    queue.complete(work, success=True)  # No test publication: still a miss.
    assert queue.on_wake(first) is None
    second = queue.on_wait(consumer(backend, "second"), key())
    assert queue.begin_next(resource_grant=True) is None
    assert queue.on_wake(second) is None
    assert queue.forget(key())
    third = queue.on_wait(consumer(backend, "third"), key())
    new_work = queue.begin_next(resource_grant=True)
    with pytest.raises(ValueError, match="issued"):
        queue.complete(work, success=True)
    queue.complete(new_work, success=False)
    assert queue.on_wake(third) is None
    fourth = queue.on_wait(consumer(backend, "fourth"), key())
    assert queue.begin_next(resource_grant=True) is None
    assert queue.on_wake(fourth) is None


def test_backend_eviction_requires_explicit_terminal_forget_for_retry():
    backend = Backend()
    queue = WaitQueue(backend)
    first = queue.on_wait(consumer(backend), key())
    work = queue.begin_next(resource_grant=True)
    backend.ready.add(key())
    queue.complete(work, success=True)
    lease = queue.on_wake(first)
    backend.ready.clear()
    assert queue.forget(key())  # Does not invalidate the already handed-off pin.
    assert backend.pins == 1
    lease.close()
    queue.on_wait(consumer(backend, "retry"), key())
    assert queue.begin_next(resource_grant=True) is not None


@pytest.mark.parametrize("state", ["queued", "running", "complete", "failed"])
def test_forget_refuses_waiters_or_inflight_work(state):
    backend = Backend()
    queue = WaitQueue(backend)
    ticket = queue.on_wait(consumer(backend), key())
    if state != "queued":
        work = queue.begin_next(resource_grant=True)
        if state != "running":
            queue.complete(work, success=state == "complete")
    with pytest.raises(ValueError, match="active"):
        queue.forget(key())
    queue.cancel(ticket)
    if state == "running":
        with pytest.raises(ValueError, match="active"):
            queue.forget(key())
    elif state == "queued":
        assert not queue.forget(key())
    else:
        assert queue.forget(key())


def test_metadata_limits_reject_without_changing_previous_demand():
    backend = Backend()
    queue = WaitQueue(backend, max_jobs=1, max_consumers=1)
    epoch = consumer(backend)
    first = queue.on_wait(epoch, key())
    with pytest.raises(OverflowError, match="consumer"):
        queue.on_wait(consumer(backend, "other"), key())
    with pytest.raises(OverflowError, match="job"):
        queue.on_wait(epoch, key("other"))
    assert queue.cancel(first)
    assert queue.stats()["waiting_consumers"] == 0


def test_terminal_metadata_is_bounded_and_eviction_keeps_no_payloads():
    backend = Backend()
    queue = WaitQueue(backend, max_jobs=1)
    for number in range(20):
        artifact = key(str(number))
        ticket = queue.on_wait(consumer(backend, str(number)), artifact)
        work = queue.begin_next(resource_grant=True)
        queue.complete(work, success=False)
        assert queue.on_wake(ticket) is None
        assert queue.stats()["retained_jobs"] == 1


def test_explicit_resource_grant_required_and_no_automatic_start():
    backend = Backend()
    queue = WaitQueue(backend)
    queue.on_wait(consumer(backend), key())
    assert backend.calls == []
    assert not queue.stats()["in_flight"]
    with pytest.raises(TypeError):
        queue.begin_next()
    for value in (None, 1, "yes"):
        with pytest.raises(TypeError):
            queue.begin_next(resource_grant=value)
    assert queue.begin_next(resource_grant=False) is None
    assert queue.begin_next(resource_grant=True) is not None


@pytest.mark.parametrize("mutation", ["task", "generation", "codec", "representation"])
def test_exact_compatibility_prevents_dedup(mutation):
    backend = Backend()
    queue = WaitQueue(backend)
    different = {
        "task": key("different-task"),  # Same tokens; real RequestPlan key differs.
        "generation": key(generation="different-page"),
        "codec": key(codec_identity="other-codec-settings"),
        "representation": key(representation="other-format"),
    }[mutation]
    queue.on_wait(consumer(backend, "first"), key())
    queue.on_wait(consumer(backend, "second"), different)
    assert queue.stats()["queued_jobs"] == 2


@pytest.mark.parametrize("tier", ["DEVICE_READY", "PERSISTED", "", None, True])
def test_only_host_ready_is_supported(tier):
    with pytest.raises(ValueError, match="HOST_READY"):
        key(tier=tier)


@pytest.mark.parametrize("wrong_binding", ["key", "consumer"])
def test_bad_backend_lease_binding_is_a_miss_without_leaked_pin(wrong_binding):
    backend = Backend()
    queue = WaitQueue(backend)
    epoch = consumer(backend)
    ticket = queue.on_wait(epoch, key())
    backend.ready.add(key())
    if wrong_binding == "key":
        backend.bad_key = key(generation="stale-page")
    else:
        backend.bad_consumer = replace(epoch, generation=1)
    assert queue.on_wake(ticket) is None
    assert backend.pins == 0 and backend.releases == 1
    assert queue.stats()["backend_errors"] == 1


@pytest.mark.parametrize("reason", ["error", "metadata_busy", "not_ready"])
def test_wake_always_terminates_demand_even_when_backend_misses(reason):
    backend = Backend()
    queue = WaitQueue(backend)
    ticket = queue.on_wait(consumer(backend), key())
    if reason != "not_ready":
        setattr(backend, reason, True)
        backend.ready.add(key())
    assert queue.on_wake(ticket) is None
    assert queue.on_wake(ticket) is None
    assert len(backend.calls) == 1
    assert queue.begin_next(resource_grant=True) is None


def test_shutdown_detaches_waits_but_requires_external_inflight_drain():
    backend = Backend()
    queue = WaitQueue(backend)
    first = queue.on_wait(consumer(backend), key())
    work = queue.begin_next(resource_grant=True)
    second = queue.on_wait(consumer(backend, "second"), key("other"))
    assert not queue.close()
    assert not queue.close()
    assert queue.stats()["waiting_consumers"] == 0
    assert queue.on_wake(first) is None and queue.on_wake(second) is None
    with pytest.raises(RuntimeError, match="closed"):
        queue.on_wait(consumer(backend), key())
    with pytest.raises(RuntimeError, match="closed"):
        queue.begin_next(resource_grant=True)
    queue.complete(work, success=False)
    assert queue.close()
    assert queue.stats()["retained_jobs"] == 0


def test_shutdown_never_releases_consumer_owned_lease():
    backend = Backend()
    queue = WaitQueue(backend)
    backend.ready.add(key())
    ticket = queue.on_wait(consumer(backend), key())
    lease = queue.on_wake(ticket)
    assert queue.close()
    assert backend.pins == 1
    lease.close()
    assert backend.pins == 0


@pytest.mark.parametrize(
    "method",
    [
        "on_wait",
        "on_wake",
        "cancel",
        "begin_next",
        "complete",
        "forget",
        "close",
        "stats",
    ],
)
def test_every_public_operation_rejects_nonowner_threads(method):
    backend = Backend()
    queue = WaitQueue(backend)
    epoch = consumer(backend)
    ticket = queue.on_wait(epoch, key())
    work = queue.begin_next(resource_grant=True)
    calls = {
        "on_wait": lambda: queue.on_wait(epoch, key()),
        "on_wake": lambda: queue.on_wake(ticket),
        "cancel": lambda: queue.cancel(ticket),
        "begin_next": lambda: queue.begin_next(resource_grant=True),
        "complete": lambda: queue.complete(work, success=True),
        "forget": lambda: queue.forget(key()),
        "close": queue.close,
        "stats": queue.stats,
    }
    failures = []

    def nonowner():
        try:
            calls[method]()
        except RuntimeError as error:
            failures.append(str(error))

    thread = Thread(target=nonowner)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert failures == ["wait queue operations require its owner thread"]
    assert queue.stats()["waiting_consumers"] == 1
    assert queue.stats()["in_flight"]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_metadata_limits(limit):
    with pytest.raises(ValueError):
        WaitQueue(Backend(), max_jobs=limit)
    with pytest.raises(ValueError):
        WaitQueue(Backend(), max_consumers=limit)


@pytest.mark.parametrize("generation", [-1, True, "1", 1.5])
def test_invalid_consumer_generation(generation):
    with pytest.raises(ValueError):
        ConsumerEpoch("engine", "request", generation)


@pytest.mark.parametrize("identifier", ["", "a" * 1025, "bad\nname", None])
def test_invalid_identifiers(identifier):
    with pytest.raises(ValueError):
        key(source_generation=identifier)
    with pytest.raises(ValueError):
        ConsumerEpoch(identifier, "request", 0)
