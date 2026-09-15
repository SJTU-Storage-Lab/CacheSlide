import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cacheslide_core.storage import (
    CacheCapacityError,
    CacheIntegrityError,
    StaleCompletionError,
    TieredPageStore,
)


def make_store(tmp_path, cpu=16, disk=100_000, **kwargs):
    return TieredPageStore(
        tmp_path / "pages", cpu_budget_bytes=cpu, disk_budget_bytes=disk, **kwargs
    )


def test_immutable_namespaced_pages_survive_restart_with_safe_filenames(tmp_path):
    key = "model/adapter:../../request?layer=2"
    with make_store(tmp_path) as store:
        store.put(key, b"data", selected_count=2)
        store.put(key, b"data", selected_count=2)
        with pytest.raises(ValueError):
            store.put(key, b"changed", selected_count=2)
        store.submit_spill(key).result(timeout=3)
        assert store.read(key) == b"data"
        stats = store.stats()
        assert stats["payload_write_bytes"] == 4
        assert stats["physical_host_write_bytes"] > 4
        assert len(list((tmp_path / "pages").glob("*.page"))) == 1
    with make_store(tmp_path) as reopened:
        assert reopened.stats()["live_cpu_bytes"] == 0
        assert reopened.read(key) == b"data"
        assert reopened.stats()["inflight_cpu_reserved_bytes"] == 0


def test_clean_first_then_descending_selected_count_with_lru_tie(tmp_path):
    with make_store(tmp_path, cpu=4) as store:
        store.put("dirty-two-old", b"a", selected_count=2)
        store.put("clean", b"b")
        store.put("dirty-five", b"c", selected_count=5)
        store.put("dirty-two-new", b"d", selected_count=2)
        assert store.evict() == ["clean"]
        assert store.evict() == ["dirty-five"]
        assert store.evict() == ["dirty-two-old"]
        assert store.evict() == ["dirty-two-new"]
        assert store.stats()["live_cpu_bytes"] == 0


def test_pinned_capacity_raises_without_discarding_data(tmp_path):
    with make_store(tmp_path, cpu=4) as store:
        store.put("a", b"aaaa")
        with store.pin("a") as data:
            assert data == b"aaaa"
            with pytest.raises(CacheCapacityError):
                store.put("b", b"bbbb")
            with pytest.raises(CacheCapacityError):
                store.evict()
            with pytest.raises(CacheCapacityError):
                store.delete("a")
        assert store.read("a") == b"aaaa"


def test_clean_unbacked_page_is_preserved_when_disk_is_full(tmp_path):
    with make_store(tmp_path, cpu=4, disk=1) as store:
        store.put("a", b"aaaa")
        with pytest.raises(CacheCapacityError):
            store.put("b", b"bbbb")
        assert store.read("a") == b"aaaa"
        assert store.stats()["live_cpu_bytes"] == 4
        assert store.stats()["live_disk_bytes"] == 0
        assert not list((tmp_path / "pages").glob("*.page"))


def test_checksum_corruption_is_rejected_before_loading_or_eviction(tmp_path):
    with make_store(tmp_path) as store:
        store.put("a", b"original")
        store.submit_spill("a").result(timeout=3)
        path = next((tmp_path / "pages").glob("*.page"))
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        # A clean page cannot be dropped when its backing copy is corrupt.
        with pytest.raises(CacheIntegrityError):
            store.evict()
        assert store.read("a") == b"original"
    with pytest.raises(CacheIntegrityError):
        make_store(tmp_path)


def test_async_load_reservations_enforce_hard_capacity(tmp_path, monkeypatch):
    with make_store(tmp_path, cpu=8, max_workers=2) as store:
        store.put("a", b"aaaa")
        store.put("b", b"bbbb")
        store.evict()
        store.evict()
        entered, release = threading.Event(), threading.Event()
        read = store._read_disk

        def blocked_read(page):
            entered.set()
            assert release.wait(3)
            return read(page)

        monkeypatch.setattr(store, "_read_disk", blocked_read)
        first = store.submit_load("a")
        assert entered.wait(3)
        second = store.submit_load("b")
        try:
            assert store.submit_load("a") is first
            assert store.stats()["inflight_cpu_reserved_bytes"] == 8
            with pytest.raises(CacheCapacityError):
                store.put("c", b"cccc")
            with pytest.raises(CacheCapacityError):
                store.delete("a")
        finally:
            release.set()
        assert first.result(timeout=3) == b"aaaa"
        assert second.result(timeout=3) == b"bbbb"
        stats = store.stats()
        assert stats["live_cpu_bytes"] == 8
        assert stats["inflight_cpu_reserved_bytes"] == 0


def test_failed_load_releases_reservation_and_preserves_backing(tmp_path, monkeypatch):
    with make_store(tmp_path, cpu=4) as store:
        store.put("a", b"aaaa")
        store.evict()
        read = store._read_disk
        monkeypatch.setattr(
            store, "_read_disk", lambda _: (_ for _ in ()).throw(OSError("injected"))
        )
        with pytest.raises(OSError):
            store.submit_load("a").result(timeout=3)
        assert store.stats()["inflight_cpu_reserved_bytes"] == 0
        assert store.stats()["live_disk_bytes"] > 0
        monkeypatch.setattr(store, "_read_disk", read)
        assert store.read("a") == b"aaaa"


def test_failed_write_preserves_unbacked_payload_and_releases_budget(
    tmp_path, monkeypatch
):
    with make_store(tmp_path, cpu=4) as store:
        store.put("a", b"aaaa")
        monkeypatch.setattr(
            store, "_write_page", lambda *_: (_ for _ in ()).throw(OSError("injected"))
        )
        with pytest.raises(OSError):
            store.submit_spill("a").result(timeout=3)
        assert store.stats()["inflight_disk_reserved_bytes"] == 0
        assert store.stats()["inflight_pages"] == 0
        assert store.read("a") == b"aaaa"


def test_refuses_nonempty_unowned_root_and_concurrent_writer(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "user.txt").write_text("keep")
    with pytest.raises(ValueError):
        TieredPageStore(foreign, cpu_budget_bytes=8, disk_budget_bytes=1000)
    assert (foreign / "user.txt").read_text() == "keep"
    with make_store(tmp_path):
        with pytest.raises(RuntimeError):
            make_store(tmp_path)


def test_concurrent_put_read_preserves_immutable_identity(tmp_path):
    with make_store(tmp_path, cpu=64) as store:

        def use_page(index):
            key = f"model:layer:{index % 4}"
            payload = bytes([index % 4]) * 4
            store.put(key, payload)
            return store.read(key)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(use_page, range(40)))
        assert all(payload == bytes([i % 4]) * 4 for i, payload in enumerate(results))
        assert store.stats()["pages"] == 4


def test_partial_disk_write_never_publishes_a_page(tmp_path, monkeypatch):
    import os

    with make_store(tmp_path, cpu=4) as store:
        store.put("a", b"aaaa")
        original_write = os.write
        writes = 0

        def fail_payload_write(fd, data):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("injected interrupted payload write")
            return original_write(fd, data)

        with monkeypatch.context() as patch:
            patch.setattr(os, "write", fail_payload_write)
            with pytest.raises(OSError):
                store.submit_spill("a").result(timeout=3)
        assert store.read("a") == b"aaaa"
        assert store.stats()["physical_host_write_bytes"] > 0
        assert store.stats()["payload_write_bytes"] == 0
        assert store.stats()["live_disk_bytes"] == 0
        assert store.stats()["inflight_disk_reserved_bytes"] == 0
        assert not list(store.root.glob("*.page"))
        assert not list(store.root.glob(".pending-*"))


def test_retired_load_cannot_publish_and_reservations_are_released(
    tmp_path, monkeypatch
):
    with make_store(tmp_path, cpu=4) as store:
        store.put("a", b"aaaa")
        store.evict()
        entered, release, closing = (threading.Event() for _ in range(3))
        original_read = store._read_disk
        original_shutdown = store._executor.shutdown

        def blocked_read(page):
            entered.set()
            assert release.wait(3)
            return original_read(page)

        def observed_shutdown(*args, **kwargs):
            closing.set()
            return original_shutdown(*args, **kwargs)

        monkeypatch.setattr(store, "_read_disk", blocked_read)
        monkeypatch.setattr(store._executor, "shutdown", observed_shutdown)
        loading = store.submit_load("a")
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=1) as pool:
            closed = pool.submit(store.close, cancel_pending=True)
            try:
                assert closing.wait(3)
            finally:
                release.set()
            closed.result(timeout=3)
        with pytest.raises(StaleCompletionError):
            loading.result(timeout=3)
        assert store.stats()["live_cpu_bytes"] == 0
        assert store.stats()["inflight_cpu_reserved_bytes"] == 0


def test_future_callbacks_observe_finished_capacity_bookkeeping(tmp_path):
    with make_store(tmp_path, cpu=4) as store:
        store.put("a", b"aaaa")
        store.evict()
        observed = []
        callback_finished = threading.Event()

        def observe(_):
            observed.append(store.stats())
            callback_finished.set()

        loading = store.submit_load("a")
        loading.add_done_callback(observe)
        loading.result(timeout=3)
        assert callback_finished.wait(3)
        assert observed[0]["live_cpu_bytes"] == 4
        assert observed[0]["inflight_cpu_reserved_bytes"] == 0
        assert observed[0]["inflight_pages"] == 0


def test_catalog_preflight_does_not_load_pages(tmp_path, monkeypatch):
    with make_store(tmp_path) as store:
        store.put("model:layer:1", b"data", selected_count=2)
        store.evict()

        def forbidden_read(_):
            raise AssertionError("catalog lookup must not load a payload")

        monkeypatch.setattr(store, "_read_disk", forbidden_read)
        before = store.stats()
        assert store.contains("model:layer:1")
        assert not store.contains("model:layer:2")
        assert store.keys("model:") == ("model:layer:1",)
        assert store.keys("other:") == ()
        meta = store.metadata("model:layer:1")
        assert meta["size"] == 4
        assert meta["selected_count"] == 2
        assert meta["resident"] is False and meta["backed"] is True
        assert meta["pins"] == 0
        assert isinstance(meta["generation"], str)
        assert store.stats() == before
        with pytest.raises(KeyError):
            store.metadata("missing")


def test_catalog_snapshots_cannot_change_store_and_reject_closed_store(tmp_path):
    with make_store(tmp_path) as store:
        store.put("layer:1", b"data")
        keys = store.keys()
        meta = store.metadata("layer:1")
        meta["size"] = 999
        meta["pins"] = 999
        assert store.metadata("layer:1")["size"] == 4
        with store.pin("layer:1"):
            assert store.metadata("layer:1")["pins"] == 1
        assert store.metadata("layer:1")["pins"] == 0
        store.put("layer:2", b"more")
        assert keys == ("layer:1",)
        assert store.keys() == ("layer:1", "layer:2")
    for lookup in (
        lambda: store.contains("layer:1"),
        store.keys,
        lambda: store.metadata("layer:1"),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            lookup()
