"""Bounded, checksummed RAM/SSD page storage for the CacheSlide sidecar.

This module owns its cache files only. It does not modify vLLM block pools.
Host write accounting counts bytes written to page files, not SSD device WAF.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class CacheCapacityError(RuntimeError):
    """The configured capacity cannot satisfy an operation safely."""


class CacheIntegrityError(RuntimeError):
    """A persisted page failed its identity or checksum check."""


class StaleCompletionError(RuntimeError):
    """An asynchronous operation belongs to a retired store generation."""


@dataclass
class _Page:
    key: str
    generation: str
    checksum: str
    size: int
    selected_count: int
    last_access: int
    payload: bytes | None = None
    path: Path | None = None
    disk_size: int = 0
    pins: int = 0
    inflight: int = 0


class TieredPageStore:
    """Immutable pages with bounded resident bytes and atomic SSD publication.

    ``root`` must be an empty directory or an existing directory created by
    this class, owned by the current user. Keys are opaque namespaced strings;
    their hashes, never their text, form filenames. A second writer cannot open
    the same directory. Capacity exhaustion is explicit so callers can fall
    back to recomputation. There is no automatic deletion of disk-only pages.
    """

    _FORMAT = "cacheslide-pages-v1"
    _MAX_HEADER = 65536

    def __init__(
        self,
        root: str | Path,
        *,
        cpu_budget_bytes: int,
        disk_budget_bytes: int,
        max_workers: int = 2,
    ):
        if cpu_budget_bytes < 0 or disk_budget_bytes < 0 or max_workers < 1:
            raise ValueError("budgets must be nonnegative and max_workers positive")
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ValueError("cache root must not be a symbolic link")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError("cache root must be a directory")
        if hasattr(os, "getuid") and self.root.stat().st_uid != os.getuid():
            raise ValueError("cache root must be owned by the current user")
        marker = self.root / ".cacheslide-store.json"
        if marker.exists():
            if (
                marker.is_symlink()
                or json.loads(marker.read_text()).get("format") != self._FORMAT
            ):
                raise ValueError("unrecognized cache directory")
        else:
            if any(self.root.iterdir()):
                raise ValueError("refusing to claim a nonempty cache directory")
            with marker.open("x", encoding="utf-8") as stream:
                json.dump({"format": self._FORMAT}, stream)
        self._lock_path = self.root / ".writer.lock"
        try:
            self._lock_fd = os.open(
                self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "cache directory already has a writer; "
                "stale locks require explicit cleanup"
            ) from exc
        self.cpu_budget_bytes = cpu_budget_bytes
        self.disk_budget_bytes = disk_budget_bytes
        self._mutex = threading.RLock()
        self._pages: dict[str, _Page] = {}
        self._cpu_bytes = self._disk_bytes = 0
        self._cpu_reserved = self._disk_reserved = 0
        self._payload_write_bytes = self._physical_host_write_bytes = 0
        self._tick = 0
        self._epoch = 0
        self._closed = False
        self._closing = False
        self._loads: dict[str, Future[bytes]] = {}
        self._spills: dict[str, Future[None]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cacheslide-io"
        )
        try:
            self._restore_catalog()
        except BaseException:
            self._executor.shutdown(wait=True)
            os.close(self._lock_fd)
            self._lock_path.unlink()
            raise

    def _restore_catalog(self) -> None:
        if any(self.root.glob(".pending-*")):
            raise CacheIntegrityError(
                "unfinished cache writes require explicit recovery"
            )
        for path in sorted(self.root.glob("*.page")):
            if path.is_symlink():
                raise CacheIntegrityError("symbolic links are not cache pages")
            header, _ = self._decode_file(path, materialize=False)
            key = header["key"]
            if path.name != self._filename(key, header["generation"]):
                raise CacheIntegrityError("page filename does not match its identity")
            if key in self._pages:
                raise CacheIntegrityError("duplicate immutable page identity")
            page = _Page(
                key,
                header["generation"],
                header["checksum"],
                header["size"],
                header["selected_count"],
                self._next_tick(),
                path=path,
                disk_size=path.stat().st_size,
            )
            self._pages[key] = page
            self._disk_bytes += page.disk_size
        if self._disk_bytes > self.disk_budget_bytes:
            raise CacheCapacityError("existing disk pages exceed the disk budget")

    @staticmethod
    def _filename(key: str, generation: str) -> str:
        return (
            hashlib.sha256(key.encode("utf-8")).hexdigest() + "." + generation + ".page"
        )

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def _check_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("page store is closed")

    @staticmethod
    def _verify(page: _Page, payload: bytes) -> None:
        if (
            len(payload) != page.size
            or hashlib.sha256(payload).hexdigest() != page.checksum
        ):
            raise CacheIntegrityError("page payload checksum mismatch")

    def _encode_header(self, page: _Page) -> bytes:
        header = {
            "format": self._FORMAT,
            "key": page.key,
            "generation": page.generation,
            "checksum": page.checksum,
            "size": page.size,
            "selected_count": page.selected_count,
        }
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > self._MAX_HEADER:
            raise ValueError("page key is too long")
        return encoded + b"\n"

    def _decode_file(
        self, path: Path, *, materialize: bool = True
    ) -> tuple[dict, bytes | None]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            with os.fdopen(os.open(path, flags), "rb") as stream:
                raw_header = stream.readline(self._MAX_HEADER + 2)
                if len(raw_header) > self._MAX_HEADER + 1 or not raw_header.endswith(
                    b"\n"
                ):
                    raise CacheIntegrityError("invalid page header")
                header = json.loads(raw_header)
                size = header.get("size")
                if (
                    header.get("format") != self._FORMAT
                    or type(size) is not int
                    or size < 0
                    or type(header.get("selected_count")) is not int
                    or header["selected_count"] < 0
                    or not isinstance(header.get("key"), str)
                    or not isinstance(header.get("generation"), str)
                ):
                    raise CacheIntegrityError("invalid page metadata")
                # A corrupted size cannot cause an unbounded allocation.
                if (
                    size > self.disk_budget_bytes
                    or stream.seek(0, os.SEEK_END) != len(raw_header) + size
                ):
                    raise CacheIntegrityError("page size does not match metadata")
                stream.seek(len(raw_header))
                payload = None
                if materialize:
                    payload = stream.read(size)
                    checksum = hashlib.sha256(payload).hexdigest()
                else:
                    digest = hashlib.sha256()
                    while chunk := stream.read(
                        min(65536, max(1, self.cpu_budget_bytes))
                    ):
                        digest.update(chunk)
                    checksum = digest.hexdigest()
                if checksum != header.get("checksum"):
                    raise CacheIntegrityError("persisted page checksum mismatch")
                return header, payload
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise CacheIntegrityError(f"cannot read cache page: {path.name}") from exc

    def _read_disk(self, page: _Page) -> bytes:
        if page.path is None:
            raise CacheIntegrityError("page has no backing copy")
        header, payload = self._decode_file(page.path)
        if header["key"] != page.key or header["generation"] != page.generation:
            raise CacheIntegrityError("persisted page identity mismatch")
        assert payload is not None
        self._verify(page, payload)
        return payload

    def put(self, key: str, payload: bytes, selected_count: int = 0) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty namespaced string")
        if (
            not isinstance(payload, bytes)
            or type(selected_count) is not int
            or selected_count < 0
        ):
            raise ValueError("payload must be bytes and selected_count nonnegative")
        with self._mutex:
            self._check_open()
            previous = self._pages.get(key)
            if previous is not None:
                if (
                    len(payload) != previous.size
                    or hashlib.sha256(payload).hexdigest() != previous.checksum
                    or previous.selected_count != selected_count
                ):
                    raise ValueError(
                        "immutable page key already exists with different data"
                    )
                if previous.payload is not None:
                    self._verify(previous, previous.payload)
                    if previous.payload != payload:
                        raise ValueError("immutable page key collision")
                else:
                    # Stream verification avoids an unreserved duplicate payload.
                    self._spill_sync(previous)
                previous.last_access = self._next_tick()
                return
            if len(payload) > self.cpu_budget_bytes:
                raise CacheCapacityError("page exceeds the CPU budget")
            self._make_cpu_space(len(payload))
            page = _Page(
                key,
                uuid.uuid4().hex,
                hashlib.sha256(payload).hexdigest(),
                len(payload),
                selected_count,
                self._next_tick(),
                payload=payload,
            )
            # Validate header size before committing the page.
            self._encode_header(page)
            self._pages[key] = page
            self._cpu_bytes += len(payload)

    def _candidates(self, exclude: str | None = None) -> list[_Page]:
        eligible = [
            p
            for p in self._pages.values()
            if p.payload is not None
            and not p.pins
            and not p.inflight
            and p.key != exclude
        ]
        return sorted(
            eligible,
            key=lambda p: (
                p.selected_count > 0,
                -p.selected_count if p.selected_count else 0,
                p.last_access,
                p.key,
            ),
        )

    def _make_cpu_space(self, needed: int, exclude: str | None = None) -> list[str]:
        evicted = []
        for page in self._candidates(exclude):
            if self._cpu_bytes + self._cpu_reserved + needed <= self.cpu_budget_bytes:
                break
            self._spill_sync(page)
            # Publication precedes resident removal, including for clean pages.
            self._cpu_bytes -= page.size
            page.payload = None
            evicted.append(page.key)
        if self._cpu_bytes + self._cpu_reserved + needed > self.cpu_budget_bytes:
            raise CacheCapacityError(
                "CPU capacity unavailable: pages are pinned or in flight"
            )
        return evicted

    def _write_page(self, page: _Page, header: bytes, payload: bytes) -> Path:
        destination = self.root / self._filename(page.key, page.generation)
        temporary = self.root / (".pending-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        published = False
        try:
            # Avoid allocating a second full-sized payload for a header prefix.
            for segment in (header, payload):
                view = memoryview(segment)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short cache page write")
                    with self._mutex:
                        self._physical_host_write_bytes += written
                    view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            # Check the complete temporary file before atomic publication.
            metadata, _ = self._decode_file(temporary, materialize=False)
            if (
                metadata["generation"] != page.generation
                or metadata["checksum"] != page.checksum
            ):
                raise CacheIntegrityError("temporary page generation mismatch")
            os.replace(temporary, destination)
            published = True
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return destination
        except BaseException:
            if published:
                destination.unlink(missing_ok=True)
            raise
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary.exists():
                temporary.unlink()

    def _spill_sync(self, page: _Page) -> None:
        if page.path is not None:
            # Do not discard the only valid copy if backing was corrupted.
            metadata, _ = self._decode_file(page.path, materialize=False)
            if (
                metadata["key"] != page.key
                or metadata["generation"] != page.generation
                or metadata["checksum"] != page.checksum
            ):
                raise CacheIntegrityError("persisted page identity mismatch")
            return
        if page.payload is None:
            raise CacheIntegrityError("cannot spill a page without resident data")
        self._verify(page, page.payload)
        header = self._encode_header(page)
        encoded_size = len(header) + page.size
        if (
            self._disk_bytes + self._disk_reserved + encoded_size
            > self.disk_budget_bytes
        ):
            raise CacheCapacityError(
                "disk capacity unavailable; resident page preserved"
            )
        self._disk_reserved += encoded_size
        try:
            path = self._write_page(page, header, page.payload)
            page.path, page.disk_size = path, encoded_size
            self._disk_bytes += encoded_size
            self._payload_write_bytes += page.size
        finally:
            self._disk_reserved -= encoded_size

    def evict(self, required_bytes: int = 0) -> list[str]:
        """Ensure ``required_bytes`` free RAM; zero explicitly evicts one page."""
        if required_bytes < 0:
            raise ValueError("required_bytes must be nonnegative")
        with self._mutex:
            self._check_open()
            if required_bytes:
                return self._make_cpu_space(required_bytes)
            candidates = self._candidates()
            if not candidates:
                raise CacheCapacityError("no unpinned resident page can be evicted")
            page = candidates[0]
            self._spill_sync(page)
            page.payload = None
            self._cpu_bytes -= page.size
            return [page.key]

    def read(self, key: str) -> bytes:
        return self.submit_load(key).result()

    def contains(self, key: str) -> bool:
        """Check the catalog without loading or touching page payloads."""
        with self._mutex:
            self._check_open()
            return key in self._pages

    def keys(self, prefix: str = "") -> tuple[str, ...]:
        """Return a deterministic independent snapshot of catalog keys."""
        if not isinstance(prefix, str):
            raise ValueError("key prefix must be a string")
        with self._mutex:
            self._check_open()
            return tuple(sorted(key for key in self._pages if key.startswith(prefix)))

    def metadata(self, key: str) -> dict[str, int | str | bool]:
        """Return scalar page metadata without loading its payload."""
        with self._mutex:
            self._check_open()
            page = self._pages[key]
            return {
                "size": page.size,
                "selected_count": page.selected_count,
                "generation": page.generation,
                "resident": page.payload is not None,
                "backed": page.path is not None,
                "pins": page.pins,
            }

    def submit_load(self, key: str) -> Future[bytes]:
        with self._mutex:
            self._check_open()
            page = self._pages[key]
            page.last_access = self._next_tick()
            if page.payload is not None:
                self._verify(page, page.payload)
                ready: Future[bytes] = Future()
                ready.set_result(page.payload)
                return ready
            if key in self._loads:
                return self._loads[key]
            if page.size > self.cpu_budget_bytes:
                raise CacheCapacityError("page exceeds the CPU budget")
            self._make_cpu_space(page.size, exclude=key)
            self._cpu_reserved += page.size
            page.inflight += 1
            epoch = self._epoch
            result: Future[bytes] = Future()
            # Public cancellation cannot cancel bookkeeping or the underlying I/O.
            result.set_running_or_notify_cancel()
            self._loads[key] = result

            def load() -> None:
                failure: BaseException | None = None
                payload: bytes | None = None
                try:
                    payload = self._read_disk(page)
                except BaseException as exc:
                    failure = exc
                with self._mutex:
                    if failure is None:
                        if (
                            self._closed
                            or epoch != self._epoch
                            or self._pages.get(key) is not page
                        ):
                            failure = StaleCompletionError(
                                "page load completed for a retired generation"
                            )
                        else:
                            page.payload = payload
                            self._cpu_bytes += page.size
                    self._cpu_reserved -= page.size
                    page.inflight -= 1
                    self._loads.pop(key, None)
                if failure is not None:
                    result.set_exception(failure)
                else:
                    assert payload is not None
                    result.set_result(payload)

            self._executor.submit(load)
            return result

    def submit_spill(self, key: str) -> Future[None]:
        with self._mutex:
            self._check_open()
            page = self._pages[key]
            if key in self._spills:
                return self._spills[key]
            result: Future[None] = Future()
            result.set_running_or_notify_cancel()
            page.inflight += 1
            epoch = self._epoch
            self._spills[key] = result

            def spill() -> None:
                failure: BaseException | None = None
                try:
                    with self._mutex:
                        if (
                            self._closed
                            or epoch != self._epoch
                            or self._pages.get(key) is not page
                        ):
                            raise StaleCompletionError(
                                "page spill belongs to a retired generation"
                            )
                        self._spill_sync(page)
                except BaseException as exc:
                    failure = exc
                finally:
                    with self._mutex:
                        page.inflight -= 1
                        self._spills.pop(key, None)
                if failure is not None:
                    result.set_exception(failure)
                else:
                    result.set_result(None)

            self._executor.submit(spill)
            return result

    @contextmanager
    def pin(self, key: str) -> Iterator[bytes]:
        with self._mutex:
            self._check_open()
            page = self._pages[key]
            page.pins += 1
        try:
            yield self.read(key)
        finally:
            with self._mutex:
                page.pins -= 1

    def delete(self, key: str) -> None:
        """Remove one explicit page owned by this store, never a directory."""
        with self._mutex:
            self._check_open()
            page = self._pages[key]
            if page.pins or page.inflight:
                raise CacheCapacityError("cannot delete a pinned or in-flight page")
            if page.path is not None:
                page.path.unlink()
                self._disk_bytes -= page.disk_size
            if page.payload is not None:
                self._cpu_bytes -= page.size
            del self._pages[key]

    def stats(self) -> dict[str, int]:
        with self._mutex:
            return {
                "pages": len(self._pages),
                "live_cpu_bytes": self._cpu_bytes,
                "live_disk_bytes": self._disk_bytes,
                "inflight_cpu_reserved_bytes": self._cpu_reserved,
                "inflight_disk_reserved_bytes": self._disk_reserved,
                "payload_write_bytes": self._payload_write_bytes,
                "physical_host_write_bytes": self._physical_host_write_bytes,
                "pinned_pages": sum(p.pins > 0 for p in self._pages.values()),
                "inflight_pages": sum(p.inflight > 0 for p in self._pages.values()),
            }

    def close(self, *, cancel_pending: bool = False) -> None:
        """Drain I/O and release directory ownership.

        ``cancel_pending`` rejects outstanding completions before draining.
        Unspilled RAM pages are ephemeral; call ``submit_spill`` for durability.
        """
        with self._mutex:
            self._closing = True
            if cancel_pending:
                self._closed = True
                self._epoch += 1
        self._executor.shutdown(wait=True)
        with self._mutex:
            if self._lock_fd >= 0:
                self._closed = True
                self._epoch += 1
                os.close(self._lock_fd)
                self._lock_fd = -1
                self._lock_path.unlink()

    def __enter__(self) -> TieredPageStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
