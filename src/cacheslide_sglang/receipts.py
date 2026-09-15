"""Bounded, immutable request receipts; no pickle or arbitrary request paths."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from pathlib import Path


class ReceiptError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ReceiptError("receipt must contain finite JSON values") from exc


class ReceiptStore:
    MAX_BYTES = 4 * 1024 * 1024

    def __init__(self, root: str | Path):
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor):
            raise ReceiptError("receipt root must be a scoped absolute directory")
        # Do not accept a symlink anywhere in a configured output path.
        for part in (root, *root.parents):
            if part.is_symlink():
                raise ReceiptError("receipt root must not contain symlinks")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = root.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ReceiptError("receipt directory must be owned by this user")
        if info.st_mode & 0o022:
            raise ReceiptError("receipt directory must not be group/world writable")
        self.root = root.resolve()
        self._identity = (info.st_dev, info.st_ino)

    @staticmethod
    def _filename(run_id: str, request_id: str, nonce: str) -> str:
        for value in (run_id, request_id, nonce):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ReceiptError(
                    "receipt identities must be bounded nonempty strings"
                )
        return (
            hashlib.sha256(canonical_json([run_id, request_id, nonce])).hexdigest()
            + ".json"
        )

    def _directory(self) -> int:
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != self._identity:
            os.close(fd)
            raise ReceiptError("receipt directory identity changed")
        return fd

    def publish(self, receipt: dict) -> Path:
        if (
            type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != 1
            or receipt.get("status")
            not in {
                "complete",
                "cancelled",
                "failed",
            }
        ):
            raise ReceiptError("invalid terminal receipt schema/status")
        name = self._filename(
            receipt.get("run_id"), receipt.get("request_id"), receipt.get("nonce")
        )
        data = canonical_json(receipt)
        if len(data) > self.MAX_BYTES:
            raise ReceiptError("receipt exceeds its byte budget")
        directory = self._directory()
        temporary = None
        try:
            candidate = ".receipt-" + secrets.token_hex(16)
            fd = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            temporary = candidate
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # link is atomic and refuses an existing name (unlike replace).
            os.link(
                temporary,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
            os.fsync(directory)
        except OSError as exc:
            raise ReceiptError("cannot atomically publish immutable receipt") from exc
        finally:
            if temporary is not None:
                os.unlink(temporary, dir_fd=directory)
            os.close(directory)
        return self.root / name

    def read(self, run_id: str, request_id: str, nonce: str) -> dict:
        directory = self._directory()
        try:
            name = self._filename(run_id, request_id, nonce)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > self.MAX_BYTES:
                    raise ReceiptError("invalid receipt file or size")
                data = stream.read(self.MAX_BYTES + 1)
                if len(data) > self.MAX_BYTES:
                    raise ReceiptError("receipt exceeds its byte budget")
            result = json.loads(data)
            canonical_json(result)
            if (
                not isinstance(result, dict)
                or type(result.get("schema_version")) is not int
                or any(
                    result.get(key) != value
                    for key, value in (
                        ("run_id", run_id),
                        ("request_id", request_id),
                        ("nonce", nonce),
                        ("schema_version", 1),
                    )
                )
            ):
                raise ReceiptError("receipt identity mismatch")
            return result
        except (OSError, ValueError) as exc:
            raise ReceiptError("missing or unreadable request receipt") from exc
        finally:
            os.close(directory)


def validate_result(
    receipt: dict, result: dict, *, plan_digest: str, input_digest: str
) -> None:
    if (
        receipt.get("status") != "complete"
        or receipt.get("resources_released") is not True
    ):
        raise ReceiptError("request did not complete and release its resources")
    if (
        receipt.get("plan_digest") != plan_digest
        or receipt.get("input_digest") != input_digest
    ):
        raise ReceiptError("receipt plan/input mismatch")
    if receipt.get("output_ids") != result.get("output_ids"):
        raise ReceiptError("receipt and native generated token IDs disagree")
    if result.get("meta_info", {}).get("id") != receipt.get("request_id"):
        raise ReceiptError("native output request identity mismatch")
    if (result.get("meta_info", {}).get("finish_reason") or {}).get("type") == "abort":
        raise ReceiptError("native request aborted")
