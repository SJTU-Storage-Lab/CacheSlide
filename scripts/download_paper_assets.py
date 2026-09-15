#!/usr/bin/env python3
"""Download pinned public research assets without executing remote model code.

Only safetensors and tokenizer/config files are selected. Partial files are not
usable artifacts: publish only after official LFS SHA256 or Git blob validation.
Existing unrelated directories, changed manifests and symlinks fail closed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

OWNER = "cacheslide-paper-assets-v1"


def emit(**values):
    print(json.dumps({"time": time.time(), **values}), flush=True)


def digest(path, algorithm="sha256", git=False):
    h = hashlib.new(algorithm)
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError(f"not an owned regular file: {path.name}")


def json_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_child(root, relative, *, create=True):
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or PurePosixPath(relative).is_absolute()
        or any(p in (".", "..") or "\\" in p for p in parts)
    ):
        raise ValueError("unsafe asset path")
    current = root
    for part in parts[:-1]:
        current = current / part
        if create:
            current.mkdir(exist_ok=True)
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("unsafe parent directory")
    return current / parts[-1]


def resolve(spec, only):
    files = []
    groups = ["models", "datasets", "archives"] if only == "all" else [only]
    for group in groups:
        for entry in spec[group]:
            if group == "archives":
                files.append(
                    {**entry, "path": f"datasets/{entry['name']}/{entry['filename']}"}
                )
                continue
            prefix = "models" if entry["kind"] == "model" else "datasets"
            api_kind = "models" if entry["kind"] == "model" else "datasets"
            api = f"https://huggingface.co/api/{api_kind}/{entry['repo']}/revision/{entry['revision']}?blobs=true"
            with urllib.request.urlopen(api, timeout=30) as response:
                metadata = json.load(response)
            if metadata.get("sha") != entry["revision"] or metadata.get("gated"):
                raise ValueError("revision mismatch or gated repository")
            chosen = []
            for item in metadata["siblings"]:
                name = item["rfilename"]
                if not any(fnmatch.fnmatchcase(name, p) for p in entry["include"]):
                    continue
                size = item["size"]
                if type(size) is not int or size <= 0:
                    raise ValueError("missing official file size")
                lfs = item.get("lfs")
                checksum = lfs["sha256"] if lfs else item["blobId"]
                if len(checksum) != (64 if lfs else 40):
                    raise ValueError("missing official checksum")
                url_prefix = "datasets/" if group == "datasets" else ""
                url = f"https://huggingface.co/{url_prefix}{entry['repo']}/resolve/{entry['revision']}/{urllib.parse.quote(name)}"
                chosen.append(
                    {
                        "path": f"{prefix}/{entry['name']}/{name}",
                        "url": url,
                        "size": size,
                        "sha256": checksum if lfs else None,
                        "git_blob_sha1": None if lfs else checksum,
                    }
                )
            if not chosen:
                raise ValueError("empty upstream selection")
            files.extend(chosen)
    return files


def verify(path, item):
    regular(path)
    if item.get("size") is not None and path.stat().st_size != item["size"]:
        raise ValueError(f"size mismatch: {item['path']}")
    if item.get("sha256"):
        if digest(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {item['path']}")
    elif digest(path, "sha1", git=True) != item["git_blob_sha1"]:
        raise ValueError(f"Git blob mismatch: {item['path']}")
    return {
        "path": item["path"],
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "official_integrity_verified": True,
    }


def download(root, item, verify_only=False):
    target = safe_child(root, item["path"], create=not verify_only)
    partial = target.with_name(target.name + ".part")
    if target.exists() or target.is_symlink():
        result = verify(target, item)
        emit(event="verified_existing", **result)
        return result
    if verify_only:
        raise FileNotFoundError(item["path"])
    bound = item.get("size", item.get("max_bytes"))
    for attempt in range(6):
        if partial.exists() or partial.is_symlink():
            regular(partial)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > bound:
            raise ValueError("oversized partial")
        if item.get("size") == offset:
            break
        # An unknown-length archive may have completed immediately before an
        # interrupted connection. Verify it instead of asking for a 416 range.
        if offset and item.get("size") is None and digest(partial) == item["sha256"]:
            break
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                if urllib.parse.urlparse(response.url).scheme != "https":
                    raise ValueError("TLS downgrade refused")
                if offset:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status != 206 or not match:
                        raise ValueError("server did not honor resume Range")
                    start, end, total = map(int, match.groups())
                    if start != offset or not start <= end < total or total > bound:
                        raise ValueError("invalid resume range")
                    if item.get("size") and total != item["size"]:
                        raise ValueError("resume total mismatch")
                elif response.status != 200:
                    raise ValueError("unexpected initial response")
                fd = os.open(
                    partial,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                    0o600,
                )
                with os.fdopen(fd, "ab") as stream:
                    last_report = offset
                    while chunk := response.read(1024 * 1024):
                        offset += len(chunk)
                        if offset > bound:
                            raise ValueError("body exceeded declared bound")
                        stream.write(chunk)
                        if offset - last_report >= 64 * 1024 * 1024:
                            emit(
                                event="partial_unverified",
                                path=item["path"],
                                bytes=offset,
                            )
                            last_report = offset
                    stream.flush()
                    os.fsync(stream.fileno())
            break
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 5:
                raise
            emit(
                event="network_retry",
                path=item["path"],
                attempt=attempt,
                status=error.code,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
        ) as error:
            if (
                isinstance(getattr(error, "reason", error), ssl.SSLError)
                or attempt == 5
            ):
                raise
            emit(
                event="network_retry",
                path=item["path"],
                attempt=attempt,
                error=type(error).__name__,
            )
        time.sleep(min(2**attempt, 16))
    result = verify(partial, item)
    os.link(partial, target)
    partial.unlink()
    emit(event="published_verified", **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/paper_assets.json",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--only", choices=("all", "models", "datasets", "archives"), default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    # Do not resolve away symlinks in the explicitly requested target.
    root = args.root.absolute()
    if root in (Path("/"), Path.home()) or len(root.parts) < 4:
        parser.error("use a dedicated asset directory")
    if args.verify_only and not root.is_dir():
        raise FileNotFoundError("verification requires an existing asset directory")
    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError("unsafe asset directory")
    spec_bytes = args.spec.read_bytes()
    identity = {
        "owner": OWNER,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "selection": args.only,
    }
    owner = root / ".owner.json"
    if owner.exists():
        regular(owner)
        if json.loads(owner.read_text()) != identity:
            raise ValueError("asset owner/spec mismatch")
    elif args.verify_only or any(root.iterdir()):
        raise ValueError("refusing unrelated nonempty asset directory")
    else:
        json_new(owner, identity)
    lock_flags = os.O_RDONLY if args.verify_only else os.O_CREAT | os.O_RDWR
    lock = os.open(root / ".download.lock", lock_flags | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = root / "download_manifest.json"
    if manifest.exists():
        regular(manifest)
        resolved = json.loads(manifest.read_text())
        if resolved["identity"] != identity:
            raise ValueError("resolved manifest identity mismatch")
        files = resolved["files"]
    else:
        if args.verify_only:
            raise FileNotFoundError("verification requires a resolved manifest")
        files = resolve(json.loads(spec_bytes), args.only)
        json_new(
            manifest,
            {"identity": identity, "spec": json.loads(spec_bytes), "files": files},
        )
    remaining = sum(
        item.get("size", item.get("max_bytes", 0))
        for item in files
        if not (root / item["path"]).is_file()
    )
    if not args.verify_only and shutil.disk_usage(root).free < remaining + 40 * 1024**3:
        raise ValueError("insufficient space with 40GiB reserve")
    emit(
        event="start",
        files=len(files),
        total_upper_bound_bytes=sum(
            i.get("size", i.get("max_bytes", 0)) for i in files
        ),
        verify_only=args.verify_only,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        receipts = list(
            executor.map(lambda item: download(root, item, args.verify_only), files)
        )
    report = root / ("verification-" + str(time.time_ns()) + ".json")
    json_new(
        report,
        {
            "identity": identity,
            "independent_verify_only": args.verify_only,
            "all_files_verified": True,
            "files": receipts,
        },
    )
    emit(
        event="all_files_verified",
        report=str(report),
        bytes=sum(r["bytes"] for r in receipts),
    )
    os.close(lock)


if __name__ == "__main__":
    main()
