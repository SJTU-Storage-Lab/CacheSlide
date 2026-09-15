"""Offline integrity, ownership and resume contract tests; no live downloads."""

import hashlib
import importlib.util
import io
import sys
import urllib.error
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "paper_downloader", Path(__file__).parents[1] / "scripts/download_paper_assets.py"
)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class Response(io.BytesIO):
    def __init__(
        self, body, *, status=200, headers=None, url="https://example.org/file"
    ):
        super().__init__(body)
        self.status, self.headers, self.url = status, headers or {}, url


def item(body=b"abcdef"):
    return {
        "path": "models/demo/weights.safetensors",
        "url": "https://example.org/file",
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_verified_publish_and_readonly_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda *a, **k: Response(b"abcdef")
    )
    receipt = downloader.download(tmp_path, item())
    assert receipt["official_integrity_verified"]
    assert (tmp_path / item()["path"]).read_bytes() == b"abcdef"
    assert not (tmp_path / (item()["path"] + ".part")).exists()
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("network on verify"),
    )
    assert downloader.download(tmp_path, item(), True) == receipt


def test_resume_uses_verified_range(tmp_path, monkeypatch):
    target = downloader.safe_child(tmp_path, item()["path"])
    target.with_name(target.name + ".part").write_bytes(b"abc")

    def open_request(request, **kwargs):
        assert request.headers["Range"] == "bytes=3-"
        return Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})

    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_request)
    assert downloader.download(tmp_path, item())["bytes"] == 6


@pytest.mark.parametrize(
    "range_header",
    ["bytes 2-5/6", "bytes 3-7/6", "bytes 3-5/7", "bytes 3-5/*", "bytes 3-5/6 suffix"],
)
def test_wrong_resume_range_never_publishes(tmp_path, monkeypatch, range_header):
    target = downloader.safe_child(tmp_path, item()["path"])
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(b"abc")
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **k: Response(
            b"def", status=206, headers={"Content-Range": range_header}
        ),
    )
    with pytest.raises(ValueError):
        downloader.download(tmp_path, item())
    assert not target.exists()
    assert partial.read_bytes() == b"abc"


@pytest.mark.parametrize("body", [b"wrong!", b"abc", b"abcdefg"])
def test_wrong_or_incomplete_body_never_publishes(tmp_path, monkeypatch, body):
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda *a, **k: Response(body)
    )
    with pytest.raises(ValueError):
        downloader.download(tmp_path, item())
    assert not (tmp_path / item()["path"]).exists()


def test_git_blob_checksum_is_not_plain_sha1(tmp_path, monkeypatch):
    asset = item()
    asset["sha256"] = None
    asset["git_blob_sha1"] = hashlib.sha1(b"blob 6\0abcdef").hexdigest()
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda *a, **k: Response(b"abcdef")
    )
    assert downloader.download(tmp_path, asset)["official_integrity_verified"]


def test_tls_downgrade_and_symlink_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **k: Response(b"abcdef", url="http://example.org/file"),
    )
    with pytest.raises(ValueError, match="TLS"):
        downloader.download(tmp_path, item())
    target = downloader.safe_child(tmp_path, item()["path"])
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"abcdef")
    target.symlink_to(unrelated)
    with pytest.raises(ValueError, match="owned regular"):
        downloader.download(tmp_path, item())
    assert unrelated.read_bytes() == b"abcdef"


def test_unknown_size_complete_partial_is_verified_without_416(tmp_path, monkeypatch):
    asset = item()
    del asset["size"]
    asset["max_bytes"] = 1024
    target = downloader.safe_child(tmp_path, asset["path"])
    target.with_name(target.name + ".part").write_bytes(b"abcdef")
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("unneeded range"),
    )
    assert downloader.download(tmp_path, asset)["bytes"] == 6


@pytest.mark.parametrize(
    "relative", ["../escape", "/absolute/file", "models/../escape", "models/\\escape"]
)
def test_path_traversal_is_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="unsafe asset path"):
        downloader.safe_child(tmp_path, relative)


def test_verify_missing_root_makes_no_directory(tmp_path, monkeypatch):
    root = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["download", "--root", str(root), "--verify-only"])
    with pytest.raises(FileNotFoundError):
        downloader.main()
    assert not root.exists()


def test_transient_http_retry_is_bounded(tmp_path, monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://example.org", 503, "busy", {}, None)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    with pytest.raises(urllib.error.HTTPError):
        downloader.download(tmp_path, item())
    assert len(calls) == 6
