"""Streaming, auditable local corpora for continued CoPE/LoRA pretraining.

The paper does not release a continued-pretraining corpus or split. This module
does not invent one: callers supply separate train/validation JSONL files and
an optional local Hugging Face tokenizer. Each line is a document, never a
synthetic Q/K/V fitting target. Documents are not packed across boundaries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import file_sha256


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ids(value) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or any(type(token) is not int or token < 0 for token in value)
    ):
        raise ValueError("token_ids must contain at least two nonnegative integers")
    return value


def _document_digest(row: dict, ids: list[int]) -> str:
    document = row.get("document_sha256", token_digest(ids))
    if (
        not isinstance(document, str)
        or len(document) != 64
        or any(c not in "0123456789abcdef" for c in document)
    ):
        raise ValueError("document_sha256 must be lowercase SHA-256")
    return document


class TokenCorpus:
    """Offset-indexed JSONL; token arrays are materialized one record at a time.

    Corpus SHA and document/sequence identities make resume and held-out
    leakage checks explicit. Runtime reads verify each indexed row's digest,
    so an in-place corpus edit cannot silently change an ongoing experiment.
    """

    def __init__(self, path: str | Path, *, max_sequence_length: int, vocab_size: int):
        if (
            type(max_sequence_length) is not int
            or max_sequence_length < 1
            or type(vocab_size) is not int
            or vocab_size < 1
        ):
            raise ValueError("sequence length and vocabulary must be positive integers")
        self.path = Path(path).resolve(strict=True)
        self.sha256 = file_sha256(self.path)
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.sequence_ids: list[str] = []
        self.document_ids: set[str] = set()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("corpus lines must be JSON objects")
                ids = _ids(row.get("token_ids"))
                if len(ids) > max_sequence_length + 1:
                    raise ValueError(
                        "corpus sequence exceeds configured sequence length"
                    )
                if max(ids) >= vocab_size:
                    raise ValueError("corpus token is outside the model vocabulary")
                identity = token_digest(ids)
                document = _document_digest(row, ids)
                self.offsets.append(offset)
                self.lengths.append(len(ids) - 1)
                self.sequence_ids.append(identity)
                self.document_ids.add(document)
        if not self.offsets:
            raise ValueError("corpus must contain at least one usable sequence")
        if file_sha256(self.path) != self.sha256:
            raise ValueError("corpus changed while indexing")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> list[int]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        ids = _ids(row.get("token_ids"))
        if token_digest(ids) != self.sequence_ids[index]:
            raise ValueError("corpus changed after indexing")
        return ids

    def manifest(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "sequences": len(self),
            "documents": len(self.document_ids),
            "supervised_tokens_per_epoch": sum(self.lengths),
        }


def assert_disjoint(train: TokenCorpus, validation: TokenCorpus) -> None:
    if train.document_ids & validation.document_ids:
        raise ValueError("train/validation document overlap is forbidden")
    if set(train.sequence_ids) & set(validation.sequence_ids):
        raise ValueError("train/validation token-sequence overlap is forbidden")


def prepare_corpus(
    source: str | Path,
    output: str | Path,
    *,
    sequence_length: int,
    tokenizer=None,
    tokenizer_provenance: dict | None = None,
    add_special_tokens: bool = True,
) -> dict:
    """Convert local text/token JSONL into transition-preserving token windows.

    A window has at most ``sequence_length + 1`` IDs and supervises at most
    ``sequence_length`` next-token targets. Neighbouring windows share exactly
    one token, so no boundary transition is duplicated or dropped. The held-out
    split must be made at the document level *before* this conversion.
    """
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    source = Path(source).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError("prepared corpus output must be a new directory")
    source_sha = file_sha256(source)
    output.mkdir(parents=True, exist_ok=False)
    count, tokens, documents = 0, 0, 0
    with source.open() as incoming, (output / "tokens.jsonl").open("x") as outgoing:
        for line_number, line in enumerate(incoming, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"source line {line_number} must be a JSON object")
            if "token_ids" in row:
                ids = _ids(row["token_ids"])
            else:
                text = row.get("text")
                if tokenizer is None or not isinstance(text, str) or not text.strip():
                    raise ValueError("text records need nonempty text and a tokenizer")
                ids = _ids(
                    tokenizer.encode(text, add_special_tokens=add_special_tokens)
                )
            document = _document_digest(row, ids)
            documents += 1
            for start in range(0, len(ids) - 1, sequence_length):
                window = ids[start : start + sequence_length + 1]
                outgoing.write(
                    json.dumps(
                        {
                            "token_ids": window,
                            "document_sha256": document,
                            "source_line": line_number,
                            "token_start": start,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
                tokens += len(window) - 1
    if not count:
        raise ValueError("source contains no usable training documents")
    if file_sha256(source) != source_sha:
        raise ValueError("source corpus changed during preparation")
    manifest = {
        "format": "cacheslide-token-corpus-v1",
        "source_path": str(source),
        "source_sha256": source_sha,
        "tokens_sha256": file_sha256(output / "tokens.jsonl"),
        "sequence_length": sequence_length,
        "documents": documents,
        "sequences": count,
        "supervised_tokens": tokens,
        "cross_document_packing": False,
        "add_special_tokens": add_special_tokens,
        "tokenizer": tokenizer_provenance,
        "paper_corpus_claim": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_local_tokenizer(path: str | Path):
    """Optional Transformers dependency; no network or remote model code."""
    from transformers import AutoTokenizer

    path = Path(path).resolve(strict=True)
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
    )
    files = {
        name: file_sha256(path / name) for name in names if (path / name).is_file()
    }
    if not {"tokenizer.json", "tokenizer.model"} & files.keys():
        raise ValueError("a complete local tokenizer snapshot is required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    return tokenizer, {
        "path": str(path),
        "files": files,
        "class": type(tokenizer).__name__,
    }
