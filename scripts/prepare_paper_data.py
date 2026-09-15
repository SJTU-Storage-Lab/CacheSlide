#!/usr/bin/env python3
"""Prepare a pinned, genuine HotpotQA-derived RPDC replay (not live Reflexion).

Only selected local Parquet/tokenizer assets are read, after size and official
digest validation. No model/dataset downloads, remote code, GPU use or generated
agent thoughts are performed. Explicit case counts prevent an accidental full
corpus preprocessing run. The recorded hash selection is independent of answers.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

PARTITIONS = ("training", "nll_validation", "calibration")
REPO = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
TOKENIZER_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
TOKENIZER_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
SOURCE_FILES = (
    "datasets/HotpotQA/distractor/train-00000-of-00002.parquet",
    "datasets/HotpotQA/distractor/train-00001-of-00002.parquet",
    "datasets/HotpotQA/distractor/validation-00000-of-00001.parquet",
    "models/Mistral-7B-Instruct-v0.2/tokenizer.json",
)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def split_for(case_id, seed):
    bucket = int(sha_text(f"{seed}:partition:{case_id}"), 16) % 100
    return (
        "training"
        if bucket < 80
        else ("nll_validation" if bucket < 90 else "calibration")
    )


def select_rows(rows, counts, seed, *, official_split):
    """Keep only deterministic smallest-hash cases in bounded selection heaps."""
    if official_split not in {"train", "validation"}:
        raise ValueError("only official train/validation splits are supported")
    if not counts or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("case counts must be explicitly positive")
    allowed = set(PARTITIONS) if official_split == "train" else {"evaluation"}
    if set(counts) != allowed:
        raise ValueError("all split case counts must be explicit")
    heaps, seen = {name: [] for name in counts}, set()
    for row in rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("missing/duplicate source case id")
        seen.add(case_id)
        group = split_for(case_id, seed) if official_split == "train" else "evaluation"
        priority = int(sha_text(f"{seed}:selection:{case_id}"), 16)
        heap = heaps[group]
        item = (-priority, case_id, row)
        if len(heap) < counts[group]:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)
    for group in heaps:
        if len(heaps[group]) != counts[group]:
            raise ValueError(f"not enough source cases in {group}; no silent subset")
    selected = {
        group: [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
        for group, heap in heaps.items()
    }
    return selected, seen


def source_parts(row):
    """Accept official HF context struct, or original Hotpot list-of-pairs."""
    if any(
        not isinstance(row.get(name), str) or not row[name]
        for name in ("id", "question", "answer")
    ):
        raise ValueError("Hotpot case needs real id, question and answer text")
    context = row.get("context")
    if isinstance(context, dict):
        titles, sentences = context.get("title"), context.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentences, list):
            raise ValueError("invalid Hotpot context struct")
        if len(titles) != len(sentences):
            raise ValueError("Hotpot titles/sentences mismatch")
        context = list(zip(titles, sentences, strict=True))
    if not isinstance(context, list) or not context:
        raise ValueError("Hotpot context must contain source documents")
    documents = []
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("invalid Hotpot context document")
        title, sentences = item
        if (
            not isinstance(title, str)
            or not title
            or not isinstance(sentences, list)
            or not sentences
            or any(not isinstance(sentence, str) for sentence in sentences)
        ):
            raise ValueError("context title/sentences must be real source strings")
        documents.append(f"Title: {title}\n{''.join(sentences)}\n\n")
    if len(documents) > 100:
        raise ValueError("too many source documents")
    return documents


def planned_case(row, encode, bos_id, *, question, operation, suffix=""):
    documents = source_parts(row)
    prefix = (
        "[INST] Answer the question using the reference documents. "
        "Return a concise answer without additional explanation.\n"
        f"Question: {row['question']}\nContext:\n"
        if question
        else "[INST] Read the following reference documents.\nContext:\n"
    )
    segments = [("control", "recompute", prefix)]
    segments += [
        (f"document-{index}", "reuse", text) for index, text in enumerate(documents)
    ]
    segments.append(("answer-control", "recompute", "\n[/INST]"))
    tokens, chunks = [], []
    for index, (name, role, text) in enumerate(segments):
        encoded = list(encode(text))
        if index == 0:
            encoded.insert(0, bos_id)
        if not encoded or any(type(token) is not int or token < 0 for token in encoded):
            raise ValueError("tokenizer returned invalid/empty chunk token IDs")
        chunks.append(
            {
                "id": name,
                "role": role,
                "start": len(tokens),
                "end": len(tokens) + len(encoded),
            }
        )
        tokens.extend(encoded)
    identity = "hotpot-" + row["id"]
    return {
        "id": identity + suffix,
        "prompt_token_ids": tokens,
        "cacheslide": {
            "version": 1,
            "operation": operation,
            "namespace": "hotpotqa-dataset-rpdc-v1",
            "task_id": identity,
            "chunks": chunks,
        },
    }


def prepare_rows(selected, encode, *, bos_id, eos_id, max_prompt_tokens):
    """Pure transformation; fixed chunks are identically tokenized in both paths."""
    if any(type(token) is not int or token < 0 for token in (bos_id, eos_id)):
        raise ValueError("explicit valid BOS/EOS token IDs are required")
    if type(max_prompt_tokens) is not int or max_prompt_tokens < 1:
        raise ValueError("positive prompt capacity is required")
    outputs = {
        name: []
        for name in (
            "train.corpus",
            "nll_validation.corpus",
            "train.provenance",
            "calibration.inputs",
            "evaluation.inputs",
            "evaluation.references",
            "population.inputs",
            "profile.inputs",
            "source_selection",
        )
    }
    seen, document_hashes = set(), {}
    for group in (*PARTITIONS, "evaluation"):
        for row in selected[group]:
            if row["id"] in seen:
                raise ValueError("source id overlaps a training/evaluation partition")
            seen.add(row["id"])
            prompt = planned_case(
                row,
                encode,
                bos_id,
                question=True,
                operation=("calibrate" if group == "calibration" else "reuse"),
            )
            if len(prompt["prompt_token_ids"]) > max_prompt_tokens:
                raise ValueError(
                    f"selected case {row['id']} has {len(prompt['prompt_token_ids'])} "
                    "prompt tokens, above capacity; no truncation or replacement"
                )
            full_text = canonical(
                {
                    "documents": source_parts(row),
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )
            document_hash = sha_text(full_text)
            if document_hash in document_hashes:
                raise ValueError(
                    "duplicate source content would leak or overweight cases"
                )
            document_hashes[document_hash] = group
            source = {
                "id": row["id"],
                "assignment": group,
                "repo": REPO,
                "revision": REVISION,
                "official_split": ("validation" if group == "evaluation" else "train"),
                "document_sha256": document_hash,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
            }
            outputs["source_selection"].append(source)
            if group in {"training", "nll_validation"}:
                answer_ids = list(encode(" " + row["answer"]))
                if not answer_ids or any(
                    type(t) is not int or t < 0 for t in answer_ids
                ):
                    raise ValueError("invalid tokenized training answer")
                ids = prompt["prompt_token_ids"] + answer_ids + [eos_id]
                corpus = {
                    "id": prompt["id"],
                    "token_ids": ids,
                    "document_sha256": document_hash,
                    "source": source,
                }
                name = (
                    "train.corpus" if group == "training" else "nll_validation.corpus"
                )
                outputs[name].append(corpus)
                if group == "training":
                    outputs["train.provenance"].append(
                        {"id": prompt["id"], "prompt_token_ids": ids}
                    )
            elif group == "calibration":
                outputs["calibration.inputs"].append(prompt)
            else:
                seed = planned_case(
                    row,
                    encode,
                    bos_id,
                    question=False,
                    operation="populate",
                    suffix="-population",
                )
                profile = planned_case(
                    row,
                    encode,
                    bos_id,
                    question=False,
                    operation="calibrate",
                    suffix="-profile",
                )
                seed_start = seed["cacheslide"]["chunks"][1]["start"]
                eval_start = prompt["cacheslide"]["chunks"][1]["start"]
                if seed_start == eval_start:
                    raise ValueError(
                        "selected prompt does not shift fixed chunk position"
                    )
                source["fixed_context_shift_tokens"] = eval_start - seed_start
                outputs["evaluation.inputs"].append(prompt)
                outputs["population.inputs"].append(seed)
                outputs["profile.inputs"].append(profile)
                outputs["evaluation.references"].append(
                    {
                        "id": prompt["id"],
                        "dataset": "hotpotqa-distractor-rpdc",
                        "split": "validation",
                        "revision": REVISION,
                        "prompt_token_ids": prompt["prompt_token_ids"],
                        "references": [row["answer"]],
                        "metric": "hotpotqa_f1",
                    }
                )
    return outputs


def verify_local_asset(root, entry):
    relative = entry.get("path")
    if not isinstance(relative, str) or relative not in SOURCE_FILES:
        raise ValueError("unrecognized local preparation asset")
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"asset not a complete regular local file: {relative}")
    expected_size = entry.get("size")
    if type(expected_size) is not int or expected_size < 1:
        raise ValueError("asset requires its official size")
    if path.stat().st_size != expected_size:
        raise ValueError(f"asset size mismatch: {relative}")
    sha256 = hashlib.sha256()
    git_hash = hashlib.sha1(f"blob {expected_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            git_hash.update(block)
    expected_sha, expected_git = entry.get("sha256"), entry.get("git_blob_sha1")
    if not (expected_sha or expected_git):
        raise ValueError("asset requires an official digest")
    if expected_sha and sha256.hexdigest() != expected_sha:
        raise ValueError(f"asset SHA256 mismatch: {relative}")
    if expected_git and git_hash.hexdigest() != expected_git:
        raise ValueError(f"asset Git blob mismatch: {relative}")
    return {"path": relative, "size": expected_size, "sha256": sha256.hexdigest()}


def parquet_rows(paths):
    import pyarrow.parquet as parquet

    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=128, columns=["id", "question", "answer", "context"]
        ):
            yield from batch.to_pylist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for group in (*PARTITIONS, "evaluation"):
        parser.add_argument(
            "--" + group.replace("_", "-") + "-count", required=True, type=int
        )
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("preparation output must be a new directory")
    counts = {
        group: getattr(args, group + "_count") for group in (*PARTITIONS, "evaluation")
    }
    if any(n < 1 for n in counts.values()):
        raise ValueError("case counts must be positive")
    manifest_path = args.asset_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    specs = manifest["spec"]
    if not any(
        s.get("repo") == REPO and s.get("revision") == REVISION
        for s in specs["datasets"]
    ):
        raise ValueError("official pinned HotpotQA dataset identity mismatch")
    if not any(
        s.get("repo") == TOKENIZER_REPO and s.get("revision") == TOKENIZER_REVISION
        for s in specs["models"]
    ):
        raise ValueError("official pinned Mistral tokenizer identity mismatch")
    entries = [
        entry for entry in manifest["files"] if entry.get("path") in SOURCE_FILES
    ]
    if {e["path"] for e in entries} != set(SOURCE_FILES) or len(entries) != len(
        SOURCE_FILES
    ):
        raise ValueError("manifest requires each preparation asset exactly once")
    verified = [verify_local_asset(args.asset_root, entry) for entry in entries]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.asset_root / SOURCE_FILES[3]))
    selected, train_ids = select_rows(
        parquet_rows([args.asset_root / path for path in SOURCE_FILES[:2]]),
        {name: counts[name] for name in PARTITIONS},
        args.seed,
        official_split="train",
    )
    evaluation, eval_ids = select_rows(
        parquet_rows([args.asset_root / SOURCE_FILES[2]]),
        {"evaluation": counts["evaluation"]},
        args.seed,
        official_split="validation",
    )
    if train_ids & eval_ids:
        raise ValueError("official train/validation source IDs overlap")
    selected.update(evaluation)
    outputs = prepare_rows(
        selected,
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        bos_id=tokenizer.token_to_id("<s>"),
        eos_id=tokenizer.token_to_id("</s>"),
        max_prompt_tokens=args.max_prompt_tokens,
    )
    result = {
        "schema_version": 1,
        "protocol": "hotpotqa-dataset-derived-rpdc-replay-v1",
        "is_live_reflexion": False,
        "paper_results_reproduced": False,
        "seed": args.seed,
        "requested_counts": counts,
        "official_train_cases": len(train_ids),
        "official_validation_cases": len(eval_ids),
        "selection": "smallest SHA256(seed:selection:id) after 80/10/10 hash partition",
        "tokenization": "Mistral INST; BOS once; independent exact chunk encoding",
        "max_prompt_tokens": args.max_prompt_tokens,
        "verified_source_assets": verified,
        "download_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "profile_preparation": (
            "Held-out documents only, no questions or answer labels; "
            "separate from adapter-training calibration split"
        ),
        "strict_reuse_caveat": (
            "A shifted contextual path can fail closed to recompute; "
            "this is not a successful cache hit or speedup."
        ),
        "output_files": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in outputs.items():
        path = args.output / (name + ".jsonl")
        with path.open("x") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
        result["output_files"][path.name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    with (args.output / "preparation.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
