"""Only small schema fixtures here; none of these are experiment results."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheslide_core.contracts import RequestPlan
from cacheslide_sglang.reproduction_report import validate_evaluation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_paper_data.py"
SPEC = importlib.util.spec_from_file_location("paper_data_for_tests", SCRIPT)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def encode(text):
    return [ord(char) + 2 for char in text]


def decode(ids):
    return "".join(chr(token - 2) for token in ids if token > 2)


def source(case_id):
    return {
        "id": case_id,
        "question": "Where did the source character travel? " + case_id,
        "answer": "ANSWER_ONLY_" + case_id,
        "context": {
            "title": ["source title"],
            "sentences": [["Real fixture context."]],
        },
    }


def selected():
    return {name: [source(name + "-case")] for name in (*data.PARTITIONS, "evaluation")}


def test_hash_selection_is_shard_order_independent_and_disjoint():
    rows = [source(str(index)) for index in range(200)]
    counts = {name: 2 for name in data.PARTITIONS}
    first, ids = data.select_rows(rows, counts, 9, official_split="train")
    reversed_order, _ = data.select_rows(
        list(reversed(rows)), counts, 9, official_split="train"
    )
    assert first == reversed_order
    assert len(ids) == 200
    selected_ids = [row["id"] for group in first.values() for row in group]
    assert len(selected_ids) == len(set(selected_ids)) == 6
    for name, group in first.items():
        assert all(data.split_for(row["id"], 9) == name for row in group)


def test_missing_duplicate_or_insufficient_source_is_not_silently_skipped():
    with pytest.raises(ValueError, match="duplicate"):
        data.select_rows(
            [source("one"), source("one")],
            {"evaluation": 1},
            9,
            official_split="validation",
        )
    with pytest.raises(ValueError, match="not enough"):
        data.select_rows(
            [source("one")], {"evaluation": 2}, 9, official_split="validation"
        )


def test_fixed_context_is_nonprefix_shifted_and_cache_profile_keys_match():
    outputs = data.prepare_rows(
        selected(), encode, bos_id=1, eos_id=2, max_prompt_tokens=4096
    )
    evaluation = outputs["evaluation.inputs"][0]
    seed = outputs["population.inputs"][0]
    profile = outputs["profile.inputs"][0]
    plans = [
        RequestPlan.parse(row["cacheslide"], row["prompt_token_ids"])
        for row in (evaluation, seed, profile)
    ]
    assert plans[0].fixed_layout == plans[1].fixed_layout == plans[2].fixed_layout
    assert len({plan.cache_key("model", 0) for plan in plans}) == 1
    assert len({plan.profile_key("model", 0) for plan in plans}) == 1
    assert plans[0].chunks[1].start != plans[1].chunks[1].start
    assert all(plan.chunks[0].role == "recompute" for plan in plans)
    assert all(plan.chunks[1].role == "reuse" for plan in plans)
    assert evaluation["prompt_token_ids"].count(1) == 1


def test_eval_gold_not_inserted_and_profile_has_no_heldout_question_or_answer():
    cases = selected()
    outputs = data.prepare_rows(
        cases, encode, bos_id=1, eos_id=2, max_prompt_tokens=4096
    )
    gold = cases["evaluation"][0]["answer"]
    question = cases["evaluation"][0]["question"]
    evaluation = decode(outputs["evaluation.inputs"][0]["prompt_token_ids"])
    assert question in evaluation and gold not in evaluation
    for name in ("population.inputs", "profile.inputs"):
        text = decode(outputs[name][0]["prompt_token_ids"])
        assert gold not in text and question not in text
        assert "Real fixture context." in text
    assert outputs["evaluation.references"][0]["references"] == [gold]
    assert outputs["evaluation.references"][0]["metric"] == "hotpotqa_f1"
    train_text = decode(outputs["train.corpus"][0]["token_ids"])
    assert cases["training"][0]["answer"] in train_text
    assert outputs["train.corpus"][0]["token_ids"][-1] == 2
    assert outputs["nll_validation.corpus"][0]["token_ids"][-1] == 2


def test_prepared_inputs_pass_hard_evaluation_leak_audit():
    outputs = data.prepare_rows(
        selected(), encode, bos_id=1, eos_id=2, max_prompt_tokens=4096
    )
    evaluation = validate_evaluation(
        outputs["evaluation.references"],
        outputs["train.provenance"],
        outputs["profile.inputs"],
    )
    assert len(evaluation) == 1


def test_original_hotpot_context_representation_matches_hf_struct():
    row = source("one")
    expected = data.source_parts(row)
    row["context"] = [["source title", ["Real fixture context."]]]
    assert data.source_parts(row) == expected


@pytest.mark.parametrize("kind", ["capacity", "duplicate_id", "duplicate_content"])
def test_selected_rows_never_truncated_or_replaced(kind):
    cases = selected()
    capacity = 4096
    if kind == "capacity":
        capacity = 1
    elif kind == "duplicate_id":
        cases["evaluation"] = copy.deepcopy(cases["training"])
    else:
        cases["evaluation"] = copy.deepcopy(cases["training"])
        cases["evaluation"][0]["id"] = "different-id"
    with pytest.raises(ValueError):
        data.prepare_rows(cases, encode, bos_id=1, eos_id=2, max_prompt_tokens=capacity)


def asset_fixture(tmp_path):
    entries = []
    for relative in data.SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = relative.encode()
        path.write_bytes(content)
        entries.append(
            {
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "spec": {
            "models": [
                {"repo": data.TOKENIZER_REPO, "revision": data.TOKENIZER_REVISION}
            ],
            "datasets": [{"repo": data.REPO, "revision": data.REVISION}],
        },
        "files": entries,
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest))
    return entries


def test_asset_verifies_size_sha_and_git_blob_before_any_parquet_parse(tmp_path):
    entries = asset_fixture(tmp_path)
    for entry in entries:
        assert data.verify_local_asset(tmp_path, entry)["sha256"] == entry["sha256"]
    entry = entries[-1]
    content = (tmp_path / entry["path"]).read_bytes()
    entry["git_blob_sha1"] = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content
    ).hexdigest()
    entry.pop("sha256")
    assert data.verify_local_asset(tmp_path, entry)["sha256"]
    entry["git_blob_sha1"] = "0" * 40
    with pytest.raises(ValueError, match="Git blob mismatch"):
        data.verify_local_asset(tmp_path, entry)


def test_partial_or_symlink_or_wrong_sha_asset_refused(tmp_path):
    entries = asset_fixture(tmp_path)
    entry = entries[0]
    path = tmp_path / entry["path"]
    path.rename(path.with_suffix(".parquet.part"))
    with pytest.raises(ValueError, match="complete regular"):
        data.verify_local_asset(tmp_path, entry)
    path.symlink_to(path.with_suffix(".parquet.part"))
    with pytest.raises(ValueError, match="complete regular"):
        data.verify_local_asset(tmp_path, entry)
    entries[1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        data.verify_local_asset(tmp_path, entries[1])


def test_cli_local_verified_preparation_preserves_exact_output_hashes(
    tmp_path, monkeypatch
):
    root = tmp_path / "assets"
    root.mkdir()
    asset_fixture(root)
    train_rows = [source(str(index)) for index in range(200)]

    def parquet(paths):
        return iter(train_rows if "train-" in paths[0].name else [source("heldout")])

    tokenizer = SimpleNamespace(
        token_to_id=lambda name: {"<s>": 1, "</s>": 2}[name],
        encode=lambda text, add_special_tokens: SimpleNamespace(ids=encode(text)),
    )
    monkeypatch.setattr(data, "parquet_rows", parquet)
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda path: tokenizer)),
    )
    output = tmp_path / "prepared"
    args = [
        "--asset-root",
        str(root),
        "--output",
        str(output),
        "--max-prompt-tokens",
        "4096",
    ]
    for group in (*data.PARTITIONS, "evaluation"):
        args += ["--" + group.replace("_", "-") + "-count", "1"]
    assert data.main(args) == 0
    result = json.loads((output / "preparation.json").read_text())
    assert result["is_live_reflexion"] is False
    assert result["paper_results_reproduced"] is False
    assert result["official_train_cases"] == 200
    for filename, expected in result["output_files"].items():
        assert (
            hashlib.sha256((output / filename).read_bytes()).hexdigest()
            == (expected["sha256"])
        )
    with pytest.raises(ValueError, match="new directory"):
        data.main(args)
