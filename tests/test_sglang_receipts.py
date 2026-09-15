import os

import pytest

from cacheslide_sglang.receipts import ReceiptError, ReceiptStore, validate_result


def receipt(**changes):
    return {
        "schema_version": 1,
        "run_id": "run",
        "request_id": "rid",
        "nonce": "nonce",
        "status": "complete",
        "resources_released": True,
        "output_ids": [4, 5],
        "plan_digest": "plan",
        "input_digest": "input",
        "metrics": {},
        **changes,
    }


def test_atomic_immutable_receipt_and_output_check(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    path = store.publish(receipt())
    assert store.read("run", "rid", "nonce") == receipt()
    with pytest.raises(ReceiptError, match="immutable"):
        store.publish(receipt(output_ids=[999]))
    assert store.read("run", "rid", "nonce")["output_ids"] == [4, 5]
    assert list(path.parent.glob(".receipt-*")) == []
    result = {"output_ids": [4, 5], "meta_info": {"id": "rid"}}
    validate_result(receipt(), result, plan_digest="plan", input_digest="input")
    with pytest.raises(ReceiptError, match="token IDs"):
        validate_result(
            receipt(output_ids=[4]), result, plan_digest="plan", input_digest="input"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"resources_released": False},
        {"plan_digest": "wrong"},
        {"input_digest": "wrong"},
        {"request_id": "another"},
    ],
)
def test_result_validation_fails_closed(change):
    with pytest.raises(ReceiptError):
        validate_result(
            receipt(**change),
            {"output_ids": [4, 5], "meta_info": {"id": "rid"}},
            plan_digest="plan",
            input_digest="input",
        )


def test_symlinks_budget_and_corruption(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReceiptError, match="symlink"):
        ReceiptStore(symlink)
    store = ReceiptStore(tmp_path / "receipts")
    with pytest.raises(ReceiptError, match="finite"):
        store.publish(receipt(metrics={"bad": float("nan")}))
    with pytest.raises(ReceiptError, match="budget"):
        store.publish(receipt(metrics={"big": "a" * store.MAX_BYTES}))
    path = store.publish(receipt())
    path.write_text('{"corrupted":true}')
    with pytest.raises(ReceiptError, match="identity"):
        store.read("run", "rid", "nonce")
    path.unlink()
    path.symlink_to(tmp_path / "not-a-receipt")
    with pytest.raises(ReceiptError):
        store.read("run", "rid", "nonce")


def test_directory_identity_and_permission_guards(tmp_path):
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    root.rename(tmp_path / "moved")
    root.mkdir()
    with pytest.raises(ReceiptError, match="identity"):
        store.publish(receipt())
    os.chmod(root, 0o777)
    with pytest.raises(ReceiptError, match="writable"):
        ReceiptStore(root)
