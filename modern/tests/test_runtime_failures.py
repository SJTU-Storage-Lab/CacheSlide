from dataclasses import replace

import pytest
import torch

from cacheslide_vllm.runtime import CacheSlideRuntime

from .test_runtime import build_runtime, plan, prefill


def install_inplace_residual_norms(model, monkeypatch):
    """Exercise native fused RMSNorm's mutation contract on the CPU model."""
    norms = [model.norm]
    for layer in model.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
    for norm in norms:
        original = norm.forward

        def fused(hidden, residual=None, original=original):
            if residual is None:
                return original(hidden)
            residual.add_(hidden)
            return original(residual), residual

        monkeypatch.setattr(norm, "forward", fused)


def test_population_disk_failure_retries_from_unmutated_embeddings(
    tmp_path, monkeypatch
):
    model, runtime = build_runtime(tmp_path)
    try:
        install_inplace_residual_norms(model, monkeypatch)
        expected = prefill(model, runtime, plan())
        original_write = runtime.store._write_page
        writes = 0

        def failed_layer_write(*args):
            nonlocal writes
            writes += 1
            if writes == 5:
                raise OSError("injected layer persistence failure")
            return original_write(*args)

        monkeypatch.setattr(runtime.store, "_write_page", failed_layer_write)
        actual = prefill(model, runtime, plan(operation="populate"))
        torch.testing.assert_close(actual, expected)
        assert runtime.last_metrics["fallback"]
        assert runtime.last_metrics["layer_rows"] == [8] * 6
        assert not runtime.store.keys()
        assert not runtime.requests["r"].created_keys
    finally:
        runtime.close()


def test_commit_failure_does_not_publish_partial_population(tmp_path, monkeypatch):
    model, runtime = build_runtime(tmp_path)
    try:
        original_write = runtime.store._write_page

        def fail_commit(page, *args):
            if page.key.endswith("/committed"):
                raise OSError("injected commit persistence failure")
            return original_write(page, *args)

        monkeypatch.setattr(runtime.store, "_write_page", fail_commit)
        actual = prefill(model, runtime, plan(operation="populate"))
        assert torch.isfinite(actual).all()
        assert runtime.last_metrics["fallback"]
        assert "commit marker" in runtime.last_metrics["reason"]
        assert not runtime.store.keys()
    finally:
        runtime.close()


def test_malformed_safetensors_snapshot_falls_back_before_sampling(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        request = plan(operation="populate")
        expected = prefill(model, runtime, request)
        corrupted_key = runtime._key(request, 0, "kv")
        runtime.store.delete(corrupted_key)
        # The storage checksum is valid; the tensor serialization is malformed.
        runtime.store.put(corrupted_key, b"invalid safetensors payload")
        runtime.store.submit_spill(corrupted_key).result(timeout=3)
        actual = prefill(model, runtime, plan(operation="reuse"))
        torch.testing.assert_close(actual, expected)
        assert runtime.last_metrics["fallback"]
        assert not runtime.last_metrics["cache_hit"]
    finally:
        runtime.close()


def test_restart_recovers_only_exact_known_uncommitted_layer_keys(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        request = plan(operation="populate")
        orphan = runtime._key(request, 0, "state")
        unrelated = orphan + "/unrelated-user-record"
        for key in (orphan, unrelated):
            runtime.store.put(key, b"old partial payload")
            runtime.store.submit_spill(key).result(timeout=3)
        settings, bundle = runtime.settings, runtime.bundle
        runtime.close()
        runtime = CacheSlideRuntime(settings, bundle, model_dtype=torch.float32)
        for layer in model.layers:
            layer.self_attn.attention_handler = runtime.attention
        actual = prefill(model, runtime, request)
        assert torch.isfinite(actual).all()
        assert not runtime.last_metrics["fallback"]
        assert runtime.last_metrics["recovered_orphan_pages"] == 1
        assert runtime.store.contains(runtime._commit_key(request))
        assert runtime.store.read(unrelated) == b"old partial payload"
        assert runtime.store.read(orphan) != b"old partial payload"
        prefill(model, runtime, plan(operation="reuse"))
        assert runtime.last_metrics["cache_hit"]
    finally:
        runtime.close()


def test_incomplete_committed_generation_is_not_deleted_by_recovery(tmp_path):
    model, runtime = build_runtime(tmp_path)
    try:
        request = plan(operation="populate")
        prefill(model, runtime, request)
        runtime.store.delete(runtime._key(request, 2, "state"))
        remaining = runtime.store.keys()
        actual = prefill(model, runtime, request)
        assert torch.isfinite(actual).all()
        assert runtime.last_metrics["fallback"]
        assert runtime.store.keys() == remaining
        assert runtime.store.contains(runtime._commit_key(request))
    finally:
        runtime.close()


def test_dtype_is_part_of_persistent_cache_identity(tmp_path):
    _, runtime = build_runtime(tmp_path)
    try:
        other_settings = replace(
            runtime.settings, cache_root=str(tmp_path / "other-cache")
        )
        other = CacheSlideRuntime(
            other_settings, runtime.bundle, model_dtype=torch.float64
        )
        try:
            assert runtime.identity != other.identity
            assert runtime._key(plan(), 0, "kv") != other._key(plan(), 0, "kv")
        finally:
            other.close()
        with pytest.raises(ValueError, match="model_dtype"):
            CacheSlideRuntime(other_settings, runtime.bundle, model_dtype="float32")
    finally:
        runtime.close()


def test_cleanup_io_failure_defers_orphans_without_aborting_recompute(
    tmp_path, monkeypatch
):
    model, runtime = build_runtime(tmp_path)
    try:
        with monkeypatch.context() as patch:

            def unavailable(*_):
                raise OSError("injected cache I/O outage")

            patch.setattr(runtime.store, "_write_page", unavailable)
            patch.setattr(runtime.store, "delete", unavailable)
            actual = prefill(model, runtime, plan(operation="populate"))
            assert torch.isfinite(actual).all()
            assert runtime.last_metrics["fallback"]
            assert runtime.requests["r"].created_keys
        runtime.release("r")
        assert not runtime.store.keys()
    finally:
        runtime.close()
