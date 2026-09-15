from dataclasses import replace

import pytest
import torch

from cacheslide_core.runtime import CacheSlideRuntime
from cacheslide_core.wca import WCANumericalError, WCAState

from .test_runtime import build_runtime, plan, prefill


def test_invalid_hybrid_positions_retry_plain_cope_and_do_not_publish(tmp_path):
    model, runtime = build_runtime(tmp_path, ccpe_position_policy="strict_contextual")
    try:
        request = plan((4, 10), operation="populate")
        actual = prefill(model, runtime, request)
        # The calibrated fixed/fixed projection is not a valid gate path for
        # this longer dynamic span. The fallback must clear ALL layer profiles.
        metrics = runtime.last_metrics
        assert metrics["position_policy_fallback"] and metrics["fallback"]
        assert metrics["ccpe_position_policy"] == "plain_cope_fallback"
        assert metrics["layer_rows"] == [9] * 6
        assert not runtime.requests["r"].profiles
        assert not runtime.store.keys()
        expected = model(torch.tensor(request.token_ids))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    finally:
        runtime.close()


def test_typed_wca_numeric_failure_restarts_dense_before_sampling(
    tmp_path, monkeypatch
):
    model, runtime = build_runtime(tmp_path)
    try:
        prefill(model, runtime, plan(operation="populate"))
        request = plan((4, 10), operation="reuse")
        expected = prefill(model, runtime, replace(request, operation="recompute"))

        def overflow(*args, **kwargs):
            raise WCANumericalError("finite float16 fusion overflow")

        monkeypatch.setattr(WCAState, "update", overflow)
        actual = prefill(model, runtime, request)
        torch.testing.assert_close(actual, expected)
        assert runtime.last_metrics["fallback"]
        assert not runtime.last_metrics["cache_hit"]
        assert not runtime.last_metrics["position_policy_fallback"]
        assert runtime.last_metrics["layer_rows"] == [9] * 6
    finally:
        runtime.close()


def test_wca_schema_error_is_not_hidden_by_dense_retry(tmp_path, monkeypatch):
    model, runtime = build_runtime(tmp_path)
    try:
        prefill(model, runtime, plan(operation="populate"))

        def invalid(*args, **kwargs):
            raise ValueError("invalid caller shape")

        monkeypatch.setattr(WCAState, "update", invalid)
        with pytest.raises(ValueError, match="invalid caller shape"):
            prefill(model, runtime, plan((4, 10), operation="reuse"))
    finally:
        runtime.close()


def test_literal_first_layer_wca_does_not_invent_nonzero_error(tmp_path):
    model, runtime = build_runtime(tmp_path, calibration_layer=0)
    try:
        prefill(model, runtime, plan(operation="populate"))
        prefill(model, runtime, plan((4, 10), operation="reuse"))
        assert runtime.last_metrics["cache_hit"]
        assert runtime.last_metrics["layer_rows"] == [9, 3, 3, 3, 3, 3]
        assert not len(runtime.requests["r"].wca.selected_indices)
    finally:
        runtime.close()


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


def test_engine_execution_identity_separates_persistent_caches(tmp_path):
    _, runtime = build_runtime(tmp_path)
    try:
        other = CacheSlideRuntime(
            replace(runtime.settings, cache_root=str(tmp_path / "sglang-cache")),
            runtime.bundle,
            model_dtype=torch.float32,
            execution_identity="sglang:0.5.19:audited-source",
        )
        try:
            assert runtime.identity != other.identity
            assert runtime._key(plan(), 0, "kv") != other._key(plan(), 0, "kv")
        finally:
            other.close()
        for invalid in (None, "", 19):
            with pytest.raises(ValueError, match="execution_identity"):
                CacheSlideRuntime(
                    runtime.settings,
                    runtime.bundle,
                    model_dtype=torch.float32,
                    execution_identity=invalid,
                )
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
