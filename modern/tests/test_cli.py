import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from cacheslide_vllm import cli


def row(tokens=(1, 2, 3, 4), case_id="case"):
    return {
        "id": case_id,
        "prompt_token_ids": list(tokens),
        "cacheslide": {
            "version": 1,
            "operation": "reuse",
            "namespace": "test",
            "task_id": "qa",
            "chunks": [
                {"id": "context", "role": "reuse", "start": 0, "end": 1},
                {"id": "query", "role": "recompute", "start": 1, "end": len(tokens)},
            ],
        },
    }


def input_file(tmp_path, rows=None):
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in (rows or [row()])) + "\n")
    return path


def native_args(tmp_path, command="bench", extra=()):
    path = input_file(tmp_path)
    return [
        command,
        "--model",
        str(tmp_path / "model"),
        "--adapter",
        str(tmp_path / "adapter"),
        "--profiles",
        str(tmp_path / "profiles"),
        "--input",
        str(path),
        "--output",
        str(tmp_path / "results"),
        "--cache-root",
        str(tmp_path / "cache"),
        *extra,
    ]


def test_token_metadata_contract_and_training_rows(tmp_path):
    path = input_file(tmp_path)
    cases = cli.read_cases(path)
    assert cases[0].token_ids == (1, 2, 3, 4)
    assert cases[0].plan.fixed_indices == (0,)
    path.write_text(json.dumps({"prompt_token_ids": [1, 2]}) + "\n")
    assert cli.read_cases(path, require_plan=False)[0].plan is None
    with pytest.raises(ValueError, match="chunk metadata"):
        cli.read_cases(path)


@pytest.mark.parametrize(
    "invalid",
    [
        {"prompt_token_ids": [True, 2]},
        {"prompt_token_ids": [1.0, 2]},
        {"prompt_token_ids": [-1, 2]},
        {"prompt_token_ids": []},
        {"prompt_token_ids": [1, 2], "unknown": 3},
    ],
)
def test_invalid_typed_input_is_rejected(tmp_path, invalid):
    path = input_file(tmp_path, [invalid])
    with pytest.raises(ValueError):
        cli.read_cases(path, require_plan=False)


def test_duplicate_json_and_case_ids_are_rejected(tmp_path):
    path = input_file(tmp_path, [row(), row()])
    with pytest.raises(ValueError, match="case ids"):
        cli.read_cases(path)
    path.write_text('{"prompt_token_ids":[1,2],"prompt_token_ids":[3,4]}\n')
    with pytest.raises(ValueError, match="duplicate JSON"):
        cli.read_cases(path, require_plan=False)


def test_native_configuration_is_eager_single_request_and_no_gpu_by_default(
    tmp_path, monkeypatch, capsys
):
    args = native_args(tmp_path, extra=("--profiles", str(tmp_path / "profiles")))
    parsed = cli.parser().parse_args(args)
    kwargs = cli.engine_kwargs(parsed)
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["max_num_batched_tokens"] >= kwargs["max_model_len"]
    assert kwargs["enforce_eager"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["async_scheduling"] is False
    assert kwargs["compilation_config"] == {"mode": 0, "cudagraph_mode": "NONE"}
    assert kwargs["attention_config"] == {"backend": "FLASH_ATTN"}
    assert kwargs["hf_overrides"]["architectures"] == ["CacheSlideLlamaForCausalLM"]
    assert kwargs["worker_cls"] == "cacheslide_vllm.worker.CacheSlideWorker"
    assert kwargs["additional_config"]["cacheslide"]["profile_path"] == str(
        tmp_path / "profiles"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("native engine was constructed without --run")

    monkeypatch.setattr(cli, "create_engine", forbidden)
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["executed"] is False and result["requires"] == "--run"
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "cache").exists()


def test_create_engine_requires_explicit_run(tmp_path):
    args = cli.parser().parse_args(native_args(tmp_path, command="generate"))
    with pytest.raises(ValueError, match="explicit --run"):
        cli.create_engine(args)


def test_benchmark_pairs_lengths_excludes_warmup_and_saves_raw_outputs(
    tmp_path, monkeypatch, capsys
):
    args = native_args(
        tmp_path, extra=("--run", "--repeats", "2", "--warmup", "1", "--token-f1")
    )
    input_file(tmp_path, [row(case_id="short"), row((1, 2, 3, 4, 5), case_id="long")])
    clock = SimpleNamespace(value=0.0, calls=0, closed=False, receipt=None)

    class Sampling:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Engine:
        def generate(self, prompt, sampling_params, use_tqdm):
            assert not use_tqdm and sampling_params.detokenize is False
            assert sampling_params.ignore_eos is True
            operation = sampling_params.extra_args["cacheslide"]["operation"]
            clock.calls += 1
            clock.receipt = {
                "request_id": str(clock.calls),
                "operation": operation,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "cache_hit": operation == "reuse",
                "fallback": False,
            }
            # Seed + four warmup requests must not affect measured means.
            clock.value += (
                100 if clock.calls <= 5 else (4 if operation == "recompute" else 2)
            )
            return [
                SimpleNamespace(
                    request_id=str(clock.calls),
                    outputs=[SimpleNamespace(token_ids=[8], finish_reason="length")],
                )
            ]

        def collective_rpc(self, method):
            if method == "cacheslide_close":
                clock.closed = True
                return [None]
            assert method == "cacheslide_metrics"
            clock.value += 1000  # Receipt time must stay outside measured latency.
            return [clock.receipt]

    monkeypatch.setattr(cli, "create_engine", lambda args: (Engine(), Sampling))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock.value)
    assert cli.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["measured_pairs"] == 4
    assert summary["baseline_mean_seconds"] == 4
    assert summary["reuse_mean_seconds"] == 2
    assert summary["baseline_over_reuse_latency_ratio"] == 2
    assert summary["reuse_validation_passed"] is True
    assert summary["baseline_validation_passed"] is True
    assert clock.closed
    assert summary["exact_generated_token_agreement_rate"] == 1
    assert summary["mean_token_id_multiset_f1"] == 1
    assert summary["is_streaming_ttft"] is False
    assert summary["timing_kind"] == "offline_prefill_plus_one_token_latency_seconds"
    assert [pair["input_tokens"] for pair in summary["pairs"]] == [4, 5, 4, 5]
    raw = [
        json.loads(line)
        for line in (tmp_path / "results/raw_outputs.jsonl").read_text().splitlines()
    ]
    assert len(raw) == 13
    assert sum(record["phase"] == "measured" for record in raw) == 8
    assert raw[0]["phase"] == "setup" and raw[0]["operation"] == "populate"
    assert json.loads((tmp_path / "results/summary.json").read_text()) == summary


def test_token_f1_counts_multiplicity_and_explicitly_ignores_order():
    assert cli._token_f1([1, 1, 2], [1, 2, 2]) == pytest.approx(2 / 3)
    assert cli._token_f1([1, 2], [2, 1]) == 1
    assert cli._token_f1([], []) == 1
    assert cli._token_f1([], [1]) == 0


@pytest.mark.parametrize("failure", ["miss", "fallback", "stale", "zero_time"])
def test_benchmark_suppresses_unvalidated_ratio(tmp_path, monkeypatch, capsys, failure):
    args = native_args(tmp_path, extra=("--run", "--repeats", "1", "--warmup", "0"))
    state = SimpleNamespace(receipt=None, calls=0, closed=False)

    class Engine:
        def generate(self, prompt, sampling_params, use_tqdm):
            state.calls += 1
            operation = sampling_params.extra_args["cacheslide"]["operation"]
            state.receipt = {
                "request_id": str(state.calls),
                "operation": operation,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "cache_hit": operation == "reuse",
                "fallback": False,
            }
            if operation == "reuse":
                if failure == "miss":
                    state.receipt["cache_hit"] = False
                elif failure == "fallback":
                    state.receipt["fallback"] = True
                elif failure == "stale":
                    state.receipt["request_id"] = "old-request"
            return [
                SimpleNamespace(
                    request_id=str(state.calls),
                    outputs=[SimpleNamespace(token_ids=[8], finish_reason="length")],
                )
            ]

        def collective_rpc(self, method):
            if method == "cacheslide_close":
                state.closed = True
                return [None]
            return [state.receipt]

    monkeypatch.setattr(
        cli,
        "create_engine",
        lambda args: (Engine(), lambda **kw: SimpleNamespace(**kw)),
    )
    if failure == "zero_time":
        monkeypatch.setattr(cli.time, "perf_counter", lambda: 1.0)
    assert cli.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["baseline_over_reuse_latency_ratio"] is None
    assert summary["reuse_validation_passed"] is (failure == "zero_time")
    assert state.closed


def test_benchmark_requires_profiles_before_engine(tmp_path, monkeypatch, capsys):
    args = native_args(tmp_path, extra=("--run",))
    index = args.index("--profiles")
    del args[index : index + 2]
    monkeypatch.setattr(
        cli, "create_engine", lambda args: pytest.fail("engine started")
    )
    assert cli.main(args) == 2
    assert "requires --profiles" in capsys.readouterr().err


def test_engine_session_drains_store_after_generation_error(
    tmp_path, monkeypatch, capsys
):
    closed = []

    class Engine:
        def generate(self, *args, **kwargs):
            raise ValueError("generation failed")

        def collective_rpc(self, method):
            closed.append(method)
            return [None]

    monkeypatch.setattr(
        cli,
        "create_engine",
        lambda args: (Engine(), lambda **kw: SimpleNamespace(**kw)),
    )
    assert cli.main(native_args(tmp_path, "generate", extra=("--run",))) == 2
    assert "generation failed" in capsys.readouterr().err
    assert closed == ["cacheslide_close"]


def test_reference_backend_and_oversized_requests_fail_before_engine(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported request reached native engine construction")

    monkeypatch.setattr(cli, "create_engine", forbidden)
    args = native_args(tmp_path, extra=("--run", "--backend", "reference"))
    assert cli.main(args) == 2
    assert "not connected" in capsys.readouterr().err
    args = native_args(tmp_path, extra=("--run", "--max-model-len", "4"))
    assert cli.main(args) == 2
    assert "max-model-len" in capsys.readouterr().err


def test_check_engine_and_inspect_are_read_only(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "verify_vllm_sources",
        lambda path, version: (
            calls.append((path, version)) or cli.compatibility_manifest()
        ),
    )
    assert (
        cli.main(
            ["check-engine", "--source-root", str(tmp_path), "--version", "0.29.0"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert calls == [(str(tmp_path), "0.29.0")]
    assert cli.main(["inspect"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["compatibility"]["vllm_version"] == "0.29.0"
    )


def test_cli_real_cpu_training_calibration_and_inspection(tmp_path, capsys):
    model = tmp_path / "model"
    model.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
    }
    (model / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(2)
    weights = {
        "model.embed_tokens.weight": torch.randn(8, 4, generator=generator),
        "lm_head.weight": torch.randn(8, 4, generator=generator),
        "model.norm.weight": torch.ones(4),
    }
    prefix = "model.layers.0."
    for name in ("input_layernorm", "post_attention_layernorm"):
        weights[prefix + name + ".weight"] = torch.ones(4)
    for name, shape in {
        "self_attn.q_proj": (4, 4),
        "self_attn.k_proj": (4, 4),
        "self_attn.v_proj": (4, 4),
        "self_attn.o_proj": (4, 4),
        "mlp.gate_proj": (8, 4),
        "mlp.up_proj": (8, 4),
        "mlp.down_proj": (4, 8),
    }.items():
        weights[prefix + name + ".weight"] = (
            torch.randn(shape, generator=generator) * 0.1
        )
    save_file(weights, str(model / "model.safetensors"))
    path = input_file(tmp_path)
    adapter, profiles = tmp_path / "adapter", tmp_path / "profiles"
    assert (
        cli.main(
            [
                "train",
                "--model",
                str(model),
                "--input",
                str(path),
                "--output",
                str(adapter),
                "--steps",
                "2",
                "--rank",
                "2",
                "--max-positions",
                "8",
            ]
        )
        == 0
    )
    trained = json.loads(capsys.readouterr().out)
    assert trained["training_steps"] == 2 and trained["training_tokens"] == 6
    assert (
        cli.main(
            [
                "calibrate",
                "--model",
                str(model),
                "--adapter",
                str(adapter),
                "--input",
                str(path),
                "--output",
                str(profiles),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["calibration_requests"] == 1
    assert (
        cli.main(
            [
                "inspect",
                "--model",
                str(model),
                "--adapter",
                str(adapter),
                "--profiles",
                str(profiles),
            ]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["adapter"]["metadata"]["trained"] is True
    assert inspected["profiles"]["metadata"]["format"] == "cacheslide-profiles-v1"
