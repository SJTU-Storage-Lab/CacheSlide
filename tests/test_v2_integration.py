"""CPU lifecycle tests plus executable, fingerprinted native V2 source contracts."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from cacheslide_vllm.compat import validate_native_model_config, verify_vllm_sources
from cacheslide_vllm.integration import (
    StepContext,
    attach_runner,
    current_step,
    step_scope,
)


def request(req_id="r", prompt=(1, 2), suffix=(), **overrides):
    fields = dict(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        prefill_token_ids=list((*prompt, *suffix)),
        sampling_params=SimpleNamespace(
            extra_args={"cacheslide": {"generation": 1}},
            prompt_logprobs=None,
            max_tokens=8,
        ),
        num_computed_tokens=0,
        mm_features=[],
        prompt_embeds=None,
        lora_request=None,
        block_ids=([0],),
        prompt_len=len(prompt),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def schedule(
    *, new=(), positions=(), ids=(), finished=(), preempted=(), req_ids=("r",)
):
    batch = SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=len(req_ids),
        num_tokens=len(positions),
        num_tokens_after_padding=len(positions),
        num_draft_tokens=0,
        positions=torch.tensor(positions, dtype=torch.long),
        input_ids=torch.tensor(ids, dtype=torch.long),
    )
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        total_num_scheduled_tokens=len(positions),
        batch=batch,
    )


class Model:
    def __init__(self):
        self.observed = []
        self.released = []
        self.fail = False

    def __call__(self, *, input_ids, positions):
        self.observed.append(current_step())
        if self.fail:
            raise RuntimeError("injected model error")
        return torch.zeros(len(positions), 3)

    def cacheslide_release_request(self, request_id):
        self.released.append(request_id)


class Runner:
    """No CUDA imitation: just the audited eager call order under test."""

    def __init__(self):
        self.model = Model()
        self.native_ids = set()

    def get_model(self):
        return self.model

    def _remove_request(self, req_id):
        existed = req_id in self.native_ids
        self.native_ids.discard(req_id)
        return existed

    def finish_requests(self, scheduler_output):
        for req_id in (
            scheduler_output.finished_req_ids | scheduler_output.preempted_req_ids
        ):
            self._remove_request(req_id)

    def add_requests(self, scheduler_output):
        for item in scheduler_output.scheduled_new_reqs:
            self._remove_request(item.req_id)
            self.native_ids.add(item.req_id)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        return scheduler_output.batch

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        context_len=0,
    ):
        if dummy_run:
            return self.model(input_ids=torch.zeros(2), positions=torch.zeros(2))
        self.finish_requests(scheduler_output)
        self.add_requests(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            return None
        batch = self.prepare_inputs(scheduler_output, None, None)
        return self.model(input_ids=batch.input_ids, positions=batch.positions)


def test_v2_request_snapshots_prefill_decode_and_cleanup_are_instance_scoped():
    runner, untouched = Runner(), Runner()
    original = Runner.execute_model
    attach_runner(runner, use_v2=True)
    attach_runner(runner, use_v2=True)
    item = request()
    assert runner.execute_model(
        schedule(new=[item], positions=(0, 1), ids=(1, 2))
    ).shape == (2, 3)
    item.sampling_params.extra_args["cacheslide"]["generation"] = 100
    runner.execute_model(schedule(positions=(2,), ids=(3,)))
    prefill, decode = runner.model.observed
    assert prefill.positions == (0, 1)
    assert decode.positions == (2,) and decode.replay_token_ids is None
    assert decode.extra_args["cacheslide"]["generation"] == 1
    assert current_step() is None
    runner.execute_model(schedule(finished=["r"]))
    assert runner.model.released[-1] == "r" and not runner.native_ids
    assert Runner.execute_model is original
    assert not hasattr(untouched, "_cacheslide_adapter_installed")


def test_v2_preemption_retires_old_state_and_carries_exact_generated_suffix():
    runner = Runner()
    attach_runner(runner)
    runner.execute_model(schedule(new=[request()], positions=(0, 1), ids=(1, 2)))
    runner.execute_model(schedule(preempted=["r"]))
    assert runner.model.released[-1] == "r"
    runner.execute_model(
        schedule(new=[request(suffix=(3, 4))], positions=range(4), ids=(1, 2, 3, 4))
    )
    step = runner.model.observed[-1]
    assert step.prompt_token_ids == (1, 2)
    assert step.replay_token_ids == (1, 2, 3, 4)
    runner.execute_model(schedule(positions=(4,), ids=(5,)))
    assert runner.model.observed[-1].replay_token_ids is None


def test_v2_hooks_drive_real_trained_runtime_through_reuse_replay_and_decode(tmp_path):
    from tests.test_runtime import build_runtime, plan

    model, runtime = build_runtime(tmp_path)

    class RuntimeModel:
        @torch.inference_mode()
        def __call__(self, *, input_ids, positions):
            return runtime.run(
                model, model.embed_tokens(input_ids), positions, current_step()
            )

        def cacheslide_release_request(self, request_id):
            runtime.release(request_id)

    def new_plan(operation, suffix=()):
        item = plan(operation=operation)
        return request(
            prompt=item.token_ids,
            suffix=suffix,
            sampling_params=SimpleNamespace(
                extra_args={"cacheslide": item.to_json()},
                prompt_logprobs=None,
                max_tokens=8,
            ),
        )

    runner = Runner()
    runner.model = RuntimeModel()
    attach_runner(runner)
    try:
        tokens = plan().token_ids
        runner.execute_model(
            schedule(
                new=[new_plan("populate")], positions=range(len(tokens)), ids=tokens
            )
        )
        runner.execute_model(schedule(finished=["r"]))
        runner.execute_model(
            schedule(new=[new_plan("reuse")], positions=range(len(tokens)), ids=tokens)
        )
        assert runtime.last_metrics["cache_hit"]
        old = runtime.requests["r"]
        runner.execute_model(schedule(positions=(len(tokens),), ids=(3,)))
        runner.execute_model(schedule(preempted=["r"]))
        assert "r" not in runtime.requests
        replay = (*tokens, 3, 4)
        result = runner.execute_model(
            schedule(
                new=[new_plan("reuse", (3, 4))],
                positions=range(len(replay)),
                ids=replay,
            )
        )
        assert result.shape == (len(replay), 8) and torch.isfinite(result).all()
        assert runtime.requests["r"] is not old
        assert (
            runtime.last_metrics["fallback"] and not runtime.last_metrics["cache_hit"]
        )
        assert runtime.last_metrics["decode_tokens"] == 2
        runner.execute_model(schedule(positions=(len(replay),), ids=(5,)))
        assert runtime.last_metrics["decode_tokens"] == 3
    finally:
        runtime.close()


def test_v2_replay_rejects_changed_generated_token_before_runtime_entry():
    runner = Runner()
    attach_runner(runner)
    with pytest.raises(ValueError, match="input IDs disagree"):
        runner.execute_model(
            schedule(new=[request(suffix=(3, 4))], positions=range(4), ids=(1, 2, 3, 9))
        )
    assert runner.model.observed == [] and current_step() is None


def test_v2_dummy_and_forward_errors_restore_context_without_creating_dummy_request():
    runner = Runner()
    attach_runner(runner)
    outer = StepContext("outer", (7,), (0,))
    with step_scope(outer):
        runner.execute_model(schedule(), dummy_run=True)
        assert current_step() is outer
        assert runner.model.observed == [None] and not runner.native_ids
        runner.model.fail = True
        with pytest.raises(RuntimeError, match="injected"):
            runner.execute_model(
                schedule(new=[request()], positions=(0, 1), ids=(1, 2))
            )
        assert current_step() is outer
        runner.model.fail = False
        runner.execute_model(schedule(positions=(0, 1), ids=(1, 2)))
    assert current_step() is None


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("prompt_embeds", torch.zeros(2, 3), "plain token"),
        ("mm_features", [object()], "plain token"),
        ("lora_request", object(), "plain token"),
        ("num_computed_tokens", 1, "complete uncached"),
        ("prefill_token_ids", [9, 2], "complete uncached"),
    ],
)
def test_v2_rejects_unsupported_new_requests_before_native_mutation(
    field, value, message
):
    runner = Runner()
    attach_runner(runner)
    with pytest.raises(ValueError, match=message):
        runner.execute_model(
            schedule(new=[request(**{field: value})], positions=(0, 1), ids=(1, 2))
        )
    assert not runner.native_ids and not runner.model.observed
    assert current_step() is None


@pytest.mark.parametrize("kind", ["padding", "draft", "multiple", "short", "wrong_ids"])
def test_v2_rejects_invalid_scheduled_rows_and_clears_context(kind):
    runner = Runner()
    attach_runner(runner)
    event = schedule(new=[request()], positions=(0, 1), ids=(1, 2))
    if kind == "padding":
        event.batch.num_tokens_after_padding = 3
    elif kind == "draft":
        event.batch.num_draft_tokens = 1
    elif kind == "multiple":
        event.batch.req_ids = ["r", "other"]
    elif kind == "short":
        event.batch.positions = torch.tensor([1])
        event.batch.input_ids = torch.tensor([2])
        event.batch.num_tokens = event.batch.num_tokens_after_padding = 1
    else:
        event.batch.input_ids = torch.tensor([2, 1])
    with pytest.raises(ValueError):
        runner.execute_model(event)
    assert current_step() is None and not runner.model.observed


def test_runner_interface_must_match_configured_version():
    with pytest.raises(ValueError, match="configured interface"):
        attach_runner(Runner(), use_v2=False)
    with pytest.raises(ValueError, match="identify"):
        attach_runner(SimpleNamespace())


@pytest.mark.parametrize(
    "replay,positions",
    [
        ((1, 2), (0, 1)),
        ((1, 3, 4), (0, 1, 2)),
        ((1, 2, 3), (1, 2, 3)),
        ((1, 2, True), (0, 1, 2)),
    ],
)
def test_replay_context_requires_full_valid_history(replay, positions):
    with pytest.raises(ValueError, match="replay"):
        StepContext("r", (1, 2), positions, replay_token_ids=replay)


@pytest.mark.parametrize(
    "field,value", [("head_dim", 8), ("rms_norm_eps", 1e-5), ("logit_scale", 2.0)]
)
def test_native_hf_overrides_cannot_change_trained_numerics(field, value):
    trained = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=16,
        head_dim=4,
    )
    native = SimpleNamespace(**{k: v for k, v in trained.items() if k != "head_dim"})
    validate_native_model_config(native, trained)
    native.head_dim = None
    validate_native_model_config(native, trained)
    setattr(native, field, value)
    with pytest.raises(ValueError, match=field):
        validate_native_model_config(native, trained)


@pytest.fixture
def native_v2():
    source = os.environ.get("CACHESLIDE_VLLM_SOURCE")
    if not source:
        pytest.skip(
            "Set CACHESLIDE_VLLM_SOURCE for native V2 source execution contracts"
        )
    root = Path(source)
    verify_vllm_sources(root, version="0.29.0")
    path = root / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    return methods


def test_native_v2_ast_proves_real_eager_scope_and_cleanup_hook_points(native_v2):
    execute = native_v2["execute_model"]
    assert [arg.arg for arg in execute.args.args] == [
        "self",
        "scheduler_output",
        "intermediate_tensors",
        "dummy_run",
        "skip_attn_for_dummy_run",
        "is_profile",
        "context_len",
    ]
    calls = {
        name: []
        for name in ("finish_requests", "add_requests", "prepare_inputs", "model")
    }
    for node in ast.walk(execute):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in calls
        ):
            calls[node.func.attr].append(node.lineno)
    assert (
        max(calls["finish_requests"])
        < min(calls["add_requests"])
        < min(calls["prepare_inputs"])
        < min(calls["model"])
    )
    scope = next(
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "set_forward_context"
            for item in node.items
        )
    )
    assert "self.model(**model_inputs)" in ast.unparse(scope)
    assert "slot_mapping=slot_mappings_by_layer" in ast.unparse(scope)
    assert "preempted_req_ids" in ast.unparse(native_v2["finish_requests"])
    assert "self._remove_request(req_id)" in ast.unparse(native_v2["add_requests"])
    assert "all_token_ids=new_req_data.prefill_token_ids" in ast.unparse(
        native_v2["add_requests"]
    )


def test_actual_native_v2_add_remove_finish_methods_preserve_adapter_replay(native_v2):
    """Execute three complete upstream methods, not handwritten cleanup doubles."""
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + [
            native_v2[name]
            for name in ("_remove_request", "finish_requests", "add_requests")
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(module), "native-v2-lifecycle", "exec"),
        namespace,
    )
    runner = Runner()
    for name in ("_remove_request", "finish_requests", "add_requests"):
        setattr(runner, name, MethodType(namespace[name], runner))

    def noop(*args, **kwargs):
        return None

    class ReqStates:
        def __init__(self):
            self.req_id_to_index = {}

        def add_request(self, req_id, **kwargs):
            self.req_id_to_index[req_id] = 0

        def remove_request(self, req_id):
            return self.req_id_to_index.pop(req_id, None)

        apply_staged_writes = staticmethod(noop)

    runner.req_states = ReqStates()
    runner.model_state = SimpleNamespace(
        add_request=noop, remove_request=noop, apply_staged_writes=noop
    )
    runner.lora_state = SimpleNamespace(add_request=noop, remove_request=noop)
    runner.block_tables = SimpleNamespace(append_block_ids=noop)
    runner.adaptive_verification = runner.pooling_runner = runner.pp_handler = (
        runner.encoder_cache
    ) = None
    runner.is_last_pp_rank = True
    runner.sampler = SimpleNamespace(add_request=noop, apply_staged_writes=noop)
    runner.prompt_logprobs_worker = SimpleNamespace(
        add_request=noop, remove_request=noop
    )
    attach_runner(runner, use_v2=True)
    runner.execute_model(schedule(new=[request()], positions=(0, 1), ids=(1, 2)))
    assert runner.req_states.req_id_to_index == {"r": 0}
    runner.execute_model(schedule(preempted=["r"]))
    assert runner.req_states.req_id_to_index == {} and runner.model.released[-1] == "r"
    runner.execute_model(
        schedule(new=[request(suffix=(3,))], positions=(0, 1, 2), ids=(1, 2, 3))
    )
    assert runner.model.observed[-1].replay_token_ids == (1, 2, 3)
    runner.execute_model(schedule(finished=["r"]))
    assert not runner.req_states.req_id_to_index
