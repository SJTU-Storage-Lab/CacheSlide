# Paper-result reporting protocol

This is an **offline report of saved real generations**, not an inference
implementation and not a declaration that the FAST '26 results were reproduced.
The current native adapter only supports one active request, batch 1, no beam.
It cannot establish the paper's concurrent, beam-search or throughput figures.

## Three distinct configurations

1. `native_baseline`: the unmodified model's original RoPE or ALiBi, no CoPE
   adapter. This is the accuracy/performance baseline relevant to the paper.
2. `adapter_recompute`: full recomputation with the exact trained CoPE adapter
   used by CacheSlide. This diagnoses cache reuse error and time savings; it is
   **not** the original model baseline.
3. `cacheslide`: the trained CoPE adapter plus CCPE, WCA and the selected storage
   implementation. A valid reuse measurement requires a verified cache hit with
   no fallback.

Run both comparisons and publish both. Model and tokenizer revisions, adapter
SHA256, deterministic generation configuration, dataset revisions and hardware
must be recorded with the experiment. The reporting manifest is a declaration
of these identities; it does not independently hash model weights or prove how
the raw record producer loaded the engine. Preserve the producer logs, artifact
verification and all original generation/RPC records too.

## Held-out evaluation input

The evaluation manifest is JSONL, separate from the engine's strict input JSONL:

```json
{"id":"hotpot-validation-001","dataset":"hotpot_qa","split":"validation","revision":"<immutable-dataset-commit>","prompt_token_ids":[11,12,13],"references":["gold answer"],"metric":"hotpotqa_f1"}
```

The shown token IDs are **schema examples**, not an experiment or language input.
Use the identical tokenizer and exact full prompt token IDs from the real engine
request. Required `--training` and `--calibration` JSONL files contain at least
`id` and `prompt_token_ids`; existing training/calibration input files work.
Every evaluation case ID and full prompt hash must be disjoint from both files.
Repeated evaluation prompts, missing reference labels, and a training split used
as evaluation fail closed. This check does not certify semantic/document-level
split separation. Shared reusable context chunks may legitimately be present in
population/calibration; held-out questions, answers and source split selection
must still be audited.

The paper identifies HotpotQA with Reflexion, Multi-Session Chat with MemGPT, and
SWE-Agent-Bench/SWE-bench, but does not provide enough pinned dataset revisions,
prompt traces or split details to silently recreate an exact evaluation. Record
explicit decisions and label new traces as a reproduction protocol, not the
paper's original traces. Do not substitute a few fabricated prompts for those
datasets.

## Answer metrics

- `hotpotqa_f1` and `hotpotqa_em`: independent implementations of the official
  [HotpotQA answer evaluator](https://github.com/hotpotqa/hotpot/blob/fa3a36370899e1d85822de61e58c85ea19993154/hotpot_evaluate_v1.py),
  verified at that immutable commit. Use one gold answer per example, as in the
  official dataset. Any normalized `yes`/`no`/`noanswer` mismatch on either side
  forces F1 to zero; zero-overlap answers also score zero, including two empty
  answers (their EM is one). Supporting-fact and joint scores are **not**
  computed. This answer score is not an end-to-end Reflexion success rate.
- `qa_f1`: best-reference answer-word F1, with SQuAD-style lowercasing, ASCII
  punctuation/article removal and whitespace tokenization.
- `exact_match`: equality under that same explicit normalization. This is a
  **text answer-match rate**, not an agent task-success or SWE resolution rate.
- `rouge_l_recall`: longest-common-subsequence length divided by the reference
  length, using non-stemmed lowercase Unicode word tokens. This is recall,
  **not ROUGE-L F1**. The paper's exact tokenizer/stemmer was not specified.

All are scored against ground-truth references, not against the recomputed
model's generated tokens. Full generated text is scored; no implicit answer
extraction or truncation is performed. If an official benchmark needs final
answer extraction, provide an independently audited extraction stage and retain
the raw generated text as well as the extracted answer. Metrics should not be
mixed into one overall "accuracy" number. Dataset scores and percentage-point
deltas are reported separately.

SWE-bench `resolved` needs the real repository/environment test harness on each
generated patch. Successful string matching or syntactically valid patches do
not establish resolution. This module therefore leaves SWE resolved rate null.

## Timing and coverage

The existing `raw_outputs.jsonl` from the native workflow can be consumed for the
same-adapter diagnostic. Its `recompute` operation is never interpreted as a
native model baseline. All cases need both methods, all requested repetitions,
and at least one successful warmup per case/method. Missing, duplicate, excess,
fallback or cache-miss measurements cannot be dropped to improve an average.

This branch reads the **actual vLLM CLI record schema**:
`generated_token_ids`, `prompt_token_ids`, `input_tokens`, `output_tokens`,
`request_id`, `finish_reason`, `runtime_metrics`, `metrics_valid` and
`metrics_error`. Full prompt IDs must match the evaluation manifest. The worker
RPC snapshot must bind the same request ID, operation and prompt length, with
no fallback and a true CacheSlide cache hit. Fixed-budget runs must finish by
length and emit exactly `max_new_tokens`, matching the existing CLI's
`ignore_eos=True`. It does not accept SGLang-shaped receipts as vLLM evidence.

The vLLM worker snapshot is **not** SGLang's immutable completion receipt:
it provides no worker output digest or resource-release attestation. These
missing guarantees are explicitly recorded, never manufactured by the report.

`elapsed_seconds` is the original **blocking whole-generation duration**. The
report publishes `mean(baseline) / mean(CacheSlide)` and, separately, the mean of
paired ratios. Population, compilation and warmups do not enter those means.
Use the same loaded engine/settings and alternate measured method order when
collecting a same-adapter diagnostic. A native model comparison needs separately
controlled warm engines and the same prompts/output budget. The report checks
saved coverage; it cannot prove thermal state or the fairness of an upstream
producer's unrecorded engine lifecycle.

TTFT is unavailable from blocking output. A streaming producer may additionally
record this exact structure using one client monotonic clock:

```json
{"first_token_event":{"clock":"client_monotonic","arrival_seconds":100.0,"first_token_seconds":100.8,"source":"stream_token_event","generated_token_count":1}}
```

The timestamp must mark the **first generated token**, not HTTP headers, an empty
SSE chunk, a second token, or the final blocking response. The full-generation
interval must use the same arrival boundary. Unless every measured pair has
valid events, the TTFT ratio remains null. These numerical timestamps illustrate
the schema only; they are not measured data.

Concurrent QPS cannot be obtained as the reciprocal of a serial request latency.
The paper's goodput additionally requires independently judged correct
completions per measured workload second. Both fields remain null here. A future
load experiment must record request arrival/completion times, offered load,
concurrency, full makespan, errors and answer correctness; it must not replace
this evidence with token throughput or an average per-request reciprocal.

## Local reporting command

Create `experiment.json` with real pinned identities:

```json
{
  "model":"<model repository>",
  "model_revision":"<immutable model commit>",
  "tokenizer":"<tokenizer repository>",
  "tokenizer_revision":"<immutable tokenizer commit>",
  "variants":{
    "native_baseline":{"position_encoding":"rope","adapter_sha256":null},
    "adapter_recompute":{"position_encoding":"trained_cope","adapter_sha256":"<64 lowercase hex digits>"},
    "cacheslide":{"position_encoding":"trained_cope","adapter_sha256":"<same 64 lowercase hex digits>"}
  },
  "generation":{"temperature":0,"batch_size":1,"beam_width":1,"max_new_tokens":256}
}
```

```bash
python -m cacheslide_vllm.reproduction_report \
  --records /results/native/raw_outputs.jsonl \
  --evaluation /data/held_out_with_references.jsonl \
  --training /data/train.jsonl --calibration /data/calibration.jsonl \
  --manifest /results/experiment.json --tokenizer /models/local-tokenizer \
  --baseline adapter_recompute --repeats 3 --warmup-min 1 \
  --output /results/new-answer-report
```

The output directory must be new. The CLI uses `local_files_only=True` and
`trust_remote_code=False`; it does not download or execute remote model code.
`transformers` is required only to run the CLI tokenizer, not to import or test
the report and metrics functions. Outputs include full paired decoded text,
token IDs, references, per-case/dataset scores, validation errors and SHA256s of
every source input file. Failure returns exit code 2; valid limited reporting
returns 0 and still explicitly sets `paper_results_reproduced: false`.

Explicit native-baseline records must have `variant: native_baseline`,
`status: complete`, and the same full prompt/generated IDs, request identity,
length completion, phase/case/repeat/timing fields as the vLLM record schema.
They need a separate genuine original-model producer, not a CacheSlide worker
snapshot. Do not relabel CacheSlide's
adapter recompute records as native baseline. Training and calibration costs,
cache population latency/bytes, and SSD hardware write amplification need
separate measured artifacts and are not filled in by this report.
