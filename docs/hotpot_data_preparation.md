# Genuine HotpotQA preparation for an engineering reproduction run

`scripts/prepare_paper_data.py` converts **verified official source records**, not
fabricated prompts, into training and held-out evaluation inputs. This initial
protocol is a **dataset-derived RPDC replay**, not a running Reflexion agent or
the paper's unpublished original traces. It invents no chain-of-thought,
observations or tool responses.

The accepted sources are pinned to HotpotQA distractor revision
`1908d6afbbead072334abe2965f91bd2709910ab` and the local Mistral-7B-Instruct-v0.2
tokenizer revision `63a8b081895390a26e140280378bc85ec8bce07a`. The checkpoint is an
engineering candidate; the paper does not specify this precise Mistral release.
The script reads the existing downloader's `download_manifest.json` and verifies
the complete size and official SHA256/Git-blob digest of both training Parquet
shards, the validation shard and tokenizer JSON before parsing them. A `.part`
file or wrong digest is rejected. It does not need model weight downloads to be
complete and does not read incomplete model weights.

Install the offline preparation dependencies in the selected isolated
environment (`pyarrow` and `tokenizers`), then run:

```bash
python scripts/prepare_paper_data.py \
  --asset-root /assets/cacheslide-paper-assets \
  --output /artifacts/new-hotpot-data \
  --training-count 128 --nll-validation-count 16 --calibration-count 8 \
  --evaluation-count 16 --seed 9 --max-prompt-tokens 8192
```

These are **engineering pilot counts**, not counts recovered from the paper.
Every count is explicit. Official training IDs are partitioned by a seeded
SHA256 into 80% adapter training, 10% held-out next-token NLL validation, and 10%
independent calibration. Each requested subset is the smallest seeded case-ID
hashes from its partition, independent of answers and source shard order.
Accuracy evaluation uses only the official validation split. All official
train/validation IDs are checked for overlap. Selected duplicate content is
rejected. There is no automatic truncation, short-case replacement, or answer
dependent filtering: a selected over-capacity case stops preparation.

## Artifacts

| File | Purpose |
| --- | --- |
| `train.corpus.jsonl` | Source context, question and gold answer, as full parent-document `token_ids` plus `document_sha256` for LoRA/CoPE training. |
| `nll_validation.corpus.jsonl` | Gold-bearing teacher-forced NLL validation from a disjoint official-train partition. Never the accuracy evaluation split. |
| `train.provenance.jsonl` | Identical training token IDs under `prompt_token_ids` for report leakage auditing. |
| `calibration.inputs.jsonl` | Question-bearing, answer-free calibration from the disjoint official-train partition. |
| `evaluation.inputs.jsonl` | Held-out official-validation prompts and exact reuse/recompute chunks, **no appended gold answers**. |
| `evaluation.references.jsonl` | The same prompt IDs plus actual source gold answers for official-semantic `hotpotqa_f1` answer scoring. |
| `population.inputs.jsonl` | Held-out reference documents with a question-free generic prefix for offline cache population. |
| `profile.inputs.jsonl` | The same question/answer-free document template for context-specific CCPE profiles. |
| `source_selection.jsonl` | Source IDs, assignments, parent-document hashes, lengths and fixed-context position shifts. |
| `preparation.json` | Source verification, protocol limitations, counts and every generated file's SHA256. |

Full parent documents in the two `.corpus.jsonl` files must pass through the
training pipeline's `prepare` stage before fixed-context training. For example,
a 512-token training context needs at most 513 token IDs per loss window; the
preparation script itself does not discard the remainder of long source
documents. The parent hash permits downstream train/validation leakage checks
after windowing.

## Non-prefix reuse and context-only profiles

The input begins with a recomputed instruction/question prefix, followed by the
ordered reusable source documents, followed by a recomputed answer marker.
Mistral `[INST]` formatting is explicit, BOS is inserted once, and each chunk is
independently tokenized with `add_special_tokens=False`. The exact resulting
token IDs, not a later concatenated-text re-tokenization, are passed to both
baseline and CacheSlide. This keeps fixed chunk contents identical while the
question-free population prefix and real question prefix have different token
lengths. The actual shift is recorded and zero-shift selections fail validation.

CCPE profile keys bind the precise fixed document layout. A profile from an
unrelated training document cannot calibrate a held-out document's cache.
Therefore `profile.inputs.jsonl` is explicitly **context-only offline cache
preparation**, not adapter training or answer-bearing calibration: it contains
the evaluation documents, but neither the held-out question nor its answer
label. This access is normal for a precomputed reusable-document cache and must
be disclosed. `calibration.inputs.jsonl` remains available for independent
training-split calibration diagnostics; it does not replace those exact-layout
profiles.

Current strict contextual validation can reject the shifted cache path and
recompute. Such a run must be recorded as a fallback, **not** a cache hit or a
paper speedup. The dataset-preparation script does not weaken this guard.

For answer reporting, use `evaluation.references.jsonl`,
`train.provenance.jsonl`, and the actual `profile.inputs.jsonl` supplied to the
experiment as described in [reporting](reproduction_reporting.md). Preserve
both raw generations and labels. The independently implemented `hotpotqa_f1`
metric matches the verified official evaluator's categorical-answer and empty
answer rules; `hotpotqa_em` is also available. It does not claim supporting-fact,
joint or Reflexion success metrics. Full generated text is scored without an
implicit final-answer extractor. Any task-specific extraction must be separately
audited and retained alongside its raw generation, and the paper's exact agent
protocol is still unavailable.
