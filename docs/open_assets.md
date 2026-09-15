# Open research assets and trained adapters

Both branches use the same pinned asset inventory in
[`configs/paper_assets.json`](../configs/paper_assets.json), downloader, split
protocol and training configuration. The engine integrations differ. A Git
checkout contains **code and provenance, not pretrained model weights, raw
datasets, author-trained adapters, or fabricated reproduction results**.

## Initial downloadable inventory

| Asset | Official source | Role and qualification |
| --- | --- | --- |
| Mistral-7B-Instruct-v0.2 | [Mistral AI](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/63a8b081895390a26e140280378bc85ec8bce07a) | Approximately 14.5 GB of safetensors. Full-attention engineering candidate; the paper does not identify an exact Mistral checkpoint. Apache-2.0. |
| HotpotQA distractor | [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab) | Official training and validation data, CC-BY-SA-4.0. Genuine data-derived replay, not unpublished Reflexion trajectories. |
| Multi-Session Chat v0.1 | [ParlAI MSC task](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/msc) | Official SHA256-pinned archive. Download only; no implemented MemGPT replay or quality result yet. Consult the dataset's redistribution terms. |
| SWE-bench | [Princeton NLP](https://huggingface.co/datasets/princeton-nlp/SWE-bench/tree/e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6) | Official task metadata. No automatic execution of repository patches or tests. Source repositories retain their licenses. Not a completed SWE-Agent evaluation. |

The paper also names MPT-30B and Llama-3-70B. They are **not included in this
initial download selection**: MPT needs a separate model adapter, and an exact
Llama checkpoint and any required license/access authorization must be resolved.
Do not silently substitute another model and label the full paper model matrix
complete. B300 results are a hardware adaptation, not measurements on the paper's
A100 setup.

## Download and verify

Run in a dedicated asset directory outside Git. The full initial selection is
about 17.2 GB at its declared upper bound, plus a mandatory 40 GiB free-space
reserve. No remote model code or pickle weights are executed.

```bash
python scripts/download_paper_assets.py \
  --root /workspace/Models/CacheSlide-paper-assets --workers 4

python scripts/download_paper_assets.py \
  --root /workspace/Models/CacheSlide-paper-assets --verify-only
```

The manifest records immutable Hugging Face commits, selected filenames,
official LFS SHA256 or Git-blob checksums, sizes and source URLs. An asset is
published at its final name only after integrity validation. `.part` files are
not models. Re-running the same command resumes interrupted downloads and
validates completed files. The root is bound to its specification/selection;
use a different dedicated root to change `--only` or the specification.

`--verify-only` requires an existing manifest and creates a separate verification
receipt; it neither downloads missing files nor turns partials into completed
assets. No archive is automatically extracted. Training preparation separately
validates the required HotpotQA/tokenizer assets and need not wait for unrelated
MSC or SWE-bench files.

## Publishing a real training run

See [continued pretraining](continued_pretraining.md) for a real next-token
training pipeline. A pilot is deliberately small and must not be called a
quality-qualified model. The release unit is the **combined CacheSlide CoPE and
LoRA adapter**, not the frozen backbone or a generic PEFT LoRA-only file.

For each published adapter, retain:

1. The exact base-model revision, source/license and SHA-bound backbone manifest.
2. `adapter/manifest.json` and `adapter/adapter.safetensors` from the completed
   run, their SHA256 hashes, and the CacheSlide commit for each engine.
3. Training configuration, seed, source/split manifests, token counts, optimizer
   steps, `metrics.jsonl`, held-out NLL/perplexity and the final training report.
4. Untouched held-out generation references, raw outputs and metric definitions;
   separate native-backbone quality from trained-adapter recompute and reuse.
5. Actual GPU/runtime versions and timing methodology. A tiny CUDA training test
   does not certify either native engine, model quality, TTFT or throughput.

Large weights and raw datasets should use a separately authorized model/dataset
host or Git LFS with adequate quota, preserving upstream licenses. GitHub stores
the code, provenance, configurations and verified result summaries. No upload
to an unspecified account or paid storage is performed by these scripts.
An artifact URL should be added only after a real upload and download/hash
verification; this repository deliberately does not list placeholder trained
weights as available.
