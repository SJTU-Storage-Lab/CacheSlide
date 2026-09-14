
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./CacheSlide.png">
    <img
      alt="CacheSlide"
      src="./CacheSlide.png"
      style="width: 100%; max-width: 1100px; height: auto;"
    >
  </picture>
</p>

<h3 align="center">
CacheSlide v1: Cross Position-Aware KV Cache Reuse for Faster Serving
</h3>
<p align="center">
  📄 <b>CacheSlide  Paper link is</b>: <a href="docs/source/CacheSlide.pdf">PDF</a>
</p>


---

## About

### Modern vLLM implementation

The new [standalone implementation in `modern/`](modern/README.md) targets
**vLLM 0.29.0** and provides trained CoPE/LoRA artifacts, bounded CCPE calibration,
request-local WCA selective execution, and SLIDE page indirection on native V1
KV allocations. It is an experimental, separately installed Llama/full-attention
Mistral path; see its [design and support boundaries](modern/docs/design.md).
It does not replace the legacy tree below or claim reproduction of the paper's
large-model performance results. Install and launch it outside this repository's
legacy `vllm/` import path.

### Legacy vLLM implementation

This repository implements **CacheSlide** on top of **vLLM 0.8.5** (pinned) ([PyPI][1]), adding:

* **Chunked (document-level) KV cache construction** and **cross-chunk reuse**
* **Cross-position-aware matching / mapping** between “recompute boundary” and “reuse boundary”
* **WCA (Weighted Cache Adaptation)** logic integrated into the attention path (ported from the LLaMA-style implementation you shared)
* **SLIDE component (chunk-level retrieval)**, decoupled inter-layer pipeline (compute/IO overlap), and SSD-spill-aware KV cache management (tiered GPU/CPU/NVMe residency)
* Optional hooks for:

  * **LoRA-based “reuse-part encoding” pretraining / pre-encoding**
  * **Cross-attention KV pre-generation** (for agent-like pipelines)

---

## Reproducibility Environment

We match the hardware/software profile described in the paper:

### Hardware

* **GPU**: 1 × NVIDIA **A100 80GB HBM** (70B models: **2 × A100**)
* **Host DRAM**: >=500 GB
* **Storage**: >=2 TB NVMe SSD
* **Interconnect**: PCIe Gen 4 (GPU interconnect)

### Software

* **OS**: Ubuntu 20.04
* **Linux kernel**: 5.16.7
* **CUDA**: 12.6

---

## Versions (Pinned)

* **vLLM**: `0.8.5` (released Apr 28, 2025) ([PyPI][1])
* **Python**: `>=3.9, <3.13` ([PyPI][1])

> Note: vLLM’s prebuilt wheels are compiled for a specific CUDA version (see vLLM’s GPU install docs). ([vLLM][2])
> If you want to *exactly* match CUDA **12.6**, build from source in your CUDA 12.6 environment.

---

## Installation

### Option A: Install pinned CacheSlide (fastest)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install  CacheSlide v1
```

(Install command and version pin are shown on PyPI.) ([PyPI][1])

## Runtime Environment Parameters (vLLM)

CacheSlide exposes runtime env vars for reproducibility and backend selection. Common ones you may want to set:

### Attention backend

```bash
# Example: force a specific backend (choose what your system supports)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```
(`VLLM_ATTENTION_BACKEND` is documented by vLLM.) 

### Cache / compilation

```bash
# Where vLLM stores artifacts/cache (optional)
export VLLM_CACHE_ROOT=/path/to/vllm_cache
```

(`VLLM_CACHE_ROOT` and other cache-related env vars are documented.) 

---

## Datasets (HuggingFace Download)

This repo uses HuggingFace Datasets (no local JSON loading by default).

### HotPotQA

Loaded from the HF dataset hub (HotPotQA).

### SWE-Agent-style evaluation

If you are using SWE-bench / SWE-agent pipelines, SWE-bench is available on HF (dataset card).

> If your exact “SWE-Agent-Bench” split is a custom 2-task/12-repo subset, you can still fetch the base dataset from HF and filter down in code.

---

## How to Run

Below is the **recommended experiment flow** that matches your “template” (build chunk KV → set reuse/recompute boundary → optionally pre-encode reuse part → pre-generate cross-attention KV → run decode).

### 1) Run a single-GPU experiment (Mistral 7B example)

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/path/to/hf_cache   # optional but recommended

python examples/Cacheslide.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset hotpot_qa \
  --split validation \
  --max_examples 200 \
  --gpu_memory_utilization 0.50 \
  --max_new_tokens 64 \
  --enable_cacheslide \
  --chunk_size 1024 \
  --suffix_len 256
```

### 2) Two-GPU run (70B-class models)

```bash
export CUDA_VISIBLE_DEVICES=0,1

python examples/Cacheslide.py  \
  --model <your-70b-model> \
  --tensor_parallel_size 2 \
  --dataset hotpot_qa \
  --split validation \
  --gpu_memory_utilization 0.90 \
  --enable_cacheslide \
  --chunk_size 1024 \
  --suffix_len 256
```

### 3) Agent-style pipeline (optional)

If you want the “agent” loop (retrieve → build chunk KV → tool/exec → reuse KV), run:

```bash
python scripts/run_agent.py \
  --agent swe_style \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset swe_bench \
  --enable_cacheslide
---

## Citation

If you use CacheSlide for your research, please cite our paper:

```bibtex
@inproceedings{liu2026CacheSlide,
  title={CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving},
  author={Yang Liu and Yunfei Gu and Liqiang Zhang and Chentao Wu and Guangtao Xue and Jie Li and Minyi Guo and Junhao Hu and Jie Meng},
  year={2026}
}
```

---

## Notes

* **CUDA 12.6 parity**: if you care about strict reproducibility vs the paper’s CUDA version, use the **build-from-source** path and ensure your driver/toolkit match.
* **Backend determinism**: pin `VLLM_ATTENTION_BACKEND` for consistent kernels across machines. 
* **Ongoing updates:** We will continue to update the codebase; as a result, some intermediate versions may be temporarily non-functional. For inquiries, please contact [liuyang370@sjtu.edu.cn](mailto:liuyang370@sjtu.edu.cn).
---
