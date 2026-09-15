# Reproduction execution status — September 15, 2026

This is an engineering progress record, not a reproduced paper-results table.

| Check | Observed result |
| --- | --- |
| vLLM branch CPU suite, pinned vLLM source enabled | 529 passed, 1 CUDA-only test skipped |
| SGLang branch CPU suite, both pinned sources enabled | 639 passed, 2 explicitly GPU-dependent skips |
| B300 BF16 checkpointed-training mechanics | 1 passed in 41.67 seconds; tiny fixture only |
| Actual HotpotQA preparation | 128 training, 16 NLL-validation, 8 calibration, 16 official-validation evaluation cases |
| Full pretrained Mistral training | Pending complete, verified model weights on B300 |
| Native-engine pretrained-model benchmarks | Not yet run; no accuracy/speedup result claimed |

The real data preparation examined 90,447 official training IDs and 7,405
validation IDs, using seed 9 and the documented answer-independent selection.
All selected prompts fit the explicit 8,192-token capacity without truncation or
short-case replacement. It used tokenizers 0.22.2 and PyArrow 21.0.0. The inputs
are genuine dataset-derived replay, not live Reflexion trajectories.

The B300 test exercised BF16 forward/backward, gradient checkpointing, optimizer
updates and adapter serialization on an actual assigned GPU. It did not use the
pretrained Mistral weights or establish model quality. The task-owned selected
GPU reservation was restored after the test and fresh 100% utilization verified;
a subsequent complete check observed all eight GPUs at 100%, without pause
flags. No other reservation was intentionally released.

The code additionally tests training, saving and loading through each native
adapter's actual bundle/mount path on CPU with a tiny local backbone. Loaded
outputs match the trained reference, and removing the mounted LoRA or CoPE
parameters changes the outputs. This verifies parameter usage, not a native
vLLM/SGLang GPU engine launch.

Official HotpotQA, SWE-bench metadata and the MSC archive were downloaded with
source integrity checks. Mistral model shards were still downloading at this
snapshot. No trained real-model adapter has been published. We do not infer
TTFT, concurrent throughput, SSD write amplification or task accuracy from
training loss, toy fixtures, or partial downloads.
