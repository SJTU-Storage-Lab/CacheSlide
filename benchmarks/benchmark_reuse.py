"""Run the explicit native benchmark; omit --run to review configuration only.

Example (after training/calibration and installing the pinned engine extra):
  python benchmarks/benchmark_reuse.py --model /model --adapter /adapter \
    --profiles /profiles --input cases.jsonl --cache-root /cache \
    --output /new-result-directory --run

Reported latency is validated offline generation (four output tokens by default),
not streaming TTFT. No engine, CUDA context, or GPU work is created without --run.
"""

from cacheslide_sglang.cli import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["bench", *sys.argv[1:]]))
