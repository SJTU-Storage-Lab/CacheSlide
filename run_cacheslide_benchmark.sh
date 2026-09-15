#!/usr/bin/env bash
# Keep caller-relative input/output paths; do not allocate GPUs on help/plan.
set -euo pipefail
workflow_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CACHESLIDE_PYTHON:-}" ]]; then
    workflow_python="$CACHESLIDE_PYTHON"
elif [[ -x "$workflow_root/.sglang-venv/bin/python" ]]; then
    workflow_python="$workflow_root/.sglang-venv/bin/python"
else
    workflow_python=python3
fi
export PYTHONPATH="$workflow_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$workflow_python" -P -m cacheslide_sglang.workflow "$@"
