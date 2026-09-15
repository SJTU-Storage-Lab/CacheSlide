#!/usr/bin/env bash
# Explicit vLLM workflow retained for cross-backend validation.
set -euo pipefail
workflow_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CACHESLIDE_PYTHON:-}" ]]; then
    workflow_python="$CACHESLIDE_PYTHON"
elif [[ -x "$workflow_root/.venv/bin/python" ]]; then
    workflow_python="$workflow_root/.venv/bin/python"
else
    workflow_python=python3
fi
export PYTHONPATH="$workflow_root/src${PYTHONPATH:+:$PYTHONPATH}"
# Do not let an unrelated cacheslide_vllm or vllm directory in cwd shadow imports.
exec "$workflow_python" -P -m cacheslide_vllm.workflow "$@"
