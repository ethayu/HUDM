#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

if command -v rg >/dev/null 2>&1; then
  py_files="$(rg --files -g '*.py')"
else
  py_files="$(git ls-files '*.py')"
fi

"$PY" -m py_compile $py_files
"$PY" -m pytest -q
"$PY" verify_mwm_benchmark.py configs/local/benchmark_pusht_smoke.yaml --static-only --no-checkpoints
