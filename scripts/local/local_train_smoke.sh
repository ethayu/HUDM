#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

if [[ "${RUN_CPU_TRAIN_SMOKE:-0}" != "1" ]]; then
  echo "CPU train smoke can be slow. Re-run with RUN_CPU_TRAIN_SMOKE=1 to opt in." >&2
  exit 2
fi

"$PY" train_mwm.py configs/local/train_pusht_cpu_smoke.yaml
