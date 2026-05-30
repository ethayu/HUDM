#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

if [[ ! -d data/upstream/pusht_expert_train.lance ]]; then
  echo "Missing data/upstream/pusht_expert_train.lance. Run prepare_upstream_lewm_data.py or copy the prepared Lance dataset." >&2
  exit 2
fi
if [[ ! -f checkpoints_mwm/upstream_lewm_pusht/world_metadata.json ]]; then
  echo "Missing checkpoints_mwm/upstream_lewm_pusht. Run prepare_upstream_lewm.py or copy the prepared checkpoint." >&2
  exit 2
fi

"$PY" benchmark_mwm.py configs/local/benchmark_pusht_smoke.yaml
"$PY" verify_mwm_benchmark.py configs/local/benchmark_pusht_smoke.yaml
