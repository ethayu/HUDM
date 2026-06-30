#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

if [[ ! -d data/upstream/pusht_expert_train.lance ]]; then
  echo "Missing data/upstream/pusht_expert_train.lance. Run python -m mwm.upstream.lewm_data or copy the prepared Lance dataset." >&2
  exit 2
fi
if [[ ! -f checkpoints_mwm/upstream_lewm_pusht/world_metadata.json ]]; then
  echo "Missing checkpoints_mwm/upstream_lewm_pusht. Run python -m mwm.upstream.lewm_checkpoints or copy the prepared checkpoint." >&2
  exit 2
fi

"$PY" -m mwm.benchmark.matrix configs/local/benchmark_pusht_smoke.yaml
"$PY" -m mwm.benchmark.verify configs/local/benchmark_pusht_smoke.yaml
