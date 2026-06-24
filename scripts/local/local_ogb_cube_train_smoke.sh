#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

cd "$ROOT"

if [[ "${RUN_CPU_TRAIN_SMOKE:-0}" != "1" ]]; then
  echo "CPU train smoke can be slow. Re-run with RUN_CPU_TRAIN_SMOKE=1 to opt in." >&2
  exit 2
fi

DATASET="data/ogb_cube_smoke.lance"
if [[ ! -e "$DATASET" ]]; then
  "$PY" collect_mwm_data.py configs/local/collect_ogb_cube_smoke.yaml
else
  echo "Using existing OGBench Cube smoke dataset: $DATASET"
fi

"$PY" train_mwm.py configs/local/train_ogb_cube_cpu_smoke.yaml

CHECKPOINT="checkpoints_mwm/local_ogb_cube_cpu_smoke"
for name in config.json weights.pt world_metadata.json; do
  if [[ ! -s "$CHECKPOINT/$name" ]]; then
    echo "Missing canonical checkpoint file: $CHECKPOINT/$name" >&2
    exit 1
  fi
done

echo "OGBench Cube smoke checkpoint ready: $CHECKPOINT"
