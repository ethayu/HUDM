#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"

"$PY" benchmark_mwm.py configs/benchmark_mwm.yaml
"$PY" verify_mwm_benchmark.py configs/benchmark_mwm.yaml
