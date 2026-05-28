#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"

"$PY" verify_mwm_data.py
"$PY" prepare_upstream_lewm.py

"$PY" train_mwm.py configs/train_mwm_lewm_pusht.yaml
"$PY" train_mwm.py configs/train_mwm_lewm_tworoom.yaml
"$PY" train_mwm.py configs/train_mwm_scheduled_pusht.yaml
"$PY" train_mwm.py configs/train_mwm_scheduled_tworoom.yaml

"$PY" benchmark_mwm.py configs/benchmark_mwm.yaml
"$PY" verify_mwm_benchmark.py configs/benchmark_mwm.yaml
