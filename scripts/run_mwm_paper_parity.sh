#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"

"$PY" prepare_upstream_lewm.py
"$PY" prepare_upstream_lewm_data.py
"$PY" verify_mwm_data.py --paper-parity

"$PY" benchmark_mwm.py configs/benchmark_mwm_paper_parity.yaml --roles upstream_lewm_converted
"$PY" verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml --roles upstream_lewm_converted

"$PY" train_mwm.py configs/train_mwm_lewm_pusht_upstream.yaml
"$PY" train_mwm.py configs/train_mwm_lewm_tworoom_upstream.yaml
"$PY" benchmark_mwm.py configs/benchmark_mwm_paper_parity.yaml
"$PY" verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml
