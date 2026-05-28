#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/run_mwm_single_level_benchmark.sh must run inside a Slurm allocation. Submit scripts/slurm_mwm_single_level_benchmark.sbatch with sbatch." >&2
  exit 2
fi

cd "$ROOT"

"$PY" verify_mwm_data.py --paper-parity
"$PY" benchmark_mwm.py configs/benchmark_mwm_paper_parity.yaml --roles upstream_lewm_converted retrained_lewm_single
"$PY" verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml \
  --roles upstream_lewm_converted retrained_lewm_single \
  --single-level-only
