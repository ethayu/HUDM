#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/run_mwm_benchmark_gate.sh must run inside a Slurm allocation. Submit scripts/slurm_mwm_benchmark_gate.sbatch with sbatch." >&2
  exit 2
fi

cd "$ROOT"

"$PY" benchmark_mwm.py configs/benchmark_mwm.yaml
"$PY" verify_mwm_benchmark.py configs/benchmark_mwm.yaml
