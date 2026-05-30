#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/run_mwm_identity_parity.sh must run inside a Slurm allocation. Submit scripts/slurm_mwm_identity_parity.sbatch with sbatch." >&2
  exit 2
fi

cd "$ROOT"

"$PY" verify_mwm_data.py --paper-parity
"$PY" benchmark_mwm.py configs/benchmark/paper_parity_pusht.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" verify_mwm_benchmark.py configs/benchmark/paper_parity_pusht.yaml \
  --roles upstream_lewm_converted retrained_lewm_identity
"$PY" benchmark_mwm.py configs/benchmark/paper_parity_tworoom.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" verify_mwm_benchmark.py configs/benchmark/paper_parity_tworoom.yaml \
  --roles upstream_lewm_converted retrained_lewm_identity
