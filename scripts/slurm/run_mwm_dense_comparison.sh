#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/slurm/run_mwm_dense_comparison.sh must run inside a Slurm allocation. Submit scripts/slurm/slurm_mwm_dense_comparison.sbatch with sbatch." >&2
  exit 2
fi

cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$PY" verify_mwm_data.py --paper-parity

status=0
run_step() {
  echo "+ $*"
  if ! "$@"; then
    status=1
    echo "WARNING: command failed; continuing so remaining benchmark environments still run: $*" >&2
  fi
}

run_step "$PY" benchmark_mwm.py configs/benchmark/dense_pusht.yaml
run_step "$PY" verify_mwm_benchmark.py configs/benchmark/dense_pusht.yaml
run_step "$PY" benchmark_mwm.py configs/benchmark/dense_reacher.yaml
run_step "$PY" verify_mwm_benchmark.py configs/benchmark/dense_reacher.yaml
run_step "$PY" benchmark_mwm.py configs/benchmark/dense_tworoom.yaml
run_step "$PY" verify_mwm_benchmark.py configs/benchmark/dense_tworoom.yaml

exit "$status"
