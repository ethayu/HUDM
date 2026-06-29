#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
ENV_NAME="${1:-tworoom}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_train_dense_highk_converge.sh must run inside a Slurm allocation." >&2
  exit 2
fi

mkdir -p logs reports/research/dense_debug/checkpoints_mwm
[[ -e data ]] || ln -s "${ARTIFACT_ROOT}/data" data
[[ -e checkpoints_mwm ]] || ln -s "${ARTIFACT_ROOT}/checkpoints_mwm" checkpoints_mwm

run_one() {
  local env_name="$1"
  local config="configs/research/train_mwm_lewm_dense_${env_name}_highk_weighted_converge.yaml"
  echo "[dense-highk-converge] env=${env_name}"
  echo "[dense-highk-converge] config=${config}"
  "${PYTHON_BIN}" -m mwm.data.verify --paper-parity
  "${PYTHON_BIN}" -m mwm.training.stable_wm "${config}"
}

case "${ENV_NAME}" in
  pusht|tworoom)
    run_one "${ENV_NAME}"
    ;;
  all)
    run_one pusht
    run_one tworoom
    ;;
  *)
    echo "Usage: scripts/research/research_train_dense_highk_converge.sh {pusht|tworoom|all}" >&2
    exit 2
    ;;
esac
