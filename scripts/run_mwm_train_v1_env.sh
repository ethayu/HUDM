#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/run_mwm_train_v1_env.sh must run inside a Slurm allocation. Submit one of the scripts/slurm_mwm_train_*_v1*.sbatch files with sbatch." >&2
  exit 2
fi

case "${1:-}" in
  pusht-single)
    CONFIG="configs/train_mwm_lewm_pusht.yaml"
    ;;
  tworoom-single)
    CONFIG="configs/train_mwm_lewm_tworoom.yaml"
    ;;
  pusht-scheduled)
    CONFIG="configs/train_mwm_scheduled_pusht.yaml"
    ;;
  tworoom-scheduled)
    CONFIG="configs/train_mwm_scheduled_tworoom.yaml"
    ;;
  *)
    echo "Usage: scripts/run_mwm_train_v1_env.sh {pusht-single|tworoom-single|pusht-scheduled|tworoom-scheduled}" >&2
    exit 2
    ;;
esac

cd "$ROOT"

"$PY" verify_mwm_data.py
"$PY" train_mwm.py "$CONFIG"
