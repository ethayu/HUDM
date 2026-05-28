#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/run_mwm_train_single_level_env.sh must run inside a Slurm allocation. Submit one of the scripts/slurm_mwm_train_*_single.sbatch files with sbatch." >&2
  exit 2
fi

case "${1:-}" in
  pusht)
    CONFIG="configs/train_mwm_lewm_pusht_upstream.yaml"
    ;;
  tworoom)
    CONFIG="configs/train_mwm_lewm_tworoom_upstream.yaml"
    ;;
  *)
    echo "Usage: scripts/run_mwm_train_single_level_env.sh {pusht|tworoom}" >&2
    exit 2
    ;;
esac

cd "$ROOT"

"$PY" verify_mwm_data.py --paper-parity
"$PY" train_mwm.py "$CONFIG"
