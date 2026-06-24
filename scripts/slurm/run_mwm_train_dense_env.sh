#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/slurm/run_mwm_train_dense_env.sh must run inside a Slurm allocation. Submit one of the scripts/slurm/slurm_mwm_train_*_dense.sbatch files with sbatch." >&2
  exit 2
fi

case "${1:-}" in
  pusht)
    CONFIG="configs/train/mwm_dense_pusht.yaml"
    ;;
  reacher)
    CONFIG="configs/train/mwm_dense_reacher.yaml"
    ;;
  ogb_cube|cube)
    CONFIG="configs/train/mwm_dense_ogb_cube.yaml"
    ;;
  tworoom)
    CONFIG="configs/train/mwm_dense_tworoom.yaml"
    ;;
  *)
    echo "Usage: scripts/slurm/run_mwm_train_dense_env.sh {pusht|reacher|ogb_cube|tworoom}" >&2
    exit 2
    ;;
esac

cd "$ROOT"

"$PY" verify_mwm_data.py --paper-parity
"$PY" train_mwm.py "$CONFIG"
