#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/slurm/run_mwm_train_identity_env.sh must run inside a Slurm allocation. Submit one of the scripts/slurm/slurm_mwm_train_*_identity.sbatch files with sbatch." >&2
  exit 2
fi

case "${1:-}" in
  pusht)
    CONFIG="configs/train/mwm_lewm_pusht_upstream.yaml"
    ;;
  reacher)
    CONFIG="configs/train/mwm_lewm_reacher_upstream.yaml"
    ;;
  ogb_cube|cube)
    CONFIG="configs/train/mwm_lewm_ogb_cube_upstream.yaml"
    ;;
  tworoom)
    CONFIG="configs/train/mwm_lewm_tworoom_upstream.yaml"
    ;;
  *)
    echo "Usage: scripts/slurm/run_mwm_train_identity_env.sh {pusht|reacher|ogb_cube|tworoom}" >&2
    exit 2
    ;;
esac

cd "$ROOT"

"$PY" -m mwm.data.verify --paper-parity
"$PY" -m mwm.training.stable_wm "$CONFIG"
