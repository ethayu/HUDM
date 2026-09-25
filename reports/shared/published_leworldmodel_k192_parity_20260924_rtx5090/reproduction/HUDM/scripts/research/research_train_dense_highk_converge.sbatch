#!/bin/bash
#SBATCH --job-name=mwm_dense_highk_conv
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_train_dense_highk_converge.sbatch must be submitted with sbatch; do not run Slurm batch scripts directly on login nodes." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs reports/research/dense_debug/checkpoints_mwm

export MWM_PYTHON="$PY"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "[dense-highk-converge] host=$(hostname)"
echo "[dense-highk-converge] job=${SLURM_JOB_ID:-unknown}"
echo "[dense-highk-converge] env=${DENSE_HIGHK_ENV:-tworoom}"
nvidia-smi || true

"$PY" - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

exec bash scripts/research/research_train_dense_highk_converge.sh "${DENSE_HIGHK_ENV:-tworoom}"
