#!/usr/bin/env bash
#SBATCH --job-name=mwm_identity_seed_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/slurm_research_identity_seed_sweep.sbatch must be submitted with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
DEFAULT_ARTIFACT_ROOT="/vast/projects/dineshj/lab/ethanyu/code/H""UDM"

cd "$ROOT"
mkdir -p logs reports/research/identity_delta/seed_sweep

export MWM_PYTHON="$PY"
export MWM_ARTIFACT_ROOT="${MWM_ARTIFACT_ROOT:-$DEFAULT_ARTIFACT_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

hostname
nvidia-smi || true
"$PY" - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

exec scripts/research/research_identity_seed_sweep.sh
