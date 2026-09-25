#!/usr/bin/env bash
#SBATCH --job-name=mwm_rel_seed_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch; do not run it directly on a login node." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs

export MWM_PYTHON="$PY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

hostname
nvidia-smi || true
"$PY" - <<'PY'
import torch

print("torch", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable inside this GPU Slurm allocation.")
print("cuda", torch.cuda.get_device_name(0))
torch.empty(1, device="cuda")
torch.cuda.synchronize()
PY

exec scripts/research/research_release_identity_seed_sweep.sh
