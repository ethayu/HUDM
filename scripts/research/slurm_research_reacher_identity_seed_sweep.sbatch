#!/usr/bin/env bash
#SBATCH --job-name=mwm_reacher_id_delta
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch must be submitted with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs reports/research/reacher_identity_delta/seed_sweep

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

print("python ok")
cuda_available = torch.cuda.is_available()
print("torch", torch.__version__, "cuda", cuda_available, torch.cuda.get_device_name(0) if cuda_available else None)
if not cuda_available:
    raise SystemExit("ERROR: CUDA is unavailable inside this GPU Slurm allocation.")
probe = torch.empty(1, device="cuda")
torch.cuda.synchronize()
print("cuda allocation probe ok", probe.device)
PY

exec scripts/research/research_reacher_identity_seed_sweep.sh
