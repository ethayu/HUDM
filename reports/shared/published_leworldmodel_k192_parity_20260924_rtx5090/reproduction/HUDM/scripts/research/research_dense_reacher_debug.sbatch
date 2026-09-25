#!/usr/bin/env bash
#SBATCH --job-name=research_dense_reacher_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch from a GPU Slurm allocation; do not run it on a login node." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="${MWM_REACHER_CONFIG:-configs/research/dense_reacher_planner_ablation.yaml}"

cd "$ROOT"
mkdir -p logs reports/research/dense_reacher_debug

export MWM_PYTHON="$PY"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

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

"$PY" -m mwm.benchmark.verify "$CONFIG" --static-only
"$PY" -m mwm.benchmark.matrix "$CONFIG"
"$PY" -m mwm.benchmark.verify "$CONFIG"
