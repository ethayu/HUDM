#!/usr/bin/env bash
#SBATCH --job-name=reacher_release20260728_finish_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch from a GPU Slurm allocation; do not run it on a login node." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
# Set MWM_REACHER_CONFIG to the goal25 or goal50 release20260728 all-fidelity
# config before submitting; --resume skips any cell that already has a
# complete sidecar with an identical resolved config, so this only fills in
# the missing cells (the new single-K 27-31 baselines plus any partial
# 24-26 coverage) instead of recomputing the whole matrix.
CONFIG="${MWM_REACHER_CONFIG:?set MWM_REACHER_CONFIG to the target release20260728 reacher config path}"

cd "$ROOT"
mkdir -p logs

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
"$PY" -m mwm.benchmark.matrix "$CONFIG" --resume
"$PY" -m mwm.benchmark.verify "$CONFIG"
