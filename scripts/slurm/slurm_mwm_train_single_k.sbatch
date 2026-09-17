#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
  exit 2
fi

ENV_SLUG="${1:-}"
LEVEL="${2:-}"
RUN_NAME="${3:-}"

case "$ENV_SLUG" in
  pusht)
    CONFIG="configs/train/mwm_lewm_dense_pusht.yaml"
    ;;
  reacher)
    CONFIG="configs/train/mwm_lewm_dense_reacher.yaml"
    ;;
  ogb_cube)
    CONFIG="configs/train/mwm_lewm_dense_ogb_cube.yaml"
    ;;
  tworoom)
    CONFIG="configs/train/mwm_lewm_dense_tworoom.yaml"
    ;;
  *)
    echo "Usage: sbatch $0 {pusht|reacher|ogb_cube|tworoom} {96|144} RUN_NAME" >&2
    exit 2
    ;;
esac

case "$LEVEL" in
  96|144) ;;
  *)
    echo "ERROR: level must be 96 or 144, got '$LEVEL'." >&2
    exit 2
    ;;
esac

if [[ -z "$RUN_NAME" ]]; then
  echo "ERROR: RUN_NAME is required." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs rollouts

export MWM_PYTHON="$PY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

hostname
nvidia-smi
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

"$PY" -m mwm.data.verify --paper-parity
"$PY" -m mwm.training.stable_wm "$CONFIG" \
  --set "model.K=[$LEVEL]" \
  --set "train.run_name=$RUN_NAME"
