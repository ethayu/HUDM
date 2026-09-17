#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with sbatch; do not run it on a login node." >&2
  exit 2
fi

ENV_SLUG="${1:-}"
LEVEL_CSV="${2:-}"
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
    echo "ERROR: unsupported environment '$ENV_SLUG'." >&2
    exit 2
    ;;
esac

case "$LEVEL_CSV" in
  96|120|144|168|192|96,120,144,168,192) ;;
  *)
    echo "ERROR: unsupported K specification '$LEVEL_CSV'." >&2
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

if [[ -e "checkpoints_mwm/$RUN_NAME" || -e "logs/mwm_training/$RUN_NAME" ]]; then
  echo "ERROR: refusing to reuse existing run path for $RUN_NAME." >&2
  exit 2
fi

export MWM_PYTHON="$PY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

hostname
nvidia-smi
"$PY" - <<'PY'
import torch

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
  --set "model.K=[$LEVEL_CSV]" \
  --set "train.run_name=$RUN_NAME"
