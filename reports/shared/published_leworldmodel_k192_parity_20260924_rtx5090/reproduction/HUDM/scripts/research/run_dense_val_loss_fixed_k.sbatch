#!/usr/bin/env bash
#SBATCH --job-name=mwm_loss_fixed_k
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig90
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
  exit 2
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
OUTPUT_ROOT="${MWM_ANALYSIS_ROOT:-reports/research/dense_val_loss_fixed_k_20260721}"
ENVIRONMENTS=(pusht reacher ogb_cube tworoom)
ENV_SLUG="${ENVIRONMENTS[$SLURM_ARRAY_TASK_ID]}"
LR_ARGS=()
CONFIG_SUFFIX="$ENV_SLUG"
if [[ -n "${MWM_LEARNING_RATE_TOKEN:-}" ]]; then
  LR_ARGS=(--learning-rate-token "$MWM_LEARNING_RATE_TOKEN")
  CONFIG_SUFFIX="${ENV_SLUG}_lr${MWM_LEARNING_RATE_TOKEN}"
fi
CONFIG_PATH="$OUTPUT_ROOT/configs/fixed_k_$CONFIG_SUFFIX.yaml"

cd "$REPO_ROOT"
mkdir -p logs "$OUTPUT_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

hostname
nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY

"$PYTHON_BIN" scripts/research/run_dense_val_loss_fixed_k.py \
  --env "$ENV_SLUG" \
  --output-root "$OUTPUT_ROOT" \
  "${LR_ARGS[@]}" \
  --write-config-only
"$PYTHON_BIN" -m mwm.benchmark.verify "$CONFIG_PATH" --static-only
"$PYTHON_BIN" scripts/research/run_dense_val_loss_fixed_k.py \
  --env "$ENV_SLUG" \
  --output-root "$OUTPUT_ROOT" \
  "${LR_ARGS[@]}"
"$PYTHON_BIN" -m mwm.benchmark.verify "$CONFIG_PATH"
