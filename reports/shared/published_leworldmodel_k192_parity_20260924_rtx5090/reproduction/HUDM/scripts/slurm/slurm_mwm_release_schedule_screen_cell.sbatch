#!/usr/bin/env bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with sbatch as a sparse Slurm array." >&2
  exit 2
fi
if [[ -z "${MWM_SCREEN_CONFIG:-}" ]]; then
  echo "ERROR: MWM_SCREEN_CONFIG must name a generated 50-episode screen config." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
NUM_SHARDS="${MWM_NUM_SHARDS:-3120}"

cd "$ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "screen_config=${MWM_SCREEN_CONFIG} matrix_index=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -m mwm.benchmark.adaptive_workflow run-screen-cell \
  --screen-config "$MWM_SCREEN_CONFIG" \
  --num-shards "$NUM_SHARDS" \
  --matrix-index "$SLURM_ARRAY_TASK_ID"
