#!/usr/bin/env bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with sbatch as a Slurm array." >&2
  exit 2
fi
if [[ -z "${MWM_DIAGNOSTIC_MANIFEST:-}" ]]; then
  echo "ERROR: MWM_DIAGNOSTIC_MANIFEST must name an immutable launch manifest." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "launch_manifest=${MWM_DIAGNOSTIC_MANIFEST} task_index=${SLURM_ARRAY_TASK_ID}"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -m mwm.benchmark.anchor_diagnostic run-task \
  --launch-manifest "$MWM_DIAGNOSTIC_MANIFEST" \
  --task-index "$SLURM_ARRAY_TASK_ID"
