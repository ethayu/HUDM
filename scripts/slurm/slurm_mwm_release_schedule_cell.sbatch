#!/usr/bin/env bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script as a Slurm array." >&2
  exit 2
fi
if [[ -z "${MWM_BENCHMARK_CONFIG:-}" ]]; then
  echo "ERROR: MWM_BENCHMARK_CONFIG must name a release schedule benchmark config." >&2
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

echo "config=${MWM_BENCHMARK_CONFIG} shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -m mwm.benchmark.matrix "$MWM_BENCHMARK_CONFIG" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --resume
