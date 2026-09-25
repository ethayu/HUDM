#!/usr/bin/env bash
#SBATCH --job-name=eval_dense_reacher_sched
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%5
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
SEEDS=(0 1 2 42 100)

cd "$ROOT"
mkdir -p logs
export MWM_PYTHON="$PY"
export MWM_SWEEP_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "array_task=${SLURM_ARRAY_TASK_ID} env=reacher seed=${MWM_SWEEP_SEED}"
hostname
nvidia-smi
bash scripts/research/research_dense_reacher_scheduler_ablation_n500.sh
