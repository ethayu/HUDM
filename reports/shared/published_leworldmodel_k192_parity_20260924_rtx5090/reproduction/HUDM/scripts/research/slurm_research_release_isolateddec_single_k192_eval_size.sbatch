#!/usr/bin/env bash
#SBATCH --job-name=mwm_isodec_k192_n500
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig90
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%16
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as a Slurm array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
ENVS=(pusht reacher ogb_cube tworoom)
SEEDS=(0 1 2 42 100)
env_index=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))

cd "$ROOT"
mkdir -p logs

export MWM_PYTHON="$PY"
export MWM_SWEEP_ENV="${ENVS[$env_index]}"
export MWM_SWEEP_SEED="${SEEDS[$seed_index]}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "array_task=${SLURM_ARRAY_TASK_ID} env=${MWM_SWEEP_ENV} seed=${MWM_SWEEP_SEED}"
hostname
nvidia-smi || true
"$PY" - <<'PY'
import torch

print("torch", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable inside this GPU Slurm allocation.")
print("cuda", torch.cuda.get_device_name(0))
torch.empty(1, device="cuda")
torch.cuda.synchronize()
PY

exec scripts/research/research_release_isolateddec_single_k192_eval_size.sh
