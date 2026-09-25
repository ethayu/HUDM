#!/usr/bin/env bash
#SBATCH --job-name=eval_reacher_init_n500
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-24%5
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi
for name in MWM_INIT_SWEEP_TAG MWM_INIT_SWEEP_REPORT_PREFIX MWM_EVAL_CONFIG; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be exported at submission." >&2
    exit 2
  fi
done

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
INIT_SEEDS=(0 1 2 42 100)
EVAL_SEEDS=(0 1 2 42 100)
INIT_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
EVAL_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
MODEL_INIT_SEED="${INIT_SEEDS[$INIT_INDEX]}"
EVAL_SEED="${EVAL_SEEDS[$EVAL_INDEX]}"
RUN_NAME="mwm_reacher_k192_${MWM_INIT_SWEEP_TAG}_nativeh5_nodecoder_init${MODEL_INIT_SEED}_fit0_20260806"

cd "$ROOT"
mkdir -p logs

export MWM_PYTHON="$PY"
export MWM_SWEEP_SEED="$EVAL_SEED"
export MWM_CANDIDATE_CHECKPOINT="checkpoints_mwm/$RUN_NAME"
export MWM_CANDIDATE_ROLE="mwm_model_init_${MODEL_INIT_SEED}"
export MWM_REPORT_TAG="${MWM_INIT_SWEEP_REPORT_PREFIX}_init${MODEL_INIT_SEED}"
export MWM_INCLUDE_UPSTREAM=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

hostname
nvidia-smi
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

exec bash scripts/research/research_reacher_checkpoint_n500.sh
