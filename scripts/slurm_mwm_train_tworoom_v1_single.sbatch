#!/usr/bin/env bash
#SBATCH --job-name=mwm_train_tworoom_v1_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs rollouts checkpoints_mwm

export MWM_PYTHON="$PY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

hostname
nvidia-smi || true
"$PY" - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

exec scripts/run_mwm_train_v1_env.sh tworoom-single
