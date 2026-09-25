#!/usr/bin/env bash
#SBATCH --job-name=k192_parity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs
export MWM_PYTHON="$PY"

hostname
nvidia-smi || true
"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable inside this allocation")
print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0))
torch.empty(1, device="cuda")
torch.cuda.synchronize()
PY

exec scripts/research/run_k192_sepopt_published_parity_20260924.sh
