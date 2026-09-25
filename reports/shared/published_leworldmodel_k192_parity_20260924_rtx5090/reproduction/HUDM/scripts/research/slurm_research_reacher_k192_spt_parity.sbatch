#!/usr/bin/env bash
#SBATCH --job-name=reacher_k192_spt_parity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=03:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
REPORT_ROOT="$ROOT/reports/research/reacher_k192_spt_parity_seed3072_sigreg009_20260804_v2"

cd "$ROOT"
mkdir -p logs reports/research

if [[ -e "$REPORT_ROOT" ]]; then
  echo "ERROR: parity report path already exists: $REPORT_ROOT" >&2
  exit 3
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8

hostname
nvidia-smi
"$PY" - <<'PY'
import importlib.metadata
import torch

version = importlib.metadata.version("stable-pretraining")
print("stable-pretraining", version)
if version != "0.1.6":
    raise SystemExit(f"ERROR: expected stable-pretraining 0.1.6, got {version}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PY

"$PY" scripts/research/research_reacher_k192_spt_parity.py \
  --config configs/train/mwm_lewm_reacher_upstream.yaml \
  --steps 101 \
  --output "$REPORT_ROOT"

test -f "$REPORT_ROOT/PASS"
