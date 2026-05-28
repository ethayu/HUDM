#!/usr/bin/env bash
#SBATCH --job-name=mwm_benchmark_gate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/slurm_mwm_benchmark_gate.sbatch must be submitted with sbatch; do not run Slurm batch scripts directly on login nodes." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs rollouts

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

exec scripts/run_mwm_benchmark_gate.sh
