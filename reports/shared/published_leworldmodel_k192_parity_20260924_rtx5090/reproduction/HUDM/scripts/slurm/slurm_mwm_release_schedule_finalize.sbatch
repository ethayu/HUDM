#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=genoa-std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=132G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi
if [[ -z "${MWM_BENCHMARK_CONFIG:-}" ]]; then
  echo "ERROR: MWM_BENCHMARK_CONFIG must name a release schedule benchmark config." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

"$PY" -m mwm.benchmark.matrix "$MWM_BENCHMARK_CONFIG" --finalize-only
"$PY" -m mwm.benchmark.verify "$MWM_BENCHMARK_CONFIG"
