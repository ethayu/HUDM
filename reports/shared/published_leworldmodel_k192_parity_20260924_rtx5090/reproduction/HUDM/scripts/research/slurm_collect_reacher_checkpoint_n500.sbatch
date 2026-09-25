#!/usr/bin/env bash
#SBATCH --job-name=collect_reacher_ckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5632M
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi
for name in MWM_CANDIDATE_ROLE MWM_REPORT_TAG; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be set." >&2
    exit 2
  fi
done

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" scripts/research/collect_reacher_checkpoint_n500.py \
  --report-tag "$MWM_REPORT_TAG" \
  --candidate-role "$MWM_CANDIDATE_ROLE"
test -s "reports/research/$MWM_REPORT_TAG/n500_summary.json"
