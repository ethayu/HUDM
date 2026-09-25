#!/usr/bin/env bash
#SBATCH --job-name=audit_tr_dense_lrh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5632M
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/research/audit_dense_tworoom_lr_horizon_20260913.py
test "$(jq -r .status reports/research/dense_tworoom_lr_horizon_epochs_20260913/completion_audit.json)" = "pass"
echo "DENSE_TWOROOM_LR_HORIZON_AUDIT_COMPLETE"
