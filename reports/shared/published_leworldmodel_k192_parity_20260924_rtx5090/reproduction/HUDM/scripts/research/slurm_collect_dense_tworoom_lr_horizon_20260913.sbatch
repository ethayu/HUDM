#!/usr/bin/env bash
#SBATCH --job-name=collect_tr_dense_lrh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5632M
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
REPORT_ROOT="$ROOT/reports/research/dense_tworoom_lr_horizon_epochs_20260913"

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT"

"$PY" scripts/research/collect_dense_tworoom_lr_horizon_training_20260913.py
"$PY" scripts/research/collect_dense_tworoom_lr_horizon_epochs_20260913.py
"$PY" scripts/research/collect_dense_tworoom_lr_horizon_terminal_20260913.py

test -s "$REPORT_ROOT/training_trajectory.json"
test -s "$REPORT_ROOT/epoch_screen_summary.json"
test -s "$REPORT_ROOT/epoch_screen_summary.csv"
test -s "$REPORT_ROOT/epoch_screen_analysis.json"
test -s "$REPORT_ROOT/terminal_n500_summary.json"
test -s "$REPORT_ROOT/terminal_n500_rows.csv"

date --iso-8601=seconds > "$REPORT_ROOT/COLLECTION_COMPLETE"
echo "DENSE_TWOROOM_LR_HORIZON_COLLECTION_COMPLETE"
