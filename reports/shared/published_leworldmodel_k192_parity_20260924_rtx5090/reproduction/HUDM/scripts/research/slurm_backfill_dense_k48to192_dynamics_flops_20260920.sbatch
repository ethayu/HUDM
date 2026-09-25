#!/usr/bin/env bash
#SBATCH --job-name=audit_dense_flops
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
SCRIPT="$ROOT/scripts/research/backfill_quick_benchmark_dynamics_flops.py"

cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

for env in tworoom ogb_cube reacher pusht; do
  benchmark_root="$ROOT/reports/research/dense_${env}_k48_72_96_120_144_168_192_quick_20260917/final_n50_seed42"
  output_dir="$ROOT/reports/research/dense_${env}_k48_72_96_120_144_168_192_quick_20260917/dynamics_flops_backfill_20260920"
  test -s "$benchmark_root/summary.json"
  "$PY" "$SCRIPT" "$benchmark_root" --output-dir "$output_dir" --device cuda
  test -s "$output_dir/dynamics_flops_backfill.json"
  test -s "$output_dir/dynamics_flops_backfill.csv"
  test -s "$output_dir/SHA256SUMS"
done

echo "DENSE_K48TO192_DYNAMICS_FLOPS_BACKFILL_COMPLETE"
