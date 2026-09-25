#!/usr/bin/env bash
#SBATCH --job-name=eval_tr_h100_e20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:20:00
#SBATCH --array=0-9%4
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
EPOCH=$((10 + SLURM_ARRAY_TASK_ID))
RUN_NAME="mwm_dense_tworoom_h100_continue20_20260914"
CHECKPOINT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports/epoch_$(printf '%03d' "$EPOCH")"
REPORT_ROOT="$ROOT/reports/research/dense_tworoom_h100_continue20_20260914"
OUTPUT_DIR="$REPORT_ROOT/screen/epoch_$(printf '%03d' "$EPOCH")"
CONFIG_PATH="$REPORT_ROOT/screen/epoch_$(printf '%03d' "$EPOCH")_config.yaml"
EVAL_CONFIG="$ROOT/configs/eval/research_tworoom_exact_seed42_n50_20260913.yaml"
MANIFEST="$ROOT/rollouts/manifests/tworoom_paper_seed42.json"

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT/screen"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite existing evaluation output: $OUTPUT_DIR" >&2
  exit 3
fi
for name in config.json weights.pt world_metadata.json; do
  test -s "$CHECKPOINT/$name"
done
test -s "$EVAL_CONFIG"
test -s "$MANIFEST"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"$PY" scripts/research/validate_tworoom_manifest_contract_20260913.py \
  --manifest "$MANIFEST" \
  --episodes 50 \
  --seed 42

echo "array_task=$SLURM_ARRAY_TASK_ID epoch=$EPOCH"
hostname
nvidia-smi
"$PY" scripts/research/eval_dense_tworoom_lr_horizon_epoch.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --eval-config "$EVAL_CONFIG" \
  --write-config-only
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH" --static-only
"$PY" scripts/research/eval_dense_tworoom_lr_horizon_epoch.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --eval-config "$EVAL_CONFIG"
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH"
echo "DENSE_TWOROOM_H100_CONTINUE20_EPOCH_EVAL_COMPLETE epoch=$EPOCH"
