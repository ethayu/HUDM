#!/usr/bin/env bash
#SBATCH --job-name=confirm_tr_dense_lrh
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%10
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
HORIZONS=(10 100)
SEEDS=(0 1 2 42 100)
HORIZON_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
LR_HORIZON="${HORIZONS[$HORIZON_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN_NAME="mwm_dense_tworoom_lr_horizon_h${LR_HORIZON}_epochs_20260913"
CHECKPOINT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports/epoch_009"
REPORT_ROOT="$ROOT/reports/research/dense_tworoom_lr_horizon_epochs_20260913"
OUTPUT_DIR="$REPORT_ROOT/terminal_n500/h${LR_HORIZON}/seed${SEED}"
CONFIG_PATH="$REPORT_ROOT/terminal_n500/h${LR_HORIZON}/seed${SEED}_config.yaml"
SOURCE_ROOT="$ROOT/reports/research/crossenv_scheduler_ablation_20260807/generated_configs"
EVAL_CONFIG="$SOURCE_ROOT/eval_snapshot_tworoom_seed${SEED}_n500.yaml"
MANIFEST="$ROOT/rollouts/manifests/release20260728_dense_k192_eval_size/tworoom_seed${SEED}_n500.json"

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT/terminal_n500/h${LR_HORIZON}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite existing confirmation output: $OUTPUT_DIR" >&2
  exit 3
fi
for name in config.json weights.pt world_metadata.json; do
  test -s "$CHECKPOINT/$name"
done
test -s "$EVAL_CONFIG"
test -s "$MANIFEST"
"$PY" scripts/research/validate_tworoom_manifest_contract_20260913.py \
  --manifest "$MANIFEST" \
  --episodes 500 \
  --seed "$SEED"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "array_task=$SLURM_ARRAY_TASK_ID lr_horizon=$LR_HORIZON seed=$SEED"
hostname
nvidia-smi
"$PY" scripts/research/eval_dense_tworoom_lr_horizon_epoch.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --episodes 500 \
  --num-envs 50 \
  --eval-config "$EVAL_CONFIG" \
  --manifest "$MANIFEST" \
  --write-config-only
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH" --static-only
"$PY" scripts/research/eval_dense_tworoom_lr_horizon_epoch.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --episodes 500 \
  --num-envs 50 \
  --eval-config "$EVAL_CONFIG" \
  --manifest "$MANIFEST"
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH"
echo "DENSE_TWOROOM_LR_HORIZON_TERMINAL_CONFIRM_COMPLETE horizon=$LR_HORIZON seed=$SEED"
