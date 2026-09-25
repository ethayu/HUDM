#!/usr/bin/env bash
#SBATCH --job-name=confirm_tr_h100_e20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:35:00
#SBATCH --array=0-9%4
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
MILESTONES=(14 19)
SEEDS=(0 1 2 42 100)
MILESTONE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
EPOCH="${MILESTONES[$MILESTONE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN_NAME="mwm_dense_tworoom_h100_continue20_20260914"
CHECKPOINT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports/epoch_$(printf '%03d' "$EPOCH")"
REPORT_ROOT="$ROOT/reports/research/dense_tworoom_h100_continue20_20260914"
OUTPUT_DIR="$REPORT_ROOT/terminal_n500/epoch_$(printf '%03d' "$EPOCH")/seed${SEED}"
CONFIG_PATH="$REPORT_ROOT/terminal_n500/epoch_$(printf '%03d' "$EPOCH")/seed${SEED}_config.yaml"
SOURCE_ROOT="$ROOT/reports/research/crossenv_scheduler_ablation_20260807/generated_configs"
EVAL_CONFIG="$SOURCE_ROOT/eval_snapshot_tworoom_seed${SEED}_n500.yaml"
MANIFEST="$ROOT/rollouts/manifests/release20260728_dense_k192_eval_size/tworoom_seed${SEED}_n500.json"

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT/terminal_n500/epoch_$(printf '%03d' "$EPOCH")"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite existing confirmation output: $OUTPUT_DIR" >&2
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
  --episodes 500 \
  --seed "$SEED"

echo "array_task=$SLURM_ARRAY_TASK_ID epoch=$EPOCH completed_epochs=$((EPOCH + 1)) seed=$SEED"
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
echo "DENSE_TWOROOM_H100_CONTINUE20_CONFIRM_COMPLETE epoch=$EPOCH seed=$SEED"
