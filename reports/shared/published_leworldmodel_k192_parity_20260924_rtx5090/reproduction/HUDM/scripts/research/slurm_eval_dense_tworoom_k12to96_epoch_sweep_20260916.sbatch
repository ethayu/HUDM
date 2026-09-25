#!/usr/bin/env bash
#SBATCH --job-name=eval_tr_k96_epochs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%10
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
EPOCH="$SLURM_ARRAY_TASK_ID"
EPOCH_PADDED="$(printf '%03d' "$EPOCH")"
CHECKPOINT="$ROOT/checkpoints_mwm/mwm_paper10_tworoom_k12to96_sepopt_20260915_epoch_exports/epoch_$EPOCH_PADDED"
REPORT_ROOT="$ROOT/reports/research/dense_tworoom_k12to96_epoch_sweep_20260916"
OUTPUT_DIR="$REPORT_ROOT/epoch_${EPOCH_PADDED}_k96_n50_seed42"
CONFIG_PATH="$REPORT_ROOT/epoch_${EPOCH_PADDED}_k96_n50_seed42_config.yaml"
EVAL_CONFIG="$ROOT/configs/eval/research_tworoom_exact_seed42_n50_20260913.yaml"
MANIFEST="$ROOT/rollouts/manifests/tworoom_paper_seed42.json"
RUNNER="$ROOT/scripts/research/eval_dense_tworoom_k12to96_quick_20260916.py"

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite existing evaluation output: $OUTPUT_DIR" >&2
  exit 3
fi
for name in config.json weights.pt world_metadata.json; do
  test -s "$CHECKPOINT/$name"
done
test -s "$EVAL_CONFIG"
test -s "$MANIFEST"
test -s "$RUNNER"
"$PY" scripts/research/validate_tworoom_manifest_contract_20260913.py \
  --manifest "$MANIFEST" \
  --episodes 50 \
  --seed 42

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "array_task=$SLURM_ARRAY_TASK_ID epoch=$EPOCH K=96"
hostname
nvidia-smi
"$PY" "$RUNNER" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --levels 96 \
  --episodes 50 \
  --num-envs 50 \
  --seed 42 \
  --eval-config "$EVAL_CONFIG" \
  --manifest "$MANIFEST" \
  --write-config-only
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH" --static-only
"$PY" "$RUNNER" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --levels 96 \
  --episodes 50 \
  --num-envs 50 \
  --seed 42 \
  --eval-config "$EVAL_CONFIG" \
  --manifest "$MANIFEST"
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH"
test -s "$OUTPUT_DIR/summary.json"
test -s "$OUTPUT_DIR/summary.csv"
echo "DENSE_TWOROOM_K96_EPOCH_EVAL_COMPLETE epoch=$EPOCH episodes=50 seed=42"
