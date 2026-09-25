#!/usr/bin/env bash
#SBATCH --job-name=train_reacher_up_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/configs/research/train_mwm_lewm_reacher_k192_paper10_nativeh5_nodecoder_init3072_fit0_20260806.yaml"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mwm-reacher-upstream-smoke-${SLURM_JOB_ID}.XXXXXX")"
SMOKE_RUN_NAME="upstream_recipe_smoke_${SLURM_JOB_ID}"
SMOKE_TRAINER_ROOT="$ROOT/logs/mwm_training/$SMOKE_RUN_NAME"
cleanup() {
  rm -rf -- "$SMOKE_ROOT"
  rm -rf -- "$SMOKE_TRAINER_ROOT"
}
trap cleanup EXIT

cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m mwm.training.stable_wm "$CONFIG" \
  --set data.training_source.path=data/upstream/reacher.lance \
  --set data.training_source.format=lance \
  --set train.checkpoint_dir="$SMOKE_ROOT/checkpoints" \
  --set train.run_name="$SMOKE_RUN_NAME" \
  --set train.batch_size=2 \
  --set train.num_workers=0 \
  --set train.limit_train_batches=2 \
  --set train.limit_val_batches=1 \
  --set train.checkpoint_every_n_train_steps=0 \
  --set train.save_top_k=0 \
  --set schedule.max_epochs=1 \
  --set schedule.lr_max_epochs=1

test -s "$SMOKE_ROOT/checkpoints/$SMOKE_RUN_NAME/config.json"
test -s "$SMOKE_ROOT/checkpoints/$SMOKE_RUN_NAME/weights.pt"
test -s "$SMOKE_ROOT/checkpoints/$SMOKE_RUN_NAME/world_metadata.json"
echo "STRICT_RECIPE_SMOKE_PASSED"
