#!/usr/bin/env bash
#SBATCH --job-name=train_tr_dense_lrh
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/configs/research/train_mwm_lewm_dense_tworoom_lr_horizon_epoch_exports_20260913.yaml"
HORIZONS=(10 100)
LR_HORIZON="${HORIZONS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="mwm_dense_tworoom_lr_horizon_h${LR_HORIZON}_epochs_20260913"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"
TRAINER_CHECKPOINTS="$TRAINER_ROOT/csv_logs/version_0/checkpoints"
EPOCH_EXPORT_ROOT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports"

cd "$ROOT"
mkdir -p logs
for path in "$CHECKPOINT_ROOT" "$TRAINER_ROOT" "$EPOCH_EXPORT_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: refusing to overwrite existing path: $path" >&2
    exit 3
  fi
done

"$PY" - "$CONFIG" "$LR_HORIZON" "$RUN_NAME" <<'PY'
import sys

from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS

config_path, horizon, run_name = sys.argv[1:]
cfg = load_config(
    DEFAULTS,
    config_path,
    [f"schedule.lr_max_epochs={horizon}", f"train.run_name={run_name}"],
)
actual = {
    "env_id": str(cfg.env_id),
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs),
    "checkpoint_steps": int(cfg.train.checkpoint_every_n_train_steps),
    "save_top_k": int(cfg.train.save_top_k),
    "run_name": str(cfg.train.run_name),
}
expected = {
    "env_id": "swm/TwoRoom-v1",
    "seed": 3072,
    "K": [96, 120, 144, 168, 192],
    "epochs": 10,
    "lr_epochs": int(horizon),
    "checkpoint_steps": 0,
    "save_top_k": -1,
    "run_name": run_name,
}
print(actual)
if actual != expected:
    raise SystemExit(f"training contract mismatch: {actual} != {expected}")
PY

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

hostname
nvidia-smi
"$PY" -m mwm.data.verify "$CONFIG"
"$PY" -m mwm.training.stable_wm "$CONFIG" \
  --set "schedule.lr_max_epochs=$LR_HORIZON" \
  --set "train.run_name=$RUN_NAME"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
test -s "$TRAINER_CHECKPOINTS/last.ckpt"

"$PY" scripts/research/export_mwm_epoch_checkpoints.py \
  --config "$CONFIG" \
  --trainer-checkpoint-dir "$TRAINER_CHECKPOINTS" \
  --output-root "$EPOCH_EXPORT_ROOT" \
  --expected-epochs 10

test -s "$EPOCH_EXPORT_ROOT/manifest.json"
echo "DENSE_TWOROOM_LR_HORIZON_TRAINING_AND_EXPORT_COMPLETE horizon=$LR_HORIZON"
