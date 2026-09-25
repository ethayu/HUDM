#!/usr/bin/env bash
#SBATCH --job-name=resume_dense_k48to192
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=2-12:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch $0 CONFIG" >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/$1"

cd "$ROOT"
mkdir -p logs
test -s "$CONFIG"

RUN_NAME="$($PY - "$CONFIG" <<'PY'
import sys
from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS
cfg = load_config(DEFAULTS, sys.argv[1], [])
print(str(cfg.train.run_name))
PY
)"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_CHECKPOINTS="$ROOT/logs/mwm_training/$RUN_NAME/csv_logs/version_0/checkpoints"
LAST_CHECKPOINT="$TRAINER_CHECKPOINTS/last.ckpt"
EPOCH_EXPORT_ROOT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports"

if [[ -s "$CHECKPOINT_ROOT/weights.pt" && -s "$EPOCH_EXPORT_ROOT/manifest.json" ]]; then
  echo "DENSE_K48TO192_RECOVERY_NOT_NEEDED run=$RUN_NAME"
  exit 0
fi
test -s "$LAST_CHECKPOINT"

RESUME_EPOCH="$($PY - "$LAST_CHECKPOINT" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = int(state.get("epoch", -1))
if epoch < 0 or epoch > 9 or len(state.get("optimizer_states", [])) != 2 or len(state.get("lr_schedulers", [])) != 2:
    raise SystemExit("checkpoint is not a valid two-optimizer 10-epoch resume point")
print(epoch)
PY
)"
echo "resume_epoch=$RESUME_EPOCH optimizer_count=2 scheduler_count=2"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

if (( RESUME_EPOCH < 9 )); then
  "$PY" -m mwm.training.stable_wm "$CONFIG" \
    --set train.clean_trainer_root=false \
    --set "train.resume_checkpoint=$LAST_CHECKPOINT"
elif [[ ! -s "$CHECKPOINT_ROOT/weights.pt" ]]; then
  "$PY" -m mwm.training.stable_wm "$CONFIG" \
    --export-from-lightning "$LAST_CHECKPOINT" \
    --output-dir "$CHECKPOINT_ROOT"
fi

if [[ -e "$EPOCH_EXPORT_ROOT" && ! -s "$EPOCH_EXPORT_ROOT/manifest.json" ]]; then
  mv "$EPOCH_EXPORT_ROOT" "${EPOCH_EXPORT_ROOT}_partial_${SLURM_JOB_ID}"
fi
"$PY" scripts/research/export_mwm_epoch_checkpoints.py \
  --config "$CONFIG" \
  --trainer-checkpoint-dir "$TRAINER_CHECKPOINTS" \
  --output-root "$EPOCH_EXPORT_ROOT" \
  --expected-epochs 10

test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$EPOCH_EXPORT_ROOT/manifest.json"
echo "DENSE_K48TO192_RECOVERY_COMPLETE run=$RUN_NAME"

