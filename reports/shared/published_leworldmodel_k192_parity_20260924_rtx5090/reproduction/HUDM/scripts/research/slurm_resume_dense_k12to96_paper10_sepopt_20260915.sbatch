#!/usr/bin/env bash
#SBATCH --job-name=resume_dense_k12to96
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || "$#" -ne 1 ]]; then
  echo "Usage: sbatch $0 CONFIG" >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/$1"

cd "$ROOT"
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

# The dependency is failure-only, but retain an idempotent success guard in
# case scheduler semantics or a manual invocation causes this job to run.
if [[ -s "$CHECKPOINT_ROOT/config.json" \
   && -s "$CHECKPOINT_ROOT/weights.pt" \
   && -s "$CHECKPOINT_ROOT/world_metadata.json" \
   && -s "$EPOCH_EXPORT_ROOT/manifest.json" ]]; then
  echo "DENSE_K12TO96_RECOVERY_NOT_NEEDED run=$RUN_NAME"
  exit 0
fi

if [[ ! -s "$LAST_CHECKPOINT" ]]; then
  echo "ERROR: no resumable Lightning checkpoint at $LAST_CHECKPOINT" >&2
  exit 3
fi

RESUME_EPOCH="$($PY - "$LAST_CHECKPOINT" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = int(state.get("epoch", -1))
optimizer_count = len(state.get("optimizer_states", []))
scheduler_count = len(state.get("lr_schedulers", []))
if epoch < 0 or epoch > 9:
    raise SystemExit(f"checkpoint epoch {epoch} is outside the expected 10-epoch run")
if optimizer_count != 2 or scheduler_count != 2:
    raise SystemExit("resume checkpoint does not preserve both optimizer/scheduler states")
print(epoch)
PY
)"
echo "resume_epoch=$RESUME_EPOCH optimizer_count=2 scheduler_count=2"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

if (( RESUME_EPOCH < 9 )); then
  $PY -m mwm.training.stable_wm "$CONFIG" \
    --set train.clean_trainer_root=false \
    --set "train.resume_checkpoint=$LAST_CHECKPOINT"
elif [[ ! -s "$CHECKPOINT_ROOT/weights.pt" ]]; then
  $PY -m mwm.training.stable_wm "$CONFIG" \
    --export-from-lightning "$LAST_CHECKPOINT" \
    --output-dir "$CHECKPOINT_ROOT"
fi

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
test -s "$LAST_CHECKPOINT"

$PY - "$LAST_CHECKPOINT" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actual = {
    "epoch": int(state.get("epoch", -1)),
    "optimizer_count": len(state.get("optimizer_states", [])),
    "scheduler_count": len(state.get("lr_schedulers", [])),
}
expected = {"epoch": 9, "optimizer_count": 2, "scheduler_count": 2}
print({"terminal_checkpoint_contract": actual})
if actual != expected:
    raise SystemExit(f"terminal checkpoint contract mismatch: {actual} != {expected}")
PY

$PY scripts/research/export_mwm_epoch_checkpoints.py \
  --config "$CONFIG" \
  --trainer-checkpoint-dir "$TRAINER_CHECKPOINTS" \
  --output-root "$EPOCH_EXPORT_ROOT" \
  --expected-epochs 10

test -s "$EPOCH_EXPORT_ROOT/manifest.json"
echo "DENSE_K12TO96_RECOVERY_TRAINING_AND_EXPORT_COMPLETE run=$RUN_NAME"
