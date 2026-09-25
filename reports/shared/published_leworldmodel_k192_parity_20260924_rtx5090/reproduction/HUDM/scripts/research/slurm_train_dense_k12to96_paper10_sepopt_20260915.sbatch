#!/usr/bin/env bash
#SBATCH --job-name=train_dense_k12to96
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch $0 CONFIG" >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"
TRAINER_CHECKPOINTS="$TRAINER_ROOT/csv_logs/version_0/checkpoints"
EPOCH_EXPORT_ROOT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports"

for path in "$CHECKPOINT_ROOT" "$TRAINER_ROOT" "$EPOCH_EXPORT_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: refusing to overwrite existing path: $path" >&2
    exit 3
  fi
done

"$PY" - "$CONFIG" <<'PY'
import sys

from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS

cfg = load_config(DEFAULTS, sys.argv[1], [])
decoder_activation_checkpointing = cfg.decoder_training.get("activation_checkpointing", False)
if isinstance(decoder_activation_checkpointing, bool):
    decoder_activation_checkpointing = [] if not decoder_activation_checkpointing else list(range(len(cfg.model.K)))
else:
    decoder_activation_checkpointing = list(decoder_activation_checkpointing)
actual = {
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs),
    "resume": cfg.train.resume_checkpoint,
    "checkpoint_steps": int(cfg.train.checkpoint_every_n_train_steps),
    "save_top_k": int(cfg.train.save_top_k),
    "world_lr": float(cfg.optim.lr),
    "world_weight_decay": float(cfg.optim.weight_decay),
    "decoder_enabled": bool(cfg.decoder_training.enabled),
    "decoder_mode": str(cfg.decoder_training.mode),
    "decoder_activation_checkpointing": decoder_activation_checkpointing,
    "decoder_lr": float(cfg.decoder_training.lr),
    "decoder_weight_decay": float(cfg.decoder_training.weight_decay),
    "batch_size": int(cfg.train.batch_size),
    "horizon": int(cfg.train.horizon),
    "sigreg_weight": float(cfg.loss.sigreg_weight),
}
expected = {
    "seed": 3072,
    "K": [12, 24, 36, 48, 60, 72, 84, 96],
    "epochs": 10,
    "lr_epochs": 10,
    "resume": None,
    "checkpoint_steps": 0,
    "save_top_k": -1,
    "world_lr": 5e-5,
    "world_weight_decay": 1e-3,
    "decoder_enabled": True,
    "decoder_mode": "separate_optimizer",
    "decoder_activation_checkpointing": [] if str(cfg.env_id) == "swm/TwoRoom-v1" else [0, 1],
    "decoder_lr": 5e-5,
    "decoder_weight_decay": 1e-3,
    "batch_size": 128,
    "horizon": 4,
    "sigreg_weight": 0.1,
}
print({"training_contract": actual, "env_id": str(cfg.env_id), "run_name": str(cfg.train.run_name)})
if actual != expected:
    raise SystemExit(f"training contract mismatch: {actual} != {expected}")
expected_history = 1 if str(cfg.env_id) == "swm/TwoRoom-v1" else 3
if int(cfg.model.history_size) != expected_history or int(cfg.loss.history_size) != expected_history:
    raise SystemExit(f"history-size contract mismatch for {cfg.env_id}")
PY

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

hostname
nvidia-smi
"$PY" -m mwm.data.verify "$CONFIG"
"$PY" -m mwm.training.stable_wm "$CONFIG"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
test -s "$TRAINER_CHECKPOINTS/last.ckpt"

"$PY" - "$TRAINER_CHECKPOINTS/last.ckpt" <<'PY'
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

"$PY" scripts/research/export_mwm_epoch_checkpoints.py \
  --config "$CONFIG" \
  --trainer-checkpoint-dir "$TRAINER_CHECKPOINTS" \
  --output-root "$EPOCH_EXPORT_ROOT" \
  --expected-epochs 10

test -s "$EPOCH_EXPORT_ROOT/manifest.json"
echo "DENSE_K12TO96_PAPER10_SEPOPT_TRAINING_AND_EXPORT_COMPLETE run=$RUN_NAME"
