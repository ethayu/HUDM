#!/usr/bin/env bash
#SBATCH --job-name=train_tr_k48to192
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=11:30:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/configs/research/train_mwm_lewm_dense_tworoom_k48_72_96_120_144_168_192_paper10_sepopt_20260917.yaml"
RUN_NAME="mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"
TRAINER_CHECKPOINTS="$TRAINER_ROOT/csv_logs/version_0/checkpoints"
EPOCH_EXPORT_ROOT="$ROOT/checkpoints_mwm/${RUN_NAME}_epoch_exports"

cd "$ROOT"
mkdir -p logs
test -s "$CONFIG"
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
actual = {
    "seed": int(cfg.seed), "env_id": str(cfg.env_id), "K": list(cfg.model.K),
    "D": int(cfg.model.D), "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs), "resume": cfg.train.resume_checkpoint,
    "save_top_k": int(cfg.train.save_top_k), "world_lr": float(cfg.optim.lr),
    "world_weight_decay": float(cfg.optim.weight_decay),
    "decoder_enabled": bool(cfg.decoder_training.enabled),
    "decoder_mode": str(cfg.decoder_training.mode), "decoder_lr": float(cfg.decoder_training.lr),
    "decoder_weight_decay": float(cfg.decoder_training.weight_decay),
    "batch_size": int(cfg.train.batch_size), "horizon": int(cfg.train.horizon),
    "history_size": int(cfg.model.history_size), "sigreg_weight": float(cfg.loss.sigreg_weight),
    "run_name": str(cfg.train.run_name),
}
expected = {
    "seed": 3072, "env_id": "swm/TwoRoom-v1", "K": [48, 72, 96, 120, 144, 168, 192],
    "D": 192, "epochs": 10, "lr_epochs": 10, "resume": None, "save_top_k": -1,
    "world_lr": 5e-5, "world_weight_decay": 1e-3, "decoder_enabled": True,
    "decoder_mode": "separate_optimizer", "decoder_lr": 5e-5,
    "decoder_weight_decay": 1e-3, "batch_size": 128, "horizon": 4,
    "history_size": 1, "sigreg_weight": 0.1,
    "run_name": "mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917",
}
print({"training_contract": actual})
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
"$PY" -m mwm.training.stable_wm "$CONFIG"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
test -s "$TRAINER_CHECKPOINTS/last.ckpt"

"$PY" - "$TRAINER_CHECKPOINTS/last.ckpt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actual = {"epoch": int(state.get("epoch", -1)), "optimizer_count": len(state.get("optimizer_states", [])), "scheduler_count": len(state.get("lr_schedulers", []))}
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
echo "DENSE_TWOROOM_K48_72_96_120_144_168_192_COMPLETE run=$RUN_NAME"

