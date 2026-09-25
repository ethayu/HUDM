#!/usr/bin/env bash
#SBATCH --job-name=train_singleton_lowk
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=16:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi
if [[ "$#" -ne 2 ]]; then
  echo "Usage: sbatch $0 ENV_SLUG K" >&2
  exit 2
fi

ENV_SLUG="$1"
K="$2"
case "$K" in
  48|72|96|120|144|168|192) ;;
  *)
    echo "ERROR: K must be one of 48,72,96,120,144,168,192; got $K" >&2
    exit 2
    ;;
esac

ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG_REL="configs/research/train_mwm_lewm_dense_${ENV_SLUG}_k48_72_96_120_144_168_192_paper10_sepopt_20260917.yaml"
CONFIG="$ROOT/$CONFIG_REL"
RUN_DATE="${MWM_RUN_DATE:-20260920}"
RUN_NAME="mwm_paper10_${ENV_SLUG}_k${K}_sepopt_retrain_${RUN_DATE}"

case "$ENV_SLUG" in
  tworoom)
    EXPECTED_ENV_ID="swm/TwoRoom-v1"
    EXPECTED_HISTORY=1
    DECODER_ACTIVATION_CHECKPOINTING="[]"
    ;;
  ogb_cube)
    EXPECTED_ENV_ID="swm/OGBCube-v0"
    EXPECTED_HISTORY=3
    DECODER_ACTIVATION_CHECKPOINTING="[0]"
    ;;
  reacher)
    EXPECTED_ENV_ID="swm/ReacherDMControl-v0"
    EXPECTED_HISTORY=3
    DECODER_ACTIVATION_CHECKPOINTING="[0]"
    ;;
  pusht)
    EXPECTED_ENV_ID="swm/PushT-v1"
    EXPECTED_HISTORY=3
    DECODER_ACTIVATION_CHECKPOINTING="[0]"
    ;;
  *)
    echo "ERROR: unsupported environment slug: $ENV_SLUG" >&2
    exit 2
    ;;
esac

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

OVERRIDES=(
  "model.K=[$K]"
  "train.run_name=$RUN_NAME"
  "decoder_training.activation_checkpointing=$DECODER_ACTIVATION_CHECKPOINTING"
)

"$PY" - "$CONFIG" "$K" "$RUN_NAME" "$EXPECTED_ENV_ID" "$EXPECTED_HISTORY" "$DECODER_ACTIVATION_CHECKPOINTING" <<'PY'
import sys
from omegaconf import OmegaConf
from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS

config, k, run_name, env_id, history, actckpt = sys.argv[1:]
overrides = [
    f"model.K=[{k}]",
    f"train.run_name={run_name}",
    f"decoder_training.activation_checkpointing={actckpt}",
]
cfg = load_config(DEFAULTS, config, overrides)
actual = {
    "env_id": str(cfg.env_id),
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "D": int(cfg.model.D),
    "history": int(cfg.model.history_size),
    "loss_history": int(cfg.loss.history_size),
    "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs),
    "resume": cfg.train.resume_checkpoint,
    "checkpoint_steps": int(cfg.train.checkpoint_every_n_train_steps),
    "save_top_k": int(cfg.train.save_top_k),
    "world_lr": float(cfg.optim.lr),
    "world_weight_decay": float(cfg.optim.weight_decay),
    "decoder_enabled": bool(cfg.decoder_training.enabled),
    "decoder_mode": str(cfg.decoder_training.mode),
    "decoder_activation_checkpointing": list(cfg.decoder_training.activation_checkpointing),
    "decoder_lr": float(cfg.decoder_training.lr),
    "decoder_weight_decay": float(cfg.decoder_training.weight_decay),
    "batch_size": int(cfg.train.batch_size),
    "horizon": int(cfg.train.horizon),
    "sigreg_weight": float(cfg.loss.sigreg_weight),
    "run_name": str(cfg.train.run_name),
}
expected = {
    "env_id": env_id,
    "seed": 3072,
    "K": [int(k)],
    "D": 192,
    "history": int(history),
    "loss_history": int(history),
    "epochs": 10,
    "lr_epochs": 10,
    "resume": None,
    "checkpoint_steps": 0,
    "save_top_k": -1,
    "world_lr": 5e-5,
    "world_weight_decay": 1e-3,
    "decoder_enabled": True,
    "decoder_mode": "separate_optimizer",
    "decoder_activation_checkpointing": list(OmegaConf.create(actckpt)),
    "decoder_lr": 5e-5,
    "decoder_weight_decay": 1e-3,
    "batch_size": 128,
    "horizon": 4,
    "sigreg_weight": 0.1,
    "run_name": run_name,
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
TRAIN_ARGS=("$CONFIG")
for override in "${OVERRIDES[@]}"; do TRAIN_ARGS+=(--set "$override"); done
"$PY" -m mwm.training.stable_wm "${TRAIN_ARGS[@]}"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
test -s "$TRAINER_CHECKPOINTS/last.ckpt"

"$PY" - "$TRAINER_CHECKPOINTS/last.ckpt" "$CHECKPOINT_ROOT/world_metadata.json" "$K" <<'PY'
import json
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
actual = {
    "epoch": int(state.get("epoch", -1)),
    "optimizer_count": len(state.get("optimizer_states", [])),
    "scheduler_count": len(state.get("lr_schedulers", [])),
    "levels": list(metadata.get("levels", [])),
    "fresh_init": bool(metadata.get("fresh_init", False)),
}
expected = {
    "epoch": 9,
    "optimizer_count": 2,
    "scheduler_count": 2,
    "levels": [int(sys.argv[3])],
    "fresh_init": True,
}
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
echo "SINGLETON_SEPOPT_TRAINING_AND_EXPORT_COMPLETE env=$ENV_SLUG K=$K run=$RUN_NAME"
