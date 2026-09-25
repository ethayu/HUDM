#!/usr/bin/env bash
#SBATCH --job-name=train_dense_reacher_sched
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/configs/research/train_mwm_lewm_dense_reacher_scheduler_ablation_seed3072_20260809.yaml"
HORIZONS=(10 100)
LR_HORIZON="${HORIZONS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="mwm_dense_reacher_spt_isolateddec_h${LR_HORIZON}_seed3072_20260809"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"

cd "$ROOT"
mkdir -p logs
if [[ -e "$CHECKPOINT_ROOT" || -e "$TRAINER_ROOT" ]]; then
  echo "ERROR: refusing to overwrite an existing scheduler-ablation run." >&2
  echo "checkpoint: $CHECKPOINT_ROOT" >&2
  echo "trainer: $TRAINER_ROOT" >&2
  exit 3
fi

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
    "data": str(cfg.data.path),
    "format": str(cfg.data.format),
    "K": list(cfg.model.K),
    "D": int(cfg.model.D),
    "batch_size": int(cfg.train.batch_size),
    "precision": str(cfg.train.precision),
    "matmul_precision": str(cfg.train.matmul_precision),
    "sigreg": float(cfg.loss.sigreg_weight),
    "decoder": bool(cfg.decoder_training.enabled),
    "decoder_mode": str(cfg.decoder_training.mode),
    "detach_decoder": bool(cfg.mwm.loss_terms.reconstructor_detach_encoder),
    "decoder_encoder_loss": bool(cfg.mwm.loss_terms.reconstructor_contributes_to_encoder_loss),
    "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs),
    "lr": float(cfg.optim.lr),
    "weight_decay": float(cfg.optim.weight_decay),
    "decoder_lr": float(cfg.decoder_training.lr),
    "decoder_weight_decay": float(cfg.decoder_training.weight_decay),
    "world_clip": float(cfg.train.gradient_clip_val),
    "decoder_clip": float(cfg.decoder_training.gradient_clip_val),
    "run_name": str(cfg.train.run_name),
    "resume": cfg.train.resume_checkpoint,
}
expected = {
    "env_id": "swm/ReacherDMControl-v0",
    "seed": 3072,
    "data": "data/upstream/reacher.lance",
    "format": "lance",
    "K": [96, 120, 144, 168, 192],
    "D": 192,
    "batch_size": 128,
    "precision": "bf16",
    "matmul_precision": "high",
    "sigreg": 0.10,
    "decoder": True,
    "decoder_mode": "separate_optimizer",
    "detach_decoder": True,
    "decoder_encoder_loss": False,
    "epochs": 10,
    "lr_epochs": int(horizon),
    "lr": 0.00005,
    "weight_decay": 0.001,
    "decoder_lr": 0.00005,
    "decoder_weight_decay": 0.001,
    "world_clip": 1.0,
    "decoder_clip": 1.0,
    "run_name": run_name,
    "resume": None,
}
print(actual)
if actual != expected:
    raise SystemExit(f"ERROR: dense Reacher scheduler-ablation config mismatch: {actual} != {expected}")
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
test -s "$TRAINER_ROOT/csv_logs/version_0/checkpoints/last.ckpt"
