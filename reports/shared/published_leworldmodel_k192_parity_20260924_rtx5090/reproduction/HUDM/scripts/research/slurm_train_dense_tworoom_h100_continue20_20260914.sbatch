#!/usr/bin/env bash
#SBATCH --job-name=train_tr_h100_e20
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=16:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="$ROOT/configs/research/train_mwm_lewm_dense_tworoom_h100_continue20_20260914.yaml"
RUN_NAME="mwm_dense_tworoom_h100_continue20_20260914"
RESUME="$ROOT/reports/research/dense_tworoom_h100_continue20_20260914/resume_inputs/h100/epoch=9-step=116820.ckpt"
SOURCE_EXPORT="$ROOT/checkpoints_mwm/mwm_dense_tworoom_lr_horizon_h100_epochs_20260913_epoch_exports/epoch_009"
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
test -s "$CONFIG"
test -s "$RESUME"
for name in config.json weights.pt world_metadata.json; do
  test -s "$SOURCE_EXPORT/$name"
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

"$PY" - "$CONFIG" "$RESUME" "$RUN_NAME" <<'PY'
import sys
from pathlib import Path

import torch

from mwm.config_cli import load_config
from mwm.io import file_sha256
from mwm.training.stable_wm_config import DEFAULTS

config_path, resume_path, run_name = sys.argv[1:]
cfg = load_config(DEFAULTS, config_path, [])
actual = {
    "env_id": str(cfg.env_id),
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs),
    "checkpoint_steps": int(cfg.train.checkpoint_every_n_train_steps),
    "save_top_k": int(cfg.train.save_top_k),
    "run_name": str(cfg.train.run_name),
    "resume": str(Path(cfg.train.resume_checkpoint).resolve()),
}
expected = {
    "env_id": "swm/TwoRoom-v1",
    "seed": 3072,
    "K": [96, 120, 144, 168, 192],
    "epochs": 20,
    "lr_epochs": 100,
    "checkpoint_steps": 0,
    "save_top_k": -1,
    "run_name": run_name,
    "resume": str(Path(resume_path).resolve()),
}
print({"training_contract": actual})
if actual != expected:
    raise SystemExit(f"training contract mismatch: {actual} != {expected}")

state = torch.load(resume_path, map_location="cpu", weights_only=False)
schedulers = state.get("lr_schedulers", [])
resume_contract = {
    "epoch": int(state.get("epoch", -1)),
    "global_step": int(state.get("global_step", -1)),
    "optimizer_count": len(state.get("optimizer_states", [])),
    "scheduler_count": len(schedulers),
    "scheduler_last_epochs": [int(item.get("last_epoch", -1)) for item in schedulers],
    "scheduler_max_steps": [int(item.get("max_steps", -1)) for item in schedulers],
    "sha256": file_sha256(Path(resume_path)),
}
expected_resume = {
    "epoch": 9,
    "global_step": 116820,
    "optimizer_count": 2,
    "scheduler_count": 2,
    "scheduler_last_epochs": [58410, 58410],
    "scheduler_max_steps": [584100, 584100],
}
print({"resume_contract": resume_contract})
if {key: resume_contract[key] for key in expected_resume} != expected_resume:
    raise SystemExit(f"resume checkpoint contract mismatch: {resume_contract}")
PY

hostname
nvidia-smi
"$PY" -m mwm.data.verify "$CONFIG"
"$PY" -m mwm.training.stable_wm "$CONFIG"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
test -s "$TRAINER_CHECKPOINTS/last.ckpt"

"$PY" - "$TRAINER_CHECKPOINTS" <<'PY'
import re
import sys
from pathlib import Path

pattern = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")
found = []
for path in Path(sys.argv[1]).glob("epoch=*-step=*.ckpt"):
    match = pattern.match(path.name)
    if match:
        found.append((int(match.group(1)), int(match.group(2))))
found.sort()
expected = [(epoch, (epoch + 1) * 11682) for epoch in range(10, 20)]
print({"numbered_checkpoints": found})
if found != expected:
    raise SystemExit(f"expected numbered checkpoints {expected}, found {found}")
PY

"$PY" scripts/research/export_mwm_epoch_checkpoints.py \
  --config "$CONFIG" \
  --trainer-checkpoint-dir "$TRAINER_CHECKPOINTS" \
  --output-root "$EPOCH_EXPORT_ROOT" \
  --expected-start-epoch 10 \
  --expected-epochs 10

test -s "$EPOCH_EXPORT_ROOT/manifest.json"
"$PY" - "$TRAINER_CHECKPOINTS/last.ckpt" "$EPOCH_EXPORT_ROOT/manifest.json" <<'PY'
import json
import sys

import torch

checkpoint_path, manifest_path = sys.argv[1:]
state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
schedulers = state.get("lr_schedulers", [])
actual = {
    "epoch": int(state.get("epoch", -1)),
    "global_step": int(state.get("global_step", -1)),
    "optimizer_count": len(state.get("optimizer_states", [])),
    "scheduler_count": len(schedulers),
    "scheduler_last_epochs": [int(item.get("last_epoch", -1)) for item in schedulers],
    "scheduler_max_steps": [int(item.get("max_steps", -1)) for item in schedulers],
}
expected = {
    "epoch": 19,
    "global_step": 233640,
    "optimizer_count": 2,
    "scheduler_count": 2,
    "scheduler_last_epochs": [116820, 116820],
    "scheduler_max_steps": [584100, 584100],
}
print({"terminal_checkpoint_contract": actual})
if actual != expected:
    raise SystemExit(f"terminal checkpoint contract mismatch: {actual} != {expected}")
manifest = json.load(open(manifest_path, encoding="utf-8"))
if [row["epoch"] for row in manifest] != list(range(10, 20)):
    raise SystemExit("canonical epoch export manifest does not contain epochs 10 through 19")
PY

echo "DENSE_TWOROOM_H100_CONTINUE20_TRAINING_AND_EXPORT_COMPLETE"
