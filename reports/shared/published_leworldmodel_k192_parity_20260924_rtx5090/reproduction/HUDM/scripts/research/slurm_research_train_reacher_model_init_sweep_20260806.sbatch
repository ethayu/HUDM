#!/usr/bin/env bash
#SBATCH --job-name=train_reacher_init
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%5
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch as an array." >&2
  exit 2
fi
for name in MWM_TRAIN_CONFIG MWM_EXPECTED_LR_MAX_EPOCHS MWM_INIT_SWEEP_TAG; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be exported at submission." >&2
    exit 2
  fi
done

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
INIT_SEEDS=(0 1 2 42 100)
MODEL_INIT_SEED="${INIT_SEEDS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="$MWM_TRAIN_CONFIG"
LR_MAX_EPOCHS="$MWM_EXPECTED_LR_MAX_EPOCHS"
RUN_NAME="mwm_reacher_k192_${MWM_INIT_SWEEP_TAG}_nativeh5_nodecoder_init${MODEL_INIT_SEED}_fit0_20260806"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"
PARITY_REPORT="$ROOT/reports/research/single_level_upstream_parity_20260806/report.json"
SOURCE_ARCHIVE="$ROOT/data/upstream/reacher.tar.zst"
SOURCE_ARCHIVE_SHA256="4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2"
SOURCE_H5_SHA256="85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06"

case "$LR_MAX_EPOCHS" in
  10)
    EXPECTED_CONFIG="$ROOT/configs/research/train_mwm_lewm_reacher_k192_paper10_nativeh5_nodecoder_init3072_fit0_20260806.yaml"
    ;;
  100)
    EXPECTED_CONFIG="$ROOT/configs/research/train_mwm_lewm_reacher_k192_epoch10_horizon100_nativeh5_nodecoder_init3072_fit0_20260806.yaml"
    ;;
  *)
    echo "ERROR: scheduler horizon must be exactly 10 or 100, got $LR_MAX_EPOCHS." >&2
    exit 2
    ;;
esac
case "$CONFIG" in
  /*) ;;
  *) CONFIG="$ROOT/$CONFIG" ;;
esac
if [[ "$CONFIG" != "$EXPECTED_CONFIG" ]]; then
  echo "ERROR: refusing non-canonical sweep config: $CONFIG" >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs
if [[ -e "$CHECKPOINT_ROOT" || -e "$TRAINER_ROOT" ]]; then
  echo "ERROR: refusing to overwrite an existing fresh-run path." >&2
  echo "checkpoint: $CHECKPOINT_ROOT" >&2
  echo "trainer: $TRAINER_ROOT" >&2
  exit 3
fi

"$PY" - "$PARITY_REPORT" "$CONFIG" "$MODEL_INIT_SEED" "$RUN_NAME" "$LR_MAX_EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS

report_path, config_path, init_seed, run_name, lr_max_epochs = sys.argv[1:]
payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
if payload.get("status") != "pass":
    raise SystemExit(f"ERROR: architecture parity did not pass: {payload}")
cfg = load_config(
    DEFAULTS,
    config_path,
    [f"train.model_init_seed={init_seed}", f"train.run_name={run_name}"],
)
actual = {
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "batch_size": int(cfg.train.batch_size),
    "training_source_format": str(cfg.data.training_source.format),
    "sigreg": float(cfg.loss.sigreg_weight),
    "lr": float(cfg.optim.lr),
    "weight_decay": float(cfg.optim.weight_decay),
    "gradient_clip_val": float(cfg.train.gradient_clip_val),
    "matmul_precision": str(cfg.train.matmul_precision),
    "epochs": int(cfg.schedule.max_epochs),
    "lr_epochs": int(cfg.schedule.lr_max_epochs),
    "decoder": bool(cfg.decoder_training.enabled),
    "model_init_seed": int(cfg.train.model_init_seed),
    "fit_seed": int(cfg.train.fit_seed),
    "resume": cfg.train.resume_checkpoint,
    "run_name": str(cfg.train.run_name),
}
expected = {
    "seed": 3072,
    "K": [192],
    "batch_size": 128,
    "training_source_format": "hdf5",
    "sigreg": 0.09,
    "lr": 0.00005,
    "weight_decay": 0.001,
    "gradient_clip_val": 1.0,
    "matmul_precision": "highest",
    "epochs": 10,
    "lr_epochs": int(lr_max_epochs),
    "decoder": False,
    "model_init_seed": int(init_seed),
    "fit_seed": 0,
    "resume": None,
    "run_name": run_name,
}
print(actual)
if actual != expected:
    raise SystemExit(f"ERROR: construction-seed sweep config mismatch: {actual} != {expected}")
PY

test -s "$SOURCE_ARCHIVE"
actual_archive_sha256="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
if [[ "$actual_archive_sha256" != "$SOURCE_ARCHIVE_SHA256" ]]; then
  echo "ERROR: source archive hash mismatch: $actual_archive_sha256" >&2
  exit 4
fi

LOCAL_SOURCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mwm-reacher-init-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  rm -rf -- "$LOCAL_SOURCE_ROOT"
}
trap cleanup EXIT
available_kib="$(df -Pk "$LOCAL_SOURCE_ROOT" | awk 'NR==2 {print $4}')"
required_kib=$((120 * 1024 * 1024))
if (( available_kib < required_kib )); then
  echo "ERROR: node-local training source needs 120 GiB; only ${available_kib} KiB are available." >&2
  exit 5
fi
tar --use-compress-program=unzstd -xf "$SOURCE_ARCHIVE" -C "$LOCAL_SOURCE_ROOT" reacher.h5
LOCAL_H5="$LOCAL_SOURCE_ROOT/reacher.h5"
test -s "$LOCAL_H5"
actual_h5_sha256="$(sha256sum "$LOCAL_H5" | awk '{print $1}')"
if [[ "$actual_h5_sha256" != "$SOURCE_H5_SHA256" ]]; then
  echo "ERROR: extracted HDF5 hash mismatch: $actual_h5_sha256" >&2
  exit 6
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

hostname
nvidia-smi
"$PY" -m mwm.data.verify "$CONFIG"
"$PY" -m mwm.training.stable_wm "$CONFIG" \
  --set "data.training_source.path=$LOCAL_H5" \
  --set "train.model_init_seed=$MODEL_INIT_SEED" \
  --set "train.run_name=$RUN_NAME"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
LIGHTNING_CHECKPOINT="$TRAINER_ROOT/csv_logs/version_0/checkpoints/last.ckpt"
test -s "$LIGHTNING_CHECKPOINT"
"$PY" - "$LIGHTNING_CHECKPOINT" "$MODEL_INIT_SEED" "$RUN_NAME" <<'PY'
import sys
from pathlib import Path

import torch

path, init_seed, run_name = sys.argv[1:]
checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
cfg = checkpoint.get("hyper_parameters", {})
if not isinstance(cfg, dict) or not cfg:
    raise SystemExit("ERROR: final Lightning checkpoint has no resolved training config")
if int(cfg["train"]["model_init_seed"]) != int(init_seed):
    raise SystemExit("ERROR: final Lightning checkpoint has the wrong model-init seed")
if cfg["train"]["run_name"] != run_name:
    raise SystemExit("ERROR: final Lightning checkpoint has the wrong run name")
PY
