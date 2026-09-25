#!/usr/bin/env bash
#SBATCH --job-name=train_reacher_paper10
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CONFIG="${MWM_TRAIN_CONFIG:-$ROOT/configs/research/train_mwm_lewm_reacher_k192_paper10_nativeh5_nodecoder_init3072_fit0_20260806.yaml}"
PARITY_REPORT="$ROOT/reports/research/single_level_upstream_parity_20260806/report.json"
SOURCE_ARCHIVE="$ROOT/data/upstream/reacher.tar.zst"
SOURCE_ARCHIVE_SHA256="4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2"
SOURCE_H5_SHA256="85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06"
RUN_NAME="${MWM_TRAIN_RUN_NAME:-mwm_reacher_k192_paper10_nativeh5_nodecoder_init3072_fit0_20260806}"
EXPECTED_MAX_EPOCHS="${MWM_EXPECTED_MAX_EPOCHS:-10}"
EXPECTED_LR_MAX_EPOCHS="${MWM_EXPECTED_LR_MAX_EPOCHS:-10}"
EXPECTED_CHECKPOINT_STEPS="${MWM_EXPECTED_CHECKPOINT_STEPS:-63980}"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"

cd "$ROOT"
mkdir -p logs

"$PY" - "$PARITY_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ERROR: missing architecture parity report: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "pass":
    raise SystemExit(f"ERROR: architecture parity did not pass: {payload}")
PY

if [[ -e "$CHECKPOINT_ROOT" || -e "$TRAINER_ROOT" ]]; then
  echo "ERROR: refusing to overwrite an existing fresh-run path." >&2
  echo "checkpoint: $CHECKPOINT_ROOT" >&2
  echo "trainer: $TRAINER_ROOT" >&2
  exit 3
fi

test -s "$SOURCE_ARCHIVE"
actual_archive_sha256="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
if [[ "$actual_archive_sha256" != "$SOURCE_ARCHIVE_SHA256" ]]; then
  echo "ERROR: source archive hash mismatch: $actual_archive_sha256" >&2
  exit 4
fi

LOCAL_SOURCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mwm-reacher-${SLURM_JOB_ID}.XXXXXX")"
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
export CONFIG RUN_NAME EXPECTED_MAX_EPOCHS EXPECTED_LR_MAX_EPOCHS EXPECTED_CHECKPOINT_STEPS

hostname
nvidia-smi
"$PY" - <<'PY'
import os

from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS

path = os.environ["CONFIG"]
cfg = load_config(DEFAULTS, path)
actual = {
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "batch_size": int(cfg.train.batch_size),
    "history_size": int(cfg.model.history_size),
    "num_preds": int(cfg.model.num_preds),
    "keys_to_load": list(cfg.data.keys_to_load),
    "training_source_format": str(cfg.data.training_source.format),
    "training_source_archive": str(cfg.data.training_source.archive),
    "training_source_archive_sha256": str(cfg.data.training_source.archive_sha256),
    "training_source_extracted_sha256": str(cfg.data.training_source.extracted_sha256),
    "sigreg": float(cfg.loss.sigreg_weight),
    "lr": float(cfg.optim.lr),
    "weight_decay": float(cfg.optim.weight_decay),
    "gradient_clip_val": float(cfg.train.gradient_clip_val),
    "matmul_precision": str(cfg.train.matmul_precision),
    "checkpoint_steps": int(cfg.train.checkpoint_every_n_train_steps),
    "save_top_k": int(cfg.train.save_top_k),
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
    "history_size": 3,
    "num_preds": 1,
    "keys_to_load": ["pixels", "action", "observation"],
    "training_source_format": "hdf5",
    "training_source_archive": "data/upstream/reacher.tar.zst",
    "training_source_archive_sha256": "4ff2385e49712caa89f21b8e0a246e2614b621d3f22cf2d1224d845e879a1cc2",
    "training_source_extracted_sha256": "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06",
    "sigreg": 0.09,
    "lr": 0.00005,
    "weight_decay": 0.001,
    "gradient_clip_val": 1.0,
    "matmul_precision": "highest",
    "checkpoint_steps": int(os.environ["EXPECTED_CHECKPOINT_STEPS"]),
    "save_top_k": -1,
    "epochs": int(os.environ["EXPECTED_MAX_EPOCHS"]),
    "lr_epochs": int(os.environ["EXPECTED_LR_MAX_EPOCHS"]),
    "decoder": False,
    "model_init_seed": 3072,
    "fit_seed": 0,
    "resume": None,
    "run_name": os.environ["RUN_NAME"],
}
print(actual)
if actual != expected:
    raise SystemExit(f"ERROR: training gate config mismatch: {actual} != {expected}")
PY

"$PY" -m mwm.data.verify "$CONFIG"
"$PY" -m mwm.training.stable_wm "$CONFIG" --set "data.training_source.path=$LOCAL_H5"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
