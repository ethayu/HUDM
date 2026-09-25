#!/usr/bin/env bash
#SBATCH --job-name=train_reacher_k192_spt009
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
CONFIG="$ROOT/configs/research/train_mwm_lewm_reacher_k192_spt_isolateddec_sigreg009_seed3072_20260804.yaml"
PARITY_PASS="$ROOT/reports/research/reacher_k192_spt_parity_seed3072_sigreg009_20260804_v2/PASS"
IDENTITY_PASS="$ROOT/reports/research/reacher_k192_june_identity_c09bff78_seed3072_20260804/PASS"
RUN_NAME="mwm_spt_isolateddec_reacher_k192_sigreg009_seed3072_20260804"
CHECKPOINT_ROOT="$ROOT/checkpoints_mwm/$RUN_NAME"
TRAINER_ROOT="$ROOT/logs/mwm_training/$RUN_NAME"

cd "$ROOT"
mkdir -p logs

test -f "$PARITY_PASS"
test -f "$IDENTITY_PASS"
if [[ -e "$CHECKPOINT_ROOT" || -e "$TRAINER_ROOT" ]]; then
  echo "ERROR: refusing to overwrite an existing fresh-run path." >&2
  echo "checkpoint: $CHECKPOINT_ROOT" >&2
  echo "trainer: $TRAINER_ROOT" >&2
  exit 3
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

hostname
nvidia-smi
"$PY" - <<'PY'
from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS

path = "configs/research/train_mwm_lewm_reacher_k192_spt_isolateddec_sigreg009_seed3072_20260804.yaml"
cfg = load_config(DEFAULTS, path)
actual = {
    "seed": int(cfg.seed),
    "K": list(cfg.model.K),
    "sigreg": float(cfg.loss.sigreg_weight),
    "epochs": int(cfg.schedule.max_epochs),
    "decoder": bool(cfg.decoder_training.enabled),
    "decoder_mode": str(cfg.decoder_training.mode),
    "resume": cfg.train.resume_checkpoint,
}
expected = {
    "seed": 3072,
    "K": [192],
    "sigreg": 0.09,
    "epochs": 10,
    "decoder": True,
    "decoder_mode": "separate_optimizer",
    "resume": None,
}
print(actual)
if actual != expected:
    raise SystemExit(f"ERROR: training gate config mismatch: {actual} != {expected}")
PY

"$PY" -m mwm.data.verify "$CONFIG"
"$PY" -m mwm.training.stable_wm "$CONFIG"

test -s "$CHECKPOINT_ROOT/config.json"
test -s "$CHECKPOINT_ROOT/weights.pt"
test -s "$CHECKPOINT_ROOT/world_metadata.json"
