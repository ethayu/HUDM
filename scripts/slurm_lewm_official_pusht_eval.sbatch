#!/usr/bin/env bash
#SBATCH --job-name=lewm_official_pusht_eval
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MWM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
OFFICIAL_ROOT="${LEWM_OFFICIAL_ROOT:-/vast/projects/dineshj/lab/ethanyu/tmp/le-wm-official}"
CACHE_ROOT="${LEWM_OFFICIAL_CACHE:-/tmp/lewm_official_eval_cache_${SLURM_JOB_ID}}"
RESULTS_DIR="$ROOT/rollouts/lewm_official_reference"
UPSTREAM_CONFIG="${LEWM_PUSHT_CONFIG_JSON:-/vast/projects/dineshj/lab/ethanyu/cache/huggingface/hub/models--quentinll--lewm-pusht/blobs/2e534ee5c6286721c0ef967e24bddbfde482abf9}"
UPSTREAM_WEIGHTS="${LEWM_PUSHT_WEIGHTS_PT:-/vast/projects/dineshj/lab/ethanyu/tmp/lewm-hf-pusht/weights.pt}"

mkdir -p "$ROOT/logs" "$RESULTS_DIR" "$CACHE_ROOT/datasets" "$CACHE_ROOT/checkpoints/local-lewm-pusht"

for required in "$OFFICIAL_ROOT/eval.py" "$ROOT/data/upstream/pusht_expert_train.h5" "$UPSTREAM_CONFIG" "$UPSTREAM_WEIGHTS"; do
  if [ ! -e "$required" ]; then
    echo "Missing required official-eval input: $required" >&2
    exit 2
  fi
done

ln -sfn "$ROOT/data/upstream/pusht_expert_train.h5" "$CACHE_ROOT/datasets/pusht_expert_train.h5"
ln -sfn "$UPSTREAM_CONFIG" "$CACHE_ROOT/checkpoints/local-lewm-pusht/config.json"
ln -sfn "$UPSTREAM_WEIGHTS" "$CACHE_ROOT/checkpoints/local-lewm-pusht/weights.pt"

export MUJOCO_GL=egl
export STABLEWM_HOME="$CACHE_ROOT"

hostname
nvidia-smi
"$PY" - <<'PY'
import stable_worldmodel as swm
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("stable_worldmodel", swm.__file__)
PY

cd "$OFFICIAL_ROOT"
"$PY" eval.py \
  policy=local-lewm-pusht \
  cache_dir="$CACHE_ROOT" \
  output.filename="$RESULTS_DIR/pusht_eval_${SLURM_JOB_ID}.txt"
