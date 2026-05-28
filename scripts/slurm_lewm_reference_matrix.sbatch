#!/usr/bin/env bash
#SBATCH --job-name=lewm_ref_matrix
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${MWM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
CACHE_ROOT="${LEWM_OFFICIAL_CACHE:-/tmp/lewm_reference_matrix_${SLURM_JOB_ID}}"
UPSTREAM_CONFIG="${LEWM_PUSHT_CONFIG_JSON:-/vast/projects/dineshj/lab/ethanyu/cache/huggingface/hub/models--quentinll--lewm-pusht/blobs/2e534ee5c6286721c0ef967e24bddbfde482abf9}"
UPSTREAM_WEIGHTS="${LEWM_PUSHT_WEIGHTS_PT:-/vast/projects/dineshj/lab/ethanyu/tmp/lewm-hf-pusht/weights.pt}"
RESULTS_DIR="$ROOT/rollouts/lewm_official_reference"
OUTPUT="$RESULTS_DIR/reference_matrix_${SLURM_JOB_ID}.json"

mkdir -p "$ROOT/logs" "$RESULTS_DIR" "$CACHE_ROOT/checkpoints/local-lewm-pusht"

for required in \
  "$ROOT/scripts/lewm_reference_matrix.py" \
  "$ROOT/rollouts/mwm_paper_reference/000_pusht_seed42_upstream_lewm_converted/eval.json" \
  "$ROOT/data/upstream/pusht_expert_train.h5" \
  "$ROOT/data/upstream/pusht_expert_train.lance" \
  "$UPSTREAM_CONFIG" \
  "$UPSTREAM_WEIGHTS"; do
  if [ ! -e "$required" ]; then
    echo "Missing required reference-matrix input: $required" >&2
    exit 2
  fi
done

ln -sfn "$UPSTREAM_CONFIG" "$CACHE_ROOT/checkpoints/local-lewm-pusht/config.json"
ln -sfn "$UPSTREAM_WEIGHTS" "$CACHE_ROOT/checkpoints/local-lewm-pusht/weights.pt"

export MUJOCO_GL=egl
export STABLEWM_HOME="$CACHE_ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

hostname
nvidia-smi
"$PY" "$ROOT/scripts/lewm_reference_matrix.py" \
  --manifest-eval-json "$ROOT/rollouts/mwm_paper_reference/000_pusht_seed42_upstream_lewm_converted/eval.json" \
  --hdf5 "$ROOT/data/upstream/pusht_expert_train.h5" \
  --lance "$ROOT/data/upstream/pusht_expert_train.lance" \
  --policy local-lewm-pusht \
  --output "$OUTPUT"
