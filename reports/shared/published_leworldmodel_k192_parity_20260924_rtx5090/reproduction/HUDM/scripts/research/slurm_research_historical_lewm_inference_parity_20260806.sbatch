#!/usr/bin/env bash
#SBATCH --job-name=hist_lewm_infer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
SOURCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mwm-historical-lewm-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$SOURCE_ROOT"' EXIT

cd "$ROOT"
mkdir -p logs
git clone --quiet --no-checkout https://github.com/lucas-maes/le-wm.git "$SOURCE_ROOT/le-wm"
git -C "$SOURCE_ROOT/le-wm" checkout --quiet 83f97d72ad067855bc89a1b74b4aff11d4dfdf0c

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$PY" scripts/research/research_historical_lewm_inference_parity.py \
  --historical-lewm-source "$SOURCE_ROOT/le-wm"
