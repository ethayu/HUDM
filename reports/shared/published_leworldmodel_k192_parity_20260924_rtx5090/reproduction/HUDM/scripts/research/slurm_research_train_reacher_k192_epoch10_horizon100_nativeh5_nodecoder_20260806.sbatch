#!/usr/bin/env bash
#SBATCH --job-name=train_reacher_h100
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

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export MWM_TRAIN_CONFIG="$ROOT/configs/research/train_mwm_lewm_reacher_k192_epoch10_horizon100_nativeh5_nodecoder_init3072_fit0_20260806.yaml"
export MWM_TRAIN_RUN_NAME="mwm_reacher_k192_epoch10_horizon100_nativeh5_nodecoder_init3072_fit0_20260806"
export MWM_EXPECTED_MAX_EPOCHS=10
export MWM_EXPECTED_LR_MAX_EPOCHS=100
export MWM_EXPECTED_CHECKPOINT_STEPS=63980

exec bash "$ROOT/scripts/research/slurm_research_train_reacher_k192_paper10_nativeh5_nodecoder_20260806.sbatch"
