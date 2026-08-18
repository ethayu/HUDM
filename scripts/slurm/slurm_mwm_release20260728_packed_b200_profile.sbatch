#!/usr/bin/env bash
#SBATCH --job-name=mwm_r20260728_pack2_profile
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=dineshj-lab
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: the packed-B200 profile is a single allocation, not an array." >&2
  exit 2
fi
if [[ -z "${MWM_PROFILE_EXPECTED_COMMIT:-}" ]]; then
  echo "ERROR: export MWM_PROFILE_EXPECTED_COMMIT as the pinned 40-character profiling commit." >&2
  exit 2
fi

ROOT="${MWM_PROFILE_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
  echo "ERROR: MWM_PROFILE_REPO_ROOT or SLURM_SUBMIT_DIR must name the frozen profiling worktree." >&2
  exit 2
fi
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
PROFILE_BASE="/ceph/projects/dineshj/lab/ethanyu/HUDM/profiling/packed_b200_release20260728"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
PROFILE_ROOT="$PROFILE_BASE/job_${SLURM_JOB_ID}_${RESTART_COUNT}"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "profile_root=$PROFILE_ROOT"
echo "repo_root=$ROOT expected_commit=$MWM_PROFILE_EXPECTED_COMMIT"
hostname

exec "$PY" scripts/research/packed_b200_release20260728_profile.py \
  --output-root "$PROFILE_ROOT" \
  --expected-commit "$MWM_PROFILE_EXPECTED_COMMIT" \
  --python "$PY" \
  --sample-interval-sec 1.0
