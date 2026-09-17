#!/usr/bin/env bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this script must be submitted as a sparse Slurm array." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
TARGETS="${MWM_CONFIRMATION_TARGETS:-configs/research/release20260728_frontier_confirmation_targets.tsv}"
OUTPUT_ROOT="${MWM_CONFIRMATION_OUTPUT_ROOT:-/ceph/projects/dineshj/lab/ethanyu/HUDM/reports/research/release20260728_frontier_confirmation_250}"

cd "$ROOT"
mkdir -p logs

target_row="$(awk -F $'\t' -v task_id="$SLURM_ARRAY_TASK_ID" \
  'NR > 1 && $1 == task_id { print; exit }' "$TARGETS")"
if [[ -z "$target_row" ]]; then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is absent from ${TARGETS}." >&2
  exit 2
fi
IFS=$'\t' read -r task_id stage matrix_slug goal_offset config matrix_index policy_id anchor_id <<<"$target_row"

if [[ "$task_id" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "ERROR: target row/task mismatch: ${task_id} != ${SLURM_ARRAY_TASK_ID}." >&2
  exit 2
fi
if [[ "$stage" == "A" && "$SLURM_JOB_NAME" == *stage_b* ]]; then
  echo "ERROR: stage-A target submitted under a stage-B job name." >&2
  exit 2
fi
if [[ "$stage" == "B" && "$SLURM_JOB_NAME" != *stage_b* ]]; then
  echo "ERROR: stage-B target requires a job name containing stage_b." >&2
  exit 2
fi

output_dir="${OUTPUT_ROOT}/${matrix_slug}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "confirmation_task=${task_id} stage=${stage} matrix=${matrix_slug} goal=${goal_offset} policy=${policy_id} anchor=${anchor_id} matrix_index=${matrix_index}"
echo "config=${config} output_dir=${output_dir}"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -m mwm.benchmark.matrix "$config" \
  --set "output_dir=${output_dir}" \
  --num-shards 3120 \
  --shard-index "$matrix_index" \
  --resume
