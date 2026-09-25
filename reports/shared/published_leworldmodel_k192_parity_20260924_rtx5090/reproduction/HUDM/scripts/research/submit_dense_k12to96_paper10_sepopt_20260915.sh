#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_SCRIPT="scripts/research/slurm_train_dense_k12to96_paper10_sepopt_20260915.sbatch"
CONFIGS=(
  "configs/research/train_mwm_lewm_dense_tworoom_k12to96_paper10_sepopt_20260915.yaml"
  "configs/research/train_mwm_lewm_dense_ogb_cube_k12to96_paper10_sepopt_20260915.yaml"
  "configs/research/train_mwm_lewm_dense_reacher_k12to96_paper10_sepopt_20260915.yaml"
  "configs/research/train_mwm_lewm_dense_pusht_k12to96_paper10_sepopt_20260915.yaml"
)
JOB_NAMES=(
  "train_tr_k12to96"
  "train_ogb_k12to96"
  "train_reach_k12to96"
  "train_pusht_k12to96"
)
NICE_VALUES=(0 100 200 300)
TIME_LIMITS=(
  "18:00:00"
  "2-12:00:00"
  "1-16:00:00"
  "2-00:00:00"
)

cd "$ROOT"
mkdir -p logs

for config in "${CONFIGS[@]}"; do
  test -s "$config"
done
test -s "$JOB_SCRIPT"

job_ids=()
for index in "${!CONFIGS[@]}"; do
  # Independent submissions permit parallel execution. Submission order records
  # the requested user priority without imposing afterok dependencies.
  job_id="$(sbatch --parsable --nice="${NICE_VALUES[$index]}" --time="${TIME_LIMITS[$index]}" --job-name="${JOB_NAMES[$index]}" "$JOB_SCRIPT" "${CONFIGS[$index]}")"
  job_ids+=("$job_id")
  printf '%s\t%s\t%s\n' "$job_id" "${JOB_NAMES[$index]}" "${CONFIGS[$index]}"
done

job_csv="$(IFS=,; echo "${job_ids[*]}")"
printf 'jobs=%s\n' "$job_csv"
squeue -j "$job_csv" -o '%.18i %.30j %.8T %.10M %.9l %.30R'
