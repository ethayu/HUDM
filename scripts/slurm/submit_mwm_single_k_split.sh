#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_SCRIPT="scripts/slurm/slurm_mwm_train_single_k.sbatch"
ENVS=(pusht reacher ogb_cube tworoom)
LEVELS=(96 144)

cd "$ROOT"
mkdir -p logs

for env_slug in "${ENVS[@]}"; do
  for level in "${LEVELS[@]}"; do
    run_name="mwm_single_k${level}_${env_slug}"
    if [[ -e "checkpoints_mwm/$run_name" || -e "logs/mwm_training/$run_name" ]]; then
      echo "ERROR: refusing to reuse existing run path for $run_name." >&2
      exit 2
    fi
  done
done

for env_slug in "${ENVS[@]}"; do
  for level in "${LEVELS[@]}"; do
    run_name="mwm_single_k${level}_${env_slug}"
    job_id="$(sbatch --parsable --job-name="$run_name" "$JOB_SCRIPT" "$env_slug" "$level" "$run_name")"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$env_slug" "$level" "$run_name"
  done
done
