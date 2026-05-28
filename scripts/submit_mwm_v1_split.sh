#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUSHT_SINGLE_JOB="$(sbatch --parsable scripts/slurm_mwm_train_pusht_v1_single.sbatch)"
TWOROOM_SINGLE_JOB="$(sbatch --parsable scripts/slurm_mwm_train_tworoom_v1_single.sbatch)"
PUSHT_SCHEDULED_JOB="$(sbatch --parsable scripts/slurm_mwm_train_pusht_v1_scheduled.sbatch)"
TWOROOM_SCHEDULED_JOB="$(sbatch --parsable scripts/slurm_mwm_train_tworoom_v1_scheduled.sbatch)"
BENCHMARK_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${PUSHT_SINGLE_JOB}:${TWOROOM_SINGLE_JOB}:${PUSHT_SCHEDULED_JOB}:${TWOROOM_SCHEDULED_JOB}" \
    scripts/slurm_mwm_v1_benchmark.sbatch
)"

printf 'PushT single train job: %s\n' "$PUSHT_SINGLE_JOB"
printf 'TwoRoom single train job: %s\n' "$TWOROOM_SINGLE_JOB"
printf 'PushT scheduled train job: %s\n' "$PUSHT_SCHEDULED_JOB"
printf 'TwoRoom scheduled train job: %s\n' "$TWOROOM_SCHEDULED_JOB"
printf 'Benchmark job: %s\n' "$BENCHMARK_JOB"
