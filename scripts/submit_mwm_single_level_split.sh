#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUSHT_JOB="$(sbatch --parsable scripts/slurm_mwm_train_pusht_single.sbatch)"
TWOROOM_JOB="$(sbatch --parsable scripts/slurm_mwm_train_tworoom_single.sbatch)"
BENCHMARK_JOB="$(sbatch --parsable --dependency="afterok:${PUSHT_JOB}:${TWOROOM_JOB}" scripts/slurm_mwm_single_level_benchmark.sbatch)"

printf 'PushT train job: %s\n' "$PUSHT_JOB"
printf 'TwoRoom train job: %s\n' "$TWOROOM_JOB"
printf 'Benchmark job: %s\n' "$BENCHMARK_JOB"
