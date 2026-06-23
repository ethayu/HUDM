#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PUSHT_JOB="$(sbatch --parsable scripts/slurm/slurm_mwm_train_pusht_identity.sbatch)"
REACHER_JOB="$(sbatch --parsable scripts/slurm/slurm_mwm_train_reacher_identity.sbatch)"
TWOROOM_JOB="$(sbatch --parsable scripts/slurm/slurm_mwm_train_tworoom_identity.sbatch)"
BENCHMARK_JOB="$(sbatch --parsable --dependency="afterok:${PUSHT_JOB}:${REACHER_JOB}:${TWOROOM_JOB}" scripts/slurm/slurm_mwm_identity_parity.sbatch)"

printf 'PushT train job: %s\n' "$PUSHT_JOB"
printf 'Reacher train job: %s\n' "$REACHER_JOB"
printf 'TwoRoom train job: %s\n' "$TWOROOM_JOB"
printf 'Benchmark job: %s\n' "$BENCHMARK_JOB"
