#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs rollouts checkpoints_mwm

PUSHT_JOB="$(sbatch --parsable scripts/slurm/slurm_mwm_train_pusht_dense.sbatch)"
TWOROOM_JOB="$(sbatch --parsable scripts/slurm/slurm_mwm_train_tworoom_dense.sbatch)"
BENCHMARK_JOB="$(sbatch --parsable --dependency="afterok:${PUSHT_JOB}:${TWOROOM_JOB}" scripts/slurm/slurm_mwm_dense_comparison.sbatch)"

printf 'PushT dense train job: %s\n' "$PUSHT_JOB"
printf 'TwoRoom dense train job: %s\n' "$TWOROOM_JOB"
printf 'Dense comparison benchmark job: %s\n' "$BENCHMARK_JOB"
printf 'monitor_active=squeue -j %s,%s,%s -o '"'"'%%.18i %%.30j %%.8T %%.10M %%.9l %%.20R'"'"'\n' "$PUSHT_JOB" "$TWOROOM_JOB" "$BENCHMARK_JOB"
printf 'monitor_complete=sacct -j %s,%s,%s --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,AllocCPUS,ReqMem\n' "$PUSHT_JOB" "$TWOROOM_JOB" "$BENCHMARK_JOB"
squeue -j "${PUSHT_JOB},${TWOROOM_JOB},${BENCHMARK_JOB}" -o '%.18i %.30j %.8T %.10M %.9l %.20R'
