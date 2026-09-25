#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for env_slug in pusht reacher ogb_cube tworoom; do
  config="configs/benchmark/release20260728_identity_parity_${env_slug}.yaml"
  job_id="$(sbatch --parsable --export=ALL,MWM_BENCHMARK_CONFIG="$config" \
    scripts/slurm/slurm_mwm_release_identity_parity.sbatch)"
  printf '%s job: %s (%s)\n' "$env_slug" "$job_id" "$config"
done
