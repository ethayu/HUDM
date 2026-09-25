#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for env_slug in pusht reacher ogb_cube tworoom; do
  job_id="$(sbatch --parsable --export=ALL,MWM_SWEEP_ENV="$env_slug" \
    scripts/research/slurm_research_release_identity_seed_sweep.sbatch)"
  printf '%s seed-sweep job: %s\n' "$env_slug" "$job_id"
done
