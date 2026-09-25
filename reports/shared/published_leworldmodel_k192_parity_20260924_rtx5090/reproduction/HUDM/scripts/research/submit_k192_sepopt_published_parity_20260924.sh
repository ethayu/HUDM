#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs reports/research/k192_sepopt_published_parity_20260924

declare -a screen_ids=()
declare -a final_ids=()
for environment in ogb_cube reacher pusht tworoom; do
  screen_id="$(sbatch --parsable --job-name="k192s_${environment}" --time=03:00:00 \
    --export=ALL,MWM_PARITY_ENV="$environment",MWM_PARITY_STAGE=screen \
    scripts/research/slurm_k192_sepopt_published_parity_20260924.sbatch)"
  final_id="$(sbatch --parsable --job-name="k192f_${environment}" --time=05:00:00 \
    --dependency="afterok:${screen_id}" \
    --export=ALL,MWM_PARITY_ENV="$environment",MWM_PARITY_STAGE=final \
    scripts/research/slurm_k192_sepopt_published_parity_20260924.sbatch)"
  screen_ids+=("$screen_id")
  final_ids+=("$final_id")
  printf '%s screen=%s final=%s\n' "$environment" "$screen_id" "$final_id"
done

dependency="afterany:$(IFS=:; echo "${final_ids[*]}")"
collector_id="$(sbatch --parsable --dependency="$dependency" \
  scripts/research/slurm_collect_k192_sepopt_published_parity_20260924.sbatch)"
printf 'collector=%s dependency=%s\n' "$collector_id" "$dependency"

python - "${screen_ids[@]}" -- "${final_ids[@]}" -- "$collector_id" <<'PY'
import json
import sys
from pathlib import Path

first = sys.argv.index("--")
second = sys.argv.index("--", first + 1)
screen = sys.argv[1:first]
final = sys.argv[first + 1:second]
collector = sys.argv[second + 1]
environments = ["ogb_cube", "reacher", "pusht", "tworoom"]
payload = {
    "screen_jobs": dict(zip(environments, screen, strict=True)),
    "final_n100_jobs": dict(zip(environments, final, strict=True)),
    "collector_job": collector,
    "screen_seeds": [0, 1, 2, 42, 100],
    "final_random_seed_generation": "random.Random(20260924).sample(range(1, 100001), 5)",
    "final_random_seeds": [2002, 71623, 82715, 86604, 91943],
    "screen_episodes_per_seed_per_model": 50,
    "final_episodes_per_seed_per_model": 100,
    "screen_qualification_mean_delta_pp": -5.0,
}
Path("reports/research/k192_sepopt_published_parity_20260924/jobs.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
