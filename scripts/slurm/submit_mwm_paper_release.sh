#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_SCRIPT="scripts/slurm/slurm_mwm_train_paper_release.sbatch"
ENVS=(pusht reacher ogb_cube tworoom)
VARIANTS=(
  "k96|96|single"
  "k120|120|single"
  "k144|144|single"
  "k168|168|single"
  "k192|192|single"
  "k96_120_144_168_192|96,120,144,168,192|multi"
)

cd "$ROOT"
mkdir -p logs

for env_slug in "${ENVS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r label level_csv profile <<< "$variant"
    run_name="mwm_paper10_${env_slug}_${label}_release20260728"
    if [[ -e "checkpoints_mwm/$run_name" || -e "logs/mwm_training/$run_name" ]]; then
      echo "ERROR: refusing to reuse existing run path for $run_name." >&2
      exit 2
    fi
    case "$profile" in
      single|multi) ;;
      *)
        echo "ERROR: unknown resource profile '$profile'." >&2
        exit 2
        ;;
    esac
  done
done

for env_slug in "${ENVS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r label level_csv profile <<< "$variant"
    run_name="mwm_paper10_${env_slug}_${label}_release20260728"
    if [[ "$profile" == "single" ]]; then
      partitions="b200-mig90,dgx-b200"
      cpus=14
      memory="112G"
    else
      partitions="dgx-b200"
      cpus=28
      memory="224G"
    fi
    job_id="$(sbatch --parsable \
      --partition="$partitions" \
      --gpus=1 \
      --cpus-per-task="$cpus" \
      --mem="$memory" \
      --job-name="$run_name" \
      "$JOB_SCRIPT" "$env_slug" "$level_csv" "$run_name")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$job_id" "$env_slug" "[$level_csv]" "$run_name" "$partitions" "$cpus" "$memory"
  done
done
