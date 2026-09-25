#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH="${SBATCH:-/cm/local/apps/slurm/current/bin/sbatch}"
cd "$ROOT"
mkdir -p logs reports/research/crossenv_scheduler_ablation_20260807

train_result="$($SBATCH --parsable scripts/research/slurm_research_train_crossenv_scheduler_ablation_20260807.sbatch)"
train_job="${train_result%%;*}"
eval_result="$($SBATCH --parsable --dependency="afterok:${train_job}" scripts/research/slurm_research_eval_crossenv_scheduler_ablation_20260807.sbatch)"
eval_job="${eval_result%%;*}"
collect_result="$($SBATCH --parsable --dependency="afterok:${eval_job}" scripts/research/slurm_collect_crossenv_scheduler_ablation_20260807.sbatch)"
collect_job="${collect_result%%;*}"

python_bin="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
"$python_bin" - "$train_job" "$eval_job" "$collect_job" <<'PY'
import json
from pathlib import Path
import sys

train_job, eval_job, collect_job = sys.argv[1:]
payload = {
    "status": "submitted",
    "training_array": train_job,
    "evaluation_array": eval_job,
    "collector": collect_job,
}
path = Path("reports/research/crossenv_scheduler_ablation_20260807/submission.json")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
