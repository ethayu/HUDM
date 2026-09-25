#!/usr/bin/env bash
#SBATCH --job-name=audit_reacher_parity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5632M
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
OUTPUT="$ROOT/reports/research/single_level_reacher_parity_20260806/completion_audit.json"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" scripts/research/audit_single_level_reacher_parity.py \
  --run-tests \
  --output "$OUTPUT"
test -s "$OUTPUT"
"$PY" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "pass":
    raise SystemExit(f"ERROR: final parity audit did not pass: {payload.get('counts')}")
print("SINGLE_LEVEL_REACHER_PARITY_AUDIT_PASSED")
PY
