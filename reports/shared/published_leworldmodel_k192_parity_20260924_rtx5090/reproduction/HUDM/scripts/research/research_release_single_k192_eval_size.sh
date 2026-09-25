#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_release_single_k192_eval_size.sh must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ -z "${MWM_SWEEP_ENV:-}" || -z "${MWM_SWEEP_SEED:-}" ]]; then
  echo "ERROR: MWM_SWEEP_ENV and MWM_SWEEP_SEED are required." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
EPISODES="${MWM_SWEEP_EPISODES:-500}"
REPORT_ROOT="reports/research/release20260728_dense_k192_eval_size"
generated_cfg="${REPORT_ROOT}/generated_configs/${MWM_SWEEP_ENV}_seed${MWM_SWEEP_SEED}_n${EPISODES}.yaml"

cd "$ROOT"

if [[ ! -f "$generated_cfg" ]]; then
  echo "ERROR: missing targeted-study config: $generated_cfg" >&2
  exit 2
fi

role=release_single_k192
echo "[single-k192-eval-size] env=${MWM_SWEEP_ENV} seed=${MWM_SWEEP_SEED} episodes=${EPISODES}"
"$PY" -m mwm.benchmark.verify "$generated_cfg" --static-only --roles "$role"
"$PY" -m mwm.benchmark.matrix "$generated_cfg" --roles "$role" --resume
"$PY" -m mwm.benchmark.matrix "$generated_cfg" --finalize-only
"$PY" -m mwm.benchmark.verify "$generated_cfg"
