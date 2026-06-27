#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
if [[ -n "${MWM_POLL_JOBS:-}" ]]; then
  JOBS="$MWM_POLL_JOBS"
elif [[ "$#" -gt 0 ]]; then
  JOBS="$(IFS=,; echo "$*")"
else
  echo "Usage: MWM_POLL_JOBS=job[,job...] $0" >&2
  exit 2
fi
SAFE_JOBS="${JOBS//[^[:alnum:]]/_}"
LOG="${MWM_POLL_LOG:-$ROOT/logs/mwm_scheduled_poll_${SAFE_JOBS}.log}"

cd "$ROOT" || exit 1
mkdir -p logs

while true; do
  {
    echo "===== $(date -Is) ====="
    squeue -j "$JOBS" -o "%i %j %T %M %R" || true
    echo "--- sacct ---"
    sacct -j "$JOBS" --format=JobID,JobName,State,Elapsed,ExitCode -P || true
  } >> "$LOG" 2>&1

  active="$(squeue -j "$JOBS" -h -t PENDING,RUNNING,CONFIGURING,COMPLETING -o "%i" 2>/dev/null | wc -l || echo 0)"
  active="${active//[[:space:]]/}"
  if [[ "${active:-0}" == "0" ]]; then
    {
      echo "===== $(date -Is) verification ====="
      "$PY" -m mwm.benchmark.verify configs/benchmark/scheduled_pusht.yaml --static-only --roles upstream_lewm_converted mwm_scheduled
      static_status="$?"
      "$PY" -m mwm.benchmark.verify configs/benchmark/scheduled_pusht.yaml --roles upstream_lewm_converted mwm_scheduled
      full_status="$?"
      "$PY" -m mwm.benchmark.verify configs/benchmark/scheduled_tworoom.yaml --static-only --roles upstream_lewm_converted mwm_scheduled
      static_tworoom_status="$?"
      "$PY" -m mwm.benchmark.verify configs/benchmark/scheduled_tworoom.yaml --roles upstream_lewm_converted mwm_scheduled
      full_tworoom_status="$?"
      echo "STATIC_VERIFY_PUSHT_EXIT=$static_status"
      echo "FULL_VERIFY_PUSHT_EXIT=$full_status"
      echo "STATIC_VERIFY_TWOROOM_EXIT=$static_tworoom_status"
      echo "FULL_VERIFY_TWOROOM_EXIT=$full_tworoom_status"
    } >> "$LOG" 2>&1
    exit 0
  fi

  sleep 1800
done
