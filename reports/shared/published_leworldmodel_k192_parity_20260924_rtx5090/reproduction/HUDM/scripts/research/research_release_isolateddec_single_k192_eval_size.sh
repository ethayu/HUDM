#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_release_isolateddec_single_k192_eval_size.sh must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ -z "${MWM_SWEEP_ENV:-}" || -z "${MWM_SWEEP_SEED:-}" ]]; then
  echo "ERROR: MWM_SWEEP_ENV and MWM_SWEEP_SEED are required." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
EPISODES="${MWM_SWEEP_EPISODES:-500}"
SOURCE_REPORT_ROOT="reports/research/release20260728_dense_k192_eval_size"
REPORT_ROOT="reports/research/release20260802_isolateddec_single_k192_eval_size"
source_cfg="${SOURCE_REPORT_ROOT}/generated_configs/${MWM_SWEEP_ENV}_seed${MWM_SWEEP_SEED}_n${EPISODES}.yaml"
generated_cfg="${REPORT_ROOT}/generated_configs/${MWM_SWEEP_ENV}_seed${MWM_SWEEP_SEED}_n${EPISODES}.yaml"
checkpoint="checkpoints_mwm/mwm_paper10_isolateddec_${MWM_SWEEP_ENV}_k192_seed3072_20260802"

cd "$ROOT"
mkdir -p "$REPORT_ROOT/generated_configs" logs

if [[ ! -f "$source_cfg" ]]; then
  echo "ERROR: missing targeted-study config: $source_cfg" >&2
  exit 2
fi
if [[ ! -f "$checkpoint/weights.pt" || ! -f "$checkpoint/config.json" || ! -f "$checkpoint/world_metadata.json" ]]; then
  echo "ERROR: incomplete isolated-decoder checkpoint: $checkpoint" >&2
  exit 2
fi

echo "[isolateddec-single-k192-eval] env=${MWM_SWEEP_ENV} seed=${MWM_SWEEP_SEED} episodes=${EPISODES}"
"$PY" - "$source_cfg" "$generated_cfg" "$checkpoint" "$EPISODES" "$MWM_SWEEP_SEED" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

src, dst, checkpoint, episodes, seed = sys.argv[1:]
cfg = OmegaConf.load(src)

if int(cfg.seed) != int(seed):
    raise SystemExit(f"source config seed mismatch: expected {seed}, got {cfg.seed}")

expected_names = ["upstream", "single_k192", "dense_fixed_finest"]
actual_names = [str(run.name) for run in cfg.runs]
if actual_names != expected_names:
    raise SystemExit(f"unexpected source run order: {actual_names}")

cfg.title = f"Release 2026-08-02 Corrected Single K=192, N={episodes}, seed={seed}"
cfg.runs.append(
    {
        "name": "single_k192_isolateddec",
        "role": "release_single_k192_isolateddec",
        "checkpoint": checkpoint,
        "matrix_index": 3,
        "eval": OmegaConf.to_container(cfg.runs[0].eval, resolve=True),
    }
)
Path(dst).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY

role=release_single_k192_isolateddec
"$PY" -m mwm.benchmark.verify "$generated_cfg" --static-only --roles "$role"
if [[ "${MWM_GENERATE_ONLY:-0}" == "1" ]]; then
  echo "[isolateddec-single-k192-eval] generation/static verification complete"
  exit 0
fi
"$PY" -m mwm.benchmark.matrix "$generated_cfg" --roles "$role" --resume
"$PY" -m mwm.benchmark.matrix "$generated_cfg" --finalize-only
"$PY" -m mwm.benchmark.verify "$generated_cfg"
