#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${MWM_SWEEP_ENV:-}" || -z "${MWM_SWEEP_SEED:-}" ]]; then
  echo "ERROR: run inside the cross-environment evaluation array." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
ENV_NAME="$MWM_SWEEP_ENV"
SEED="$MWM_SWEEP_SEED"
EPISODES=500
SOURCE_ROOT="$ROOT/reports/research/release20260802_isolateddec_single_k192_eval_size"
REPORT_ROOT="$ROOT/reports/research/crossenv_scheduler_ablation_20260807"
SOURCE_CONFIG="$SOURCE_ROOT/generated_configs/${ENV_NAME}_seed${SEED}_n${EPISODES}.yaml"
GENERATED_CONFIG="$REPORT_ROOT/generated_configs/${ENV_NAME}_seed${SEED}_n${EPISODES}.yaml"
OUTPUT_DIR="$ROOT/rollouts/mwm_release20260728_dense_k192_eval_size_${ENV_NAME}_seed${SEED}_n${EPISODES}"
ROLE_H10="scheduler_h10_rerun_v2"
ROLE_H100="scheduler_h100_v2"
CHECKPOINT_H10="$ROOT/checkpoints_mwm/mwm_scheduler_ablation_${ENV_NAME}_k192_h10_seed3072_20260807"
CHECKPOINT_H100="$ROOT/checkpoints_mwm/mwm_scheduler_ablation_${ENV_NAME}_k192_h100_seed3072_20260807"
BASELINE_RESOLVED_CONFIG="$OUTPUT_DIR/003_single_k192_isolateddec/resolved_config.yaml"
EVAL_SNAPSHOT="$REPORT_ROOT/generated_configs/eval_snapshot_${ENV_NAME}_seed${SEED}_n${EPISODES}.yaml"

cd "$ROOT"
mkdir -p "$REPORT_ROOT/generated_configs" logs
for checkpoint in "$CHECKPOINT_H10" "$CHECKPOINT_H100"; do
  test -s "$checkpoint/config.json"
  test -s "$checkpoint/weights.pt"
  test -s "$checkpoint/world_metadata.json"
done
test -s "$SOURCE_CONFIG"
test -s "$OUTPUT_DIR/summary.csv"
test -s "$BASELINE_RESOLVED_CONFIG"

"$PY" - "$SOURCE_CONFIG" "$GENERATED_CONFIG" "$CHECKPOINT_H10" "$CHECKPOINT_H100" "$ENV_NAME" "$SEED" "$BASELINE_RESOLVED_CONFIG" "$EVAL_SNAPSHOT" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

(
    source,
    destination,
    checkpoint_h10,
    checkpoint_h100,
    env_name,
    seed,
    baseline_resolved_config,
    eval_snapshot,
) = sys.argv[1:]
cfg = OmegaConf.load(source)
if int(cfg.seed) != int(seed):
    raise SystemExit(f"source seed mismatch: {cfg.seed} != {seed}")
expected_roles = [
    "upstream_lewm_converted",
    "release_single_k192",
    "release_dense_fixed_finest",
    "release_single_k192_isolateddec",
]
actual_roles = [str(run.role) for run in cfg.runs]
if actual_roles != expected_roles:
    raise SystemExit(f"unexpected source roles: {actual_roles}")
if int(cfg.runs[0].eval.episodes) != 500:
    raise SystemExit("source config is not N=500")
snapshot = OmegaConf.load(baseline_resolved_config)
if int(snapshot.eval.episodes) != 500 or int(snapshot.eval.seed) != int(seed):
    raise SystemExit("archived baseline evaluator seed/episode mismatch")
if str(snapshot.eval.manifest_path) != str(cfg.manifest.path):
    raise SystemExit("archived baseline evaluator manifest mismatch")
if str(snapshot.eval.get("sampling", "")) != "stable_worldmodel":
    raise SystemExit("archived baseline evaluator sampling mismatch")
if snapshot.eval.get("goal_indexing", None) not in (None, "exact"):
    raise SystemExit("archived baseline evaluator does not use exact goal indexing")
Path(eval_snapshot).write_text(OmegaConf.to_yaml(snapshot), encoding="utf-8")
cfg.eval_config = eval_snapshot
base_eval = OmegaConf.to_container(cfg.runs[0].eval, resolve=True)
cfg.title = f"Cross-environment LR scheduler ablation, {env_name}, N=500, seed={seed}"
cfg.runs.append(
    {
        "name": "scheduler_h10_rerun_v2",
        "role": "scheduler_h10_rerun_v2",
        "checkpoint": checkpoint_h10,
        "matrix_index": 6,
        "eval": base_eval,
    }
)
cfg.runs.append(
    {
        "name": "scheduler_h100_v2",
        "role": "scheduler_h100_v2",
        "checkpoint": checkpoint_h100,
        "matrix_index": 7,
        "eval": base_eval,
    }
)
Path(destination).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY

"$PY" -m mwm.benchmark.verify "$GENERATED_CONFIG" --static-only --roles "$ROLE_H10" "$ROLE_H100"
if [[ "${MWM_GENERATE_ONLY:-0}" == "1" ]]; then
  echo "CROSSENV_SCHEDULER_STATIC_GATE_PASSED"
  exit 0
fi
"$PY" -m mwm.benchmark.matrix "$GENERATED_CONFIG" --roles "$ROLE_H10" "$ROLE_H100" --resume
"$PY" -m mwm.benchmark.matrix "$GENERATED_CONFIG" --finalize-only
"$PY" -m mwm.benchmark.verify "$GENERATED_CONFIG"
