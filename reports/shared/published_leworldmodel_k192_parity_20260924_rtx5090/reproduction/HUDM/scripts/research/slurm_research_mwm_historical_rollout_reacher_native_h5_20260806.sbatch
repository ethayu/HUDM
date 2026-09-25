#!/usr/bin/env bash
#SBATCH --job-name=mwm_hist_rollout_h5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
LOCAL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mwm-historical-rollout-${SLURM_JOB_ID}.XXXXXX")"
LOCAL_H5="$LOCAL_ROOT/reacher.h5"
N_ITER="${MWM_N_ITER:-30}"
OUTPUT_TAG="${MWM_OUTPUT_TAG:-reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold010_historical_rollout_20260806}"
OUTPUT="$ROOT/rollouts/$OUTPUT_TAG/eval.json"
MANIFEST="$ROOT/rollouts/manifests/reacher_historical_goal24_native_h5_seed42_n50_iter30_threshold010_20260806.json"

cleanup() {
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

cd "$ROOT"
mkdir -p logs "$(dirname "$OUTPUT")"
if [[ -e "$OUTPUT" ]]; then
  echo "ERROR: refusing to overwrite $OUTPUT" >&2
  exit 3
fi

tar --use-compress-program=unzstd -xf "$ROOT/data/upstream/reacher.tar.zst" -C "$LOCAL_ROOT" reacher.h5
echo "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06  $LOCAL_H5" | sha256sum -c -

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

"$PY" -m mwm.eval.runner configs/eval/paper_reacher.yaml \
  --set "data.path=$LOCAL_H5" \
  --set "data.identity_path=data/upstream/reacher.h5" \
  --set data.format=hdf5 \
  --set eval.episodes=50 \
  --set eval.num_envs=50 \
  --set "eval.manifest_path=$MANIFEST" \
  --set "eval.output_path=$OUTPUT" \
  --set eval.save_video=false \
  --set "planner.n_iter=$N_ITER" \
  --set planner.batch_size=1 \
  --set planner.rollout_semantics=upstream_lewm_historical

test -s "$OUTPUT"
"$PY" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
diagnostics = payload["planning_diagnostics"]
expected_calls = int(diagnostics["cem_cost_calls"])
if diagnostics.get("rollout_semantics_counts") != {"upstream_lewm_historical": expected_calls}:
    raise SystemExit(f"Unexpected rollout-semantics provenance: {diagnostics}")
if diagnostics.get("policy_semantics_counts") != {"upstream_lewm_historical": expected_calls}:
    raise SystemExit(f"Unexpected policy-semantics provenance: {diagnostics}")
print(json.dumps(payload["swm_results"], indent=2))
PY
