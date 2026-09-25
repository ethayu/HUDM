#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${MWM_SWEEP_SEED:-}" ]]; then
  echo "ERROR: run inside the Reacher N=500 Slurm array with MWM_SWEEP_SEED set." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
SEED="$MWM_SWEEP_SEED"
EPISODES=500
SOURCE_ROOT="$ROOT/reports/research/release20260728_dense_k192_eval_size"
REPORT_ROOT="$ROOT/reports/research/reacher_k192_spt_sigreg009_n500_20260804"
SOURCE_CONFIG="$SOURCE_ROOT/generated_configs/reacher_seed${SEED}_n${EPISODES}.yaml"
GENERATED_CONFIG="$REPORT_ROOT/generated_configs/reacher_seed${SEED}_n${EPISODES}.yaml"
CHECKPOINT="$ROOT/checkpoints_mwm/mwm_spt_isolateddec_reacher_k192_sigreg009_seed3072_20260804"
OUTPUT_DIR="$ROOT/rollouts/mwm_reacher_k192_spt_sigreg009_seed${SEED}_n${EPISODES}_20260804"

cd "$ROOT"
mkdir -p "$REPORT_ROOT/generated_configs" logs

test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/weights.pt"
test -s "$CHECKPOINT/world_metadata.json"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to mix with existing evaluation output: $OUTPUT_DIR" >&2
  exit 3
fi

"$PY" - "$SOURCE_CONFIG" "$GENERATED_CONFIG" "$CHECKPOINT" "$OUTPUT_DIR" "$SEED" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

source, destination, checkpoint, output_dir, seed = sys.argv[1:]
cfg = OmegaConf.load(source)
if int(cfg.seed) != int(seed):
    raise SystemExit(f"source config seed mismatch: {cfg.seed} != {seed}")
if int(cfg.runs[0].eval.episodes) != 500:
    raise SystemExit("source config is not the fixed N=500 evaluation")
if str(cfg.manifest.path) != f"rollouts/manifests/release20260728_dense_k192_eval_size/reacher_seed{seed}_n500.json":
    raise SystemExit(f"unexpected fixed manifest: {cfg.manifest.path}")

upstream = OmegaConf.to_container(cfg.runs[0], resolve=True)
if upstream["name"] != "upstream" or upstream["checkpoint"] != "checkpoints_mwm/upstream_lewm_reacher":
    raise SystemExit(f"unexpected upstream run: {upstream}")
candidate = {
    "name": "spt_isolateddec_sigreg009",
    "role": "spt_isolateddec_sigreg009",
    "checkpoint": checkpoint,
    "eval": dict(upstream["eval"]),
}
cfg.output_dir = output_dir
cfg.title = f"Reacher K=192 SPT isolated decoder SIGReg 0.09 vs upstream, N=500, seed={seed}"
cfg.runs = [upstream, candidate]
Path(destination).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY

"$PY" -m mwm.benchmark.verify "$GENERATED_CONFIG" --static-only
"$PY" -m mwm.benchmark.matrix "$GENERATED_CONFIG"
"$PY" -m mwm.benchmark.matrix "$GENERATED_CONFIG" --finalize-only
"$PY" -m mwm.benchmark.verify "$GENERATED_CONFIG"

test -s "$OUTPUT_DIR/summary.csv"

