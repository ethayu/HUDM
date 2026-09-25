#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${MWM_SWEEP_SEED:-}" ]]; then
  echo "ERROR: run inside the Reacher N=500 Slurm array with MWM_SWEEP_SEED set." >&2
  exit 2
fi
for name in MWM_CANDIDATE_CHECKPOINT MWM_CANDIDATE_ROLE MWM_REPORT_TAG; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be set." >&2
    exit 2
  fi
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
SEED="$MWM_SWEEP_SEED"
EPISODES=500
SOURCE_ROOT="$ROOT/reports/research/release20260728_dense_k192_eval_size"
REPORT_ROOT="$ROOT/reports/research/$MWM_REPORT_TAG"
EVAL_CONFIG="${MWM_EVAL_CONFIG:-configs/eval/paper_reacher.yaml}"
EVAL_DATA_SOURCE="${MWM_EVAL_DATA_SOURCE:-native_hdf5}"
INCLUDE_UPSTREAM="${MWM_INCLUDE_UPSTREAM:-1}"
MANIFEST_ROOT="${MWM_MANIFEST_ROOT:-rollouts/manifests/single_level_upstream_lewm_historical_20260806}"
STATIC_ONLY="${MWM_STATIC_ONLY:-0}"
SOURCE_CONFIG="$SOURCE_ROOT/generated_configs/reacher_seed${SEED}_n${EPISODES}.yaml"
GENERATED_CONFIG="$REPORT_ROOT/generated_configs/reacher_seed${SEED}_n${EPISODES}.yaml"
CHECKPOINT="$MWM_CANDIDATE_CHECKPOINT"
OUTPUT_DIR="$ROOT/rollouts/${MWM_REPORT_TAG}_seed${SEED}_n${EPISODES}"
NATIVE_H5=""

case "$CHECKPOINT" in
  /*) ;;
  *) CHECKPOINT="$ROOT/$CHECKPOINT" ;;
esac

cd "$ROOT"
mkdir -p "$REPORT_ROOT/generated_configs" logs

test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/weights.pt"
test -s "$CHECKPOINT/world_metadata.json"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to mix with existing evaluation output: $OUTPUT_DIR" >&2
  exit 3
fi

if [[ "$INCLUDE_UPSTREAM" != "0" && "$INCLUDE_UPSTREAM" != "1" ]]; then
  echo "ERROR: MWM_INCLUDE_UPSTREAM must be 0 or 1, got $INCLUDE_UPSTREAM." >&2
  exit 2
fi
if [[ "$STATIC_ONLY" != "0" && "$STATIC_ONLY" != "1" ]]; then
  echo "ERROR: MWM_STATIC_ONLY must be 0 or 1, got $STATIC_ONLY." >&2
  exit 2
fi
if [[ "$EVAL_DATA_SOURCE" != "native_hdf5" && "$EVAL_DATA_SOURCE" != "lance" ]]; then
  echo "ERROR: MWM_EVAL_DATA_SOURCE must be native_hdf5 or lance, got $EVAL_DATA_SOURCE." >&2
  exit 2
fi
if [[ "$EVAL_DATA_SOURCE" == "native_hdf5" && "$STATIC_ONLY" == "0" ]]; then
  NATIVE_ROOT="/tmp/mwm-reacher-eval-${SLURM_JOB_ID}"
  NATIVE_H5="$NATIVE_ROOT/reacher.h5"
  mkdir -p "$NATIVE_ROOT"
  tar --use-compress-program=unzstd -xf "$ROOT/data/upstream/reacher.tar.zst" -C "$NATIVE_ROOT" reacher.h5
  echo "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06  $NATIVE_H5" | sha256sum -c -
fi

"$PY" - "$SOURCE_CONFIG" "$GENERATED_CONFIG" "$CHECKPOINT" "$OUTPUT_DIR" "$SEED" "$MWM_CANDIDATE_ROLE" "$EVAL_CONFIG" "$INCLUDE_UPSTREAM" "$MANIFEST_ROOT" "$EVAL_DATA_SOURCE" "$NATIVE_H5" <<'PY'
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
from mwm.data.manifest import manifest_sha256, write_manifest

(
    source,
    destination,
    checkpoint,
    output_dir,
    seed,
    role,
    eval_config,
    include_upstream,
    manifest_root,
    eval_data_source,
    native_h5,
) = sys.argv[1:]
cfg = OmegaConf.load(source)
if int(cfg.seed) != int(seed):
    raise SystemExit(f"source config seed mismatch: {cfg.seed} != {seed}")
if int(cfg.runs[0].eval.episodes) != 500:
    raise SystemExit("source config is not the fixed N=500 evaluation")
source_manifest = f"rollouts/manifests/release20260728_dense_k192_eval_size/reacher_seed{seed}_n500.json"
if str(cfg.manifest.path) != source_manifest:
    raise SystemExit(f"unexpected source manifest: {cfg.manifest.path}")
expected_manifest = f"{manifest_root}/reacher_seed{seed}_n500.json"
if not Path(expected_manifest).is_file():
    raise SystemExit(f"missing exact-upstream manifest: {expected_manifest}")
if eval_config not in {"configs/eval/paper_reacher.yaml", "configs/eval/public_lewm_reacher.yaml"}:
    raise SystemExit(f"unsupported parity eval config: {eval_config}")
if not Path(eval_config).is_file():
    raise SystemExit(f"missing parity eval config: {eval_config}")

upstream = OmegaConf.to_container(cfg.runs[0], resolve=True)
if upstream["name"] != "upstream" or upstream["checkpoint"] != "checkpoints_mwm/upstream_lewm_reacher":
    raise SystemExit(f"unexpected upstream run: {upstream}")
candidate = {
    "name": role,
    "role": role,
    "checkpoint": checkpoint,
    "eval": dict(upstream["eval"]),
}
cfg.output_dir = output_dir
cfg.title = f"Reacher checkpoint parity, N=500, seed={seed}, role={role}"
if eval_data_source == "native_hdf5" and native_h5:
    generated_root = Path(destination).parent
    runtime_eval = generated_root / f"eval_reacher_seed{seed}_n500_native_hdf5.yaml"
    runtime_manifest = generated_root / f"manifest_reacher_seed{seed}_n500_native_hdf5.json"
    runtime_cfg = OmegaConf.load(eval_config)
    runtime_cfg.data.path = native_h5
    runtime_cfg.data.format = "hdf5"
    runtime_cfg.data.identity_path = "data/upstream/reacher.h5"
    runtime_eval.write_text(OmegaConf.to_yaml(runtime_cfg), encoding="utf-8")

    manifest = json.loads(Path(expected_manifest).read_text(encoding="utf-8"))
    manifest["dataset_path"] = "data/upstream/reacher.h5"
    manifest["dataset_metadata"] = {
        "path": "data/upstream/reacher.h5",
        "format": "hdf5",
        "member_sha256": "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06",
        "pixels_key": "pixels",
        "action_key": "action",
        "column_names": ["pixels", "action", "qpos", "qvel", "observation"],
        "split_ratio": 1.0,
    }
    manifest["manifest_sha256"] = manifest_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_manifest(runtime_manifest, manifest)
    cfg.eval_config = str(runtime_eval)
    cfg.manifest.path = str(runtime_manifest)
    cfg.manifest.group = f"single_level_upstream_lewm_historical_native_hdf5_seed{seed}_n500"
else:
    cfg.eval_config = eval_config
    cfg.manifest.path = expected_manifest
    cfg.manifest.group = f"single_level_upstream_lewm_historical_seed{seed}_n500"
cfg.runs = [upstream, candidate] if int(include_upstream) else [candidate]
Path(destination).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY

"$PY" -m mwm.benchmark.verify "$GENERATED_CONFIG" --static-only
if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "REACHER_N500_STATIC_GATE_PASSED"
  exit 0
fi
"$PY" -m mwm.benchmark.matrix "$GENERATED_CONFIG"
"$PY" -m mwm.benchmark.matrix "$GENERATED_CONFIG" --finalize-only
"$PY" -m mwm.benchmark.verify "$GENERATED_CONFIG"

test -s "$OUTPUT_DIR/summary.csv"
