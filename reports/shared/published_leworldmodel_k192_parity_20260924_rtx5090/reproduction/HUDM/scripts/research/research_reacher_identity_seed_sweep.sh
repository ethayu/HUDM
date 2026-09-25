#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_reacher_identity_seed_sweep.sh must run inside a Slurm allocation." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
DEFAULT_ARTIFACT_ROOT="/vast/projects/dineshj/lab/ethanyu/code/H""UDM"
EPISODES="${MWM_REACHER_SWEEP_EPISODES:-200}"
NUM_ENVS="${MWM_REACHER_SWEEP_NUM_ENVS:-50}"
SEEDS=(${MWM_REACHER_SWEEP_SEEDS:-0 1 2 42 100})
BASE_CFG="${MWM_REACHER_SWEEP_BASE_CFG:-configs/research/reacher_identity_delta/reacher_benchmark_seed42.yaml}"
OUT_ROOT="${MWM_REACHER_SWEEP_OUT:-reports/research/reacher_identity_delta/seed_sweep}"

cd "$ROOT"
mkdir -p "$OUT_ROOT/generated_configs" logs

export MWM_PYTHON="$PY"
export MWM_ARTIFACT_ROOT="${MWM_ARTIFACT_ROOT:-$DEFAULT_ARTIFACT_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

for seed in "${SEEDS[@]}"; do
  out="${OUT_ROOT}/reacher_seed${seed}"
  generated_cfg="${OUT_ROOT}/generated_configs/reacher_seed${seed}.yaml"
  mkdir -p "$(dirname "$generated_cfg")"
  echo "[reacher-identity-sweep] seed=${seed} episodes=${EPISODES} num_envs=${NUM_ENVS} out=${out}"
  "$PY" - "$BASE_CFG" "$generated_cfg" "$seed" "$out" "$EPISODES" "$NUM_ENVS" "$OUT_ROOT" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

src, dst, seed, output_dir, episodes, num_envs, out_root = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.seed = int(seed)
cfg.output_dir = output_dir
cfg.manifest.group = "reacher_identity_delta"
cfg.manifest.dir = str(Path(out_root) / "manifests")
for run in cfg.runs:
    run.eval.episodes = int(episodes)
    run.eval.num_envs = int(num_envs)
Path(dst).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
  "$PY" -m mwm.benchmark.matrix "$generated_cfg"
done
