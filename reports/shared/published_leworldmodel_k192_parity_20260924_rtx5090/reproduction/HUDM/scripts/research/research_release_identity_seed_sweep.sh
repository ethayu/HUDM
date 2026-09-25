#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_release_identity_seed_sweep.sh must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ -z "${MWM_SWEEP_ENV:-}" ]]; then
  echo "ERROR: MWM_SWEEP_ENV must be one of pusht, reacher, ogb_cube, or tworoom." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
SEEDS=(${MWM_SWEEP_SEEDS:-0 1 2 100})
REPORT_ROOT="reports/research/release20260728_identity_seed_sweep"
MANIFEST_ROOT="rollouts/manifests/release20260728_identity_seed_sweep"

cd "$ROOT"
mkdir -p "$REPORT_ROOT/generated_configs" "$MANIFEST_ROOT" logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

case "$MWM_SWEEP_ENV" in
  pusht|reacher|ogb_cube|tworoom)
    base_cfg="configs/benchmark/release20260728_identity_parity_${MWM_SWEEP_ENV}.yaml"
    ;;
  *)
    echo "ERROR: unknown MWM_SWEEP_ENV=$MWM_SWEEP_ENV" >&2
    exit 2
    ;;
esac

for seed in "${SEEDS[@]}"; do
  output_dir="rollouts/mwm_release20260728_identity_parity_${MWM_SWEEP_ENV}_seed${seed}"
  manifest_path="${MANIFEST_ROOT}/${MWM_SWEEP_ENV}_seed${seed}.json"
  generated_cfg="${REPORT_ROOT}/generated_configs/${MWM_SWEEP_ENV}_seed${seed}.yaml"
  echo "[release-identity-seed-sweep] env=${MWM_SWEEP_ENV} seed=${seed} output=${output_dir}"
  "$PY" - "$base_cfg" "$generated_cfg" "$seed" "$output_dir" "$manifest_path" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

src, dst, seed, output_dir, manifest_path = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.seed = int(seed)
cfg.output_dir = output_dir
cfg.title = f"{cfg.title} (seed {seed})"
cfg.manifest = {
    "group": f"release20260728_identity_parity_{cfg.env_id}_seed{seed}",
    "path": manifest_path,
}
Path(dst).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
  "$PY" -m mwm.benchmark.verify "$generated_cfg" --static-only
  "$PY" -m mwm.benchmark.matrix "$generated_cfg" --resume
  "$PY" -m mwm.benchmark.verify "$generated_cfg"
done
