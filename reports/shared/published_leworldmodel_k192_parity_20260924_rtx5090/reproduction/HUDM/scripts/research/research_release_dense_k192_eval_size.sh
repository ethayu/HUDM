#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research/research_release_dense_k192_eval_size.sh must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ -z "${MWM_SWEEP_ENV:-}" || -z "${MWM_SWEEP_SEED:-}" ]]; then
  echo "ERROR: MWM_SWEEP_ENV and MWM_SWEEP_SEED are required." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
EPISODES="${MWM_SWEEP_EPISODES:-500}"
NUM_ENVS="${MWM_SWEEP_NUM_ENVS:-50}"
REPORT_ROOT="reports/research/release20260728_dense_k192_eval_size"
MANIFEST_ROOT="rollouts/manifests/release20260728_dense_k192_eval_size"

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

output_dir="rollouts/mwm_release20260728_dense_k192_eval_size_${MWM_SWEEP_ENV}_seed${MWM_SWEEP_SEED}_n${EPISODES}"
manifest_path="${MANIFEST_ROOT}/${MWM_SWEEP_ENV}_seed${MWM_SWEEP_SEED}_n${EPISODES}.json"
generated_cfg="${REPORT_ROOT}/generated_configs/${MWM_SWEEP_ENV}_seed${MWM_SWEEP_SEED}_n${EPISODES}.yaml"

echo "[dense-k192-eval-size] env=${MWM_SWEEP_ENV} seed=${MWM_SWEEP_SEED} episodes=${EPISODES} output=${output_dir}"
"$PY" - "$base_cfg" "$generated_cfg" "$MWM_SWEEP_SEED" "$output_dir" "$manifest_path" "$EPISODES" "$NUM_ENVS" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

src, dst, seed, output_dir, manifest_path, episodes, num_envs = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.seed = int(seed)
cfg.output_dir = output_dir
cfg.title = f"Release 2026-07-28 Upstream vs Dense K=192, N={episodes}, seed={seed}"
cfg.manifest = {
    "group": f"release20260728_dense_k192_{cfg.env_id}_seed{seed}_n{episodes}",
    "path": manifest_path,
}
for run in cfg.runs:
    run.eval = OmegaConf.merge(run.get("eval", {}), {"episodes": int(episodes), "num_envs": int(num_envs)})
Path(dst).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY

roles=(upstream_lewm_converted release_dense_fixed_finest)
"$PY" -m mwm.benchmark.verify "$generated_cfg" --static-only --roles "${roles[@]}"
"$PY" -m mwm.benchmark.matrix "$generated_cfg" --roles "${roles[@]}" --resume
"$PY" -m mwm.benchmark.verify "$generated_cfg" --roles "${roles[@]}"
