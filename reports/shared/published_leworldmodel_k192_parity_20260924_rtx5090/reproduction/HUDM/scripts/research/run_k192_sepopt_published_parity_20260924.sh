#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run_k192_sepopt_published_parity_20260924.sh requires a Slurm allocation." >&2
  exit 2
fi
if [[ -z "${MWM_PARITY_ENV:-}" || -z "${MWM_PARITY_STAGE:-}" ]]; then
  echo "ERROR: MWM_PARITY_ENV and MWM_PARITY_STAGE are required." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
REPORT_ROOT="reports/research/k192_sepopt_published_parity_20260924"
BASE_CFG="configs/research/k192_sepopt_published_parity_20260924/${MWM_PARITY_ENV}.yaml"

case "$MWM_PARITY_ENV" in
  ogb_cube|reacher|pusht|tworoom) ;;
  *) echo "ERROR: unknown MWM_PARITY_ENV=$MWM_PARITY_ENV" >&2; exit 2 ;;
esac

case "$MWM_PARITY_STAGE" in
  screen)
    EPISODES=50
    SEEDS=(0 1 2 42 100)
    STAGE_DIR="screen_n50"
    ;;
  final)
    EPISODES=100
    SEEDS=(2002 71623 82715 86604 91943)
    STAGE_DIR="final_n100"
    QUALIFICATION="$REPORT_ROOT/screen_${MWM_PARITY_ENV}_summary.json"
    if [[ ! -f "$QUALIFICATION" ]]; then
      echo "ERROR: missing screen qualification: $QUALIFICATION" >&2
      exit 3
    fi
    if ! "$PY" - "$QUALIFICATION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("qualified_for_n100") is True else 1)
PY
    then
      echo "[parity] ${MWM_PARITY_ENV} did not meet the predeclared screen threshold; n=100 skipped."
      exit 0
    fi
    ;;
  *) echo "ERROR: MWM_PARITY_STAGE must be screen or final." >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p "$REPORT_ROOT/generated_configs/$STAGE_DIR/$MWM_PARITY_ENV" \
  "$REPORT_ROOT/manifests/$STAGE_DIR" logs

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

for seed in "${SEEDS[@]}"; do
  output_dir="$REPORT_ROOT/$STAGE_DIR/${MWM_PARITY_ENV}/seed_${seed}"
  manifest_path="$REPORT_ROOT/manifests/$STAGE_DIR/${MWM_PARITY_ENV}_seed${seed}.json"
  generated_cfg="$REPORT_ROOT/generated_configs/$STAGE_DIR/${MWM_PARITY_ENV}/seed_${seed}.yaml"
  echo "[parity] stage=$MWM_PARITY_STAGE env=$MWM_PARITY_ENV seed=$seed episodes=$EPISODES"
  "$PY" - "$BASE_CFG" "$generated_cfg" "$seed" "$output_dir" "$manifest_path" "$EPISODES" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

src, dst, seed, output_dir, manifest_path, episodes = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.seed = int(seed)
cfg.output_dir = output_dir
cfg.title = f"{cfg.title} (evaluation seed {seed}, n={episodes})"
cfg.manifest = {
    "group": f"k192_sepopt_published_parity_{cfg.env_id}_seed{seed}_n{episodes}",
    "path": manifest_path,
}
for run in cfg.runs:
    run.eval = {
        "episodes": int(episodes),
        "num_envs": 50,
        "budget": 50,
    }
Path(dst).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
  "$PY" -m mwm.benchmark.verify "$generated_cfg" --static-only
  "$PY" -m mwm.benchmark.matrix "$generated_cfg" --resume
  "$PY" -m mwm.benchmark.verify "$generated_cfg"
done

"$PY" scripts/research/collect_k192_sepopt_published_parity_20260924.py \
  --stage "$MWM_PARITY_STAGE" --environment "$MWM_PARITY_ENV"
