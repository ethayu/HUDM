#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/research_identity_seed_sweep.sh must run inside a Slurm allocation." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
EPISODES="${MWM_SWEEP_EPISODES:-50}"
SEEDS=(${MWM_SWEEP_SEEDS:-0 1 2 42 100})
ENVS=(${MWM_SWEEP_ENVS:-pusht tworoom})

cd "$ROOT"
mkdir -p reports/research/identity_delta/seed_sweep logs

export MWM_PYTHON="$PY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

run_env() {
  local env_name="$1"
  local cfg="$2"
  for seed in "${SEEDS[@]}"; do
    local out="reports/research/identity_delta/seed_sweep/${env_name}_seed${seed}"
    local generated_cfg="reports/research/identity_delta/seed_sweep/generated_configs/${env_name}_seed${seed}.yaml"
    mkdir -p "$(dirname "$generated_cfg")"
    echo "[identity-sweep] env=${env_name} seed=${seed} episodes=${EPISODES} out=${out}"
    "$PY" - "$cfg" "$generated_cfg" "$seed" "$out" "$EPISODES" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

src, dst, seed, output_dir, episodes = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.seed = int(seed)
cfg.output_dir = output_dir
for run in cfg.runs:
    run.eval.episodes = int(episodes)
    run.eval.num_envs = int(episodes)
Path(dst).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
    "$PY" benchmark_mwm.py "$generated_cfg"
  done
}

for env_name in "${ENVS[@]}"; do
  case "$env_name" in
    pusht)
      run_env pusht configs/research/identity_delta_pusht_benchmark.yaml
      ;;
    tworoom)
      run_env tworoom configs/research/identity_delta_tworoom_benchmark.yaml
      ;;
    *)
      echo "ERROR: unknown MWM_SWEEP_ENVS entry: ${env_name}" >&2
      exit 2
      ;;
  esac
done
