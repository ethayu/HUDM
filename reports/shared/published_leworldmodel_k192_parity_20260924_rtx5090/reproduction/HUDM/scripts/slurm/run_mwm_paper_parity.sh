#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: scripts/slurm/run_mwm_paper_parity.sh must run inside a Slurm allocation. Submit scripts/slurm/slurm_mwm_paper_parity.sbatch with sbatch." >&2
  exit 2
fi

cd "$ROOT"

"$PY" -m mwm.upstream.lewm_checkpoints
"$PY" -m mwm.upstream.lewm_data
"$PY" -m mwm.data.verify --paper-parity

"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_pusht.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_pusht.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_reacher.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_reacher.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_ogb_cube.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_ogb_cube.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_tworoom.yaml --roles upstream_lewm_converted
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_tworoom.yaml --roles upstream_lewm_converted

"$PY" -m mwm.training.stable_wm configs/train/mwm_lewm_pusht_upstream.yaml
"$PY" -m mwm.training.stable_wm configs/train/mwm_lewm_reacher_upstream.yaml
"$PY" -m mwm.training.stable_wm configs/train/mwm_lewm_ogb_cube_upstream.yaml
"$PY" -m mwm.training.stable_wm configs/train/mwm_lewm_tworoom_upstream.yaml
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_pusht.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_pusht.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_reacher.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_reacher.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_ogb_cube.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_ogb_cube.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.matrix configs/benchmark/paper_parity_tworoom.yaml --roles upstream_lewm_converted retrained_lewm_identity
"$PY" -m mwm.benchmark.verify configs/benchmark/paper_parity_tworoom.yaml --roles upstream_lewm_converted retrained_lewm_identity
