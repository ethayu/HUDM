#!/usr/bin/env bash
#SBATCH --job-name=upstream_reacher_h5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch." >&2
  exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
RAW_MODEL="/vast/home/e/ethanyu/.stable_worldmodel/checkpoints/models--quentinll--lewm-reacher"
LEWM_COMMIT="83f97d72ad067855bc89a1b74b4aff11d4dfdf0c"
SWM_COMMIT="2096f6a17498f6283881e141d1c0579536815485"
EVAL_ROOT="/tmp/mwm-upstream-reacher-h5-${SLURM_JOB_ID}"
SOURCE_ROOT="$EVAL_ROOT/le-wm"
SWM_SOURCE_ROOT="$EVAL_ROOT/stable-worldmodel"
CEM_STEPS="${MWM_CEM_STEPS:-30}"
REPORT="${MWM_REPORT:-$ROOT/reports/research/single_level_reacher_parity_20260806/upstream_eval_native_h5_seed42_iter30_threshold010.txt}"

cd "$ROOT"
mkdir -p logs "$EVAL_ROOT/dmc" "$EVAL_ROOT/datasets/dmc" "$EVAL_ROOT/checkpoints" "$(dirname "$REPORT")"
test -s "$RAW_MODEL/config.json"
test -s "$RAW_MODEL/weights.pt"

git clone --quiet --no-checkout https://github.com/lucas-maes/le-wm.git "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" checkout --quiet "$LEWM_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$LEWM_COMMIT"
git clone --quiet --no-checkout https://github.com/galilai-group/stable-worldmodel.git "$SWM_SOURCE_ROOT"
git -C "$SWM_SOURCE_ROOT" checkout --quiet "$SWM_COMMIT"
test "$(git -C "$SWM_SOURCE_ROOT" rev-parse HEAD)" = "$SWM_COMMIT"

tar --use-compress-program=unzstd -xf "$ROOT/data/upstream/reacher.tar.zst" -C "$EVAL_ROOT" reacher.h5
echo "85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06  $EVAL_ROOT/reacher.h5" | sha256sum -c -
ln -s "$EVAL_ROOT/reacher.h5" "$EVAL_ROOT/datasets/dmc/reacher_random.h5"
ln -s "$EVAL_ROOT/reacher.h5" "$EVAL_ROOT/dmc/reacher_random.h5"

export STABLEWM_HOME="$EVAL_ROOT"
export PYTHONPATH="$SWM_SOURCE_ROOT:$SOURCE_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# The released Hugging Face artifact is config.json + weights.pt, whereas the
# paper-era AutoCostModel only accepts the Python object checkpoint emitted by
# LeWM's ModelObjectCallBack. Reconstitute that serialization without changing
# the model or evaluator source, and verify every loaded tensor first.
UPSTREAM_OBJECT_STEM="$EVAL_ROOT/models--quentinll--lewm-reacher"
"$PY" - "$RAW_MODEL" "${UPSTREAM_OBJECT_STEM}_object.ckpt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

source = Path(sys.argv[1])
output = Path(sys.argv[2])
config = json.loads((source / "config.json").read_text(encoding="utf-8"))
# The Hub artifact was re-exported after JEPA was renamed/moved into
# stable_worldmodel.wm.lewm. The initial LeWM commit built the same graph from
# its local jepa.py/module.py classes; use those historical import paths so the
# object can be loaded by the untouched paper-era evaluator.
config["_target_"] = "jepa.JEPA"
config["predictor"]["_target_"] = "module.ARPredictor"
config["action_encoder"]["_target_"] = "module.Embedder"
config["projector"]["_target_"] = "module.MLP"
config["pred_proj"]["_target_"] = "module.MLP"
model = instantiate(config)
state = torch.load(source / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=True)
for name, tensor in model.state_dict().items():
    if not torch.equal(tensor, state[name]):
        raise RuntimeError(f"Object reconstruction changed tensor {name}")
torch.save(model, output)
print(
    "Reconstituted historical object checkpoint:",
    output,
    "parameters=", sum(parameter.numel() for parameter in model.parameters()),
    "state_sha256=", hashlib.sha256((source / "weights.pt").read_bytes()).hexdigest(),
)
PY

cd "$SOURCE_ROOT"
"$PY" eval.py \
  --config-name=reacher.yaml \
  policy="$UPSTREAM_OBJECT_STEM" \
  cache_dir="$EVAL_ROOT" \
  "solver.n_steps=$CEM_STEPS" \
  output.filename=upstream_reacher_native_h5_results.txt

test -s "$EVAL_ROOT/upstream_reacher_native_h5_results.txt"
cp "$EVAL_ROOT/upstream_reacher_native_h5_results.txt" "$REPORT"
