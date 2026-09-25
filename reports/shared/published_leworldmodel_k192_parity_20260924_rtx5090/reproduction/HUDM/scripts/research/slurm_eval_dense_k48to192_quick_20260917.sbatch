#!/usr/bin/env bash
#SBATCH --job-name=eval_dense_k48to192
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: sbatch $0 ENV_SLUG RUN_NAME" >&2
  exit 2
fi

ENV_SLUG="$1"
RUN_NAME="$2"
ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"

case "$ENV_SLUG" in
  ogb_cube)
    ENV_ID="swm/OGBCube-v0"
    EVAL_CONFIG="$ROOT/configs/eval/research_ogb_cube_exact_seed42_n50_20260917.yaml"
    MANIFEST="$ROOT/rollouts/manifests/ogb_cube_paper_seed42.json"
    ;;
  reacher)
    ENV_ID="swm/ReacherDMControl-v0"
    EVAL_CONFIG="$ROOT/configs/eval/research_reacher_exact_seed42_n50_20260917.yaml"
    MANIFEST="$ROOT/rollouts/manifests/reacher_paper_seed42.json"
    ;;
  pusht)
    ENV_ID="swm/PushT-v1"
    EVAL_CONFIG="$ROOT/configs/eval/research_pusht_exact_seed42_n50_20260917.yaml"
    MANIFEST="$ROOT/rollouts/manifests/pusht_paper_seed42.json"
    ;;
  *)
    echo "Unsupported environment slug: $ENV_SLUG" >&2
    exit 2
    ;;
esac

CHECKPOINT="$ROOT/checkpoints_mwm/$RUN_NAME"
REPORT_ROOT="$ROOT/reports/research/dense_${ENV_SLUG}_k48_72_96_120_144_168_192_quick_20260917"
OUTPUT_DIR="$REPORT_ROOT/final_n50_seed42"
CONFIG_PATH="$REPORT_ROOT/final_n50_seed42_config.yaml"
RUNNER="$ROOT/scripts/research/eval_dense_fixed_levels_quick.py"

cd "$ROOT"
mkdir -p logs "$REPORT_ROOT"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite existing evaluation output: $OUTPUT_DIR" >&2
  exit 3
fi
for name in config.json weights.pt world_metadata.json; do test -s "$CHECKPOINT/$name"; done

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --env-id "$ENV_ID"
  --expected-levels 48 72 96 120 144 168 192
  --episodes 50
  --num-envs 50
  --seed 42
  --eval-config "$EVAL_CONFIG"
  --manifest "$MANIFEST"
)
"$PY" "$RUNNER" "${COMMON_ARGS[@]}" --write-config-only
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH" --static-only
# Run the exact matrix that was just statically verified. Re-entering RUNNER
# here rewrites CONFIG_PATH and can race with other benchmark preparation,
# silently replacing the exact-semantics eval template before execution.
"$PY" -m mwm.benchmark.matrix "$CONFIG_PATH"
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH"
test -s "$OUTPUT_DIR/summary.json"
test -s "$OUTPUT_DIR/summary.csv"
echo "DENSE_${ENV_SLUG}_K48TO192_QUICK_EVAL_COMPLETE episodes=50 seed=42"
