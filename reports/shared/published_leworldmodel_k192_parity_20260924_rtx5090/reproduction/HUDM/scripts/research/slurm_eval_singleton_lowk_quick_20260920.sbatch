#!/usr/bin/env bash
#SBATCH --job-name=eval_singleton_lowk
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: sbatch $0 ENV_SLUG K" >&2
  exit 2
fi

ENV_SLUG="$1"
K="$2"
case "$K" in
  48|72|96|120|144|168|192) ;;
  *)
    echo "ERROR: K must be one of 48,72,96,120,144,168,192; got $K" >&2
    exit 2
    ;;
esac

ROOT="${SLURM_SUBMIT_DIR:-/vast/projects/dineshj/lab/ethanyu/code/HUDM}"
PY="${MWM_PYTHON:-/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python}"
RUN_DATE="${MWM_RUN_DATE:-20260920}"
RUN_NAME="mwm_paper10_${ENV_SLUG}_k${K}_sepopt_retrain_${RUN_DATE}"

case "$ENV_SLUG" in
  tworoom)
    ENV_ID="swm/TwoRoom-v1"
    EVAL_CONFIG="$ROOT/configs/eval/research_tworoom_exact_seed42_n50_20260913.yaml"
    MANIFEST="$ROOT/rollouts/manifests/tworoom_paper_seed42.json"
    ;;
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
    echo "ERROR: unsupported environment slug: $ENV_SLUG" >&2
    exit 2
    ;;
esac

CHECKPOINT="$ROOT/checkpoints_mwm/$RUN_NAME"
REPORT_ROOT="$ROOT/reports/research/singleton_${ENV_SLUG}_k${K}_sepopt_retrain_quick_${RUN_DATE}"
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
  --expected-levels "$K"
  --episodes 50
  --num-envs 50
  --seed 42
  --eval-config "$EVAL_CONFIG"
  --manifest "$MANIFEST"
  --flop-accounting dynamics_audit
)
"$PY" "$RUNNER" "${COMMON_ARGS[@]}" --write-config-only
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH" --static-only
"$PY" -m mwm.benchmark.matrix "$CONFIG_PATH"
"$PY" -m mwm.benchmark.verify "$CONFIG_PATH"
test -s "$OUTPUT_DIR/summary.json"
test -s "$OUTPUT_DIR/summary.csv"
echo "SINGLETON_QUICK_COMPLETE environment=$ENV_SLUG K=$K episodes=50 seed=42 flop_accounting=dynamics_audit"
