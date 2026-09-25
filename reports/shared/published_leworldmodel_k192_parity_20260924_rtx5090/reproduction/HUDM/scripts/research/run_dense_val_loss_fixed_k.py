from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mwm.benchmark.matrix import main as run_benchmark_matrix


LEVELS = (6, 12, 48, 96, 144, 192)
LEARNING_RATES = (
    ("1e-6", "1em6"),
    ("1e-5", "1em5"),
    ("2e-5", "2em5"),
    ("5e-5", "5em5"),
)
RUN_SUFFIX = "warm10_20260702_070021"
DEFAULT_OUTPUT_ROOT = Path("reports/research/dense_val_loss_fixed_k_20260721")

ENVIRONMENTS = {
    "pusht": {
        "env_id": "swm/PushT-v1",
        "eval_config": "configs/eval/paper_pusht.yaml",
        "manifest": "rollouts/manifests/pusht_paper_seed42.json",
    },
    "reacher": {
        "env_id": "swm/ReacherDMControl-v0",
        "eval_config": "configs/eval/paper_reacher.yaml",
        "manifest": "rollouts/manifests/reacher_paper_seed42.json",
    },
    "ogb_cube": {
        "env_id": "swm/OGBCube-v0",
        "eval_config": "configs/eval/paper_ogb_cube.yaml",
        "manifest": "rollouts/manifests/ogb_cube_paper_seed42.json",
    },
    "tworoom": {
        "env_id": "swm/TwoRoom-v1",
        "eval_config": "configs/eval/paper_tworoom.yaml",
        "manifest": "rollouts/manifests/tworoom_paper_seed42.json",
    },
}


def build_config(env_slug: str, output_root: Path, learning_rate_token: str | None = None) -> dict:
    spec = ENVIRONMENTS[env_slug]
    runs = []
    for _, lr_token in LEARNING_RATES:
        if learning_rate_token is not None and lr_token != learning_rate_token:
            continue
        checkpoint = f"checkpoints_mwm/mwm_dense_{env_slug}_lr{lr_token}_{RUN_SUFFIX}"
        for level_idx, k in enumerate(LEVELS):
            runs.append(
                {
                    "name": f"lr{lr_token}_fixed_k{k}",
                    "role": f"dense_{env_slug}_lr{lr_token}_fixed_k{k}",
                    "checkpoint": checkpoint,
                    "planner": {
                        "scheduler": {
                            "enabled": True,
                            "mpc": {"mode": "fixed", "level": level_idx},
                            "cem": {"mode": "fixed", "level": "base"},
                            "rollout": {"mode": "fixed", "level": "base"},
                        }
                    },
                }
            )
    return {
        "output_dir": str(
            output_root
            / "rollouts"
            / (env_slug if learning_rate_token is None else f"{env_slug}_lr{learning_rate_token}")
        ),
        "title": f"Dense LR sweep: validation loss vs fixed-K planning ({env_slug})",
        "env_id": spec["env_id"],
        "seed": 42,
        "eval_config": spec["eval_config"],
        "manifest": {
            "group": f"{env_slug}_paper_seed42",
            "path": spec["manifest"],
        },
        "runs": runs,
    }


def write_config(env_slug: str, output_root: Path, learning_rate_token: str | None = None) -> Path:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    suffix = env_slug if learning_rate_token is None else f"{env_slug}_lr{learning_rate_token}"
    config_path = config_dir / f"fixed_k_{suffix}.yaml"
    config_path.write_text(
        OmegaConf.to_yaml(build_config(env_slug, output_root, learning_rate_token)), encoding="utf-8"
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every released dense LR-sweep checkpoint at each fixed Matryoshka level."
    )
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--learning-rate-token",
        choices=[token for _, token in LEARNING_RATES],
        default=None,
        help="Optionally evaluate just one learning-rate block into its own shard directory.",
    )
    parser.add_argument(
        "--write-config-only",
        action="store_true",
        help="Write the resolved benchmark matrix without running GPU evaluation.",
    )
    args = parser.parse_args()

    config_path = write_config(args.env, args.output_root, args.learning_rate_token)
    print(config_path)
    if not args.write_config_only:
        run_benchmark_matrix(str(config_path))


if __name__ == "__main__":
    main()
