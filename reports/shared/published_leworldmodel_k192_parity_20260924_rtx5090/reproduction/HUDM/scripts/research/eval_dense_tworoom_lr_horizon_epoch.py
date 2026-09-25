from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from omegaconf import OmegaConf

import mwm.eval.execution as eval_execution
from mwm.benchmark.matrix import main as run_benchmark_matrix


LEVELS = (96, 120, 144, 168, 192)
ROLE_SPECS = (
    ("fixed_k96", 0, None),
    ("fixed_k144", 2, None),
    ("fixed_k192", 4, None),
    ("k192_cost_k96", 4, 96),
    ("k192_cost_k144", 4, 144),
)


def _cost_with_prefix(model: Any, prefix_k: int) -> MethodType:
    original = model.get_cost_with_fidelity

    @torch.no_grad()
    def get_cost_with_fidelity(
        self: Any,
        infos: dict[str, Any],
        candidates: torch.Tensor,
        decision: Any,
    ) -> torch.Tensor:
        original(infos, candidates, decision)
        pred_emb = infos["predicted_emb"]
        goal_emb = infos["goal_emb"]
        terminal_k = int(
            getattr(decision, "metadata", {}).get("terminal_k")
            or getattr(decision, "base_k", None)
            or self.K[-1]
        )
        if prefix_k > terminal_k:
            raise ValueError(f"Cost prefix K={prefix_k} exceeds terminal transition K={terminal_k}")
        if goal_emb.ndim == 2:
            goal_emb = goal_emb[:, None, None, :]
        elif goal_emb.ndim == 3:
            goal_emb = goal_emb[:, None, :, :]
        goal = goal_emb[..., -1:, :prefix_k].expand_as(pred_emb[..., -1:, :prefix_k])
        cost = (pred_emb[..., -1:, :prefix_k] - goal.detach()).pow(2).sum(
            dim=tuple(range(2, pred_emb.ndim))
        )
        diagnostics = dict(getattr(self, "_last_cost_diagnostics", {}))
        diagnostics.update({"research_cost_prefix_k": prefix_k, "prefix_criterion": True})
        self._last_cost_diagnostics = diagnostics
        return cost

    return MethodType(get_cost_with_fidelity, model)


_ORIGINAL_BUILD_POLICY = eval_execution.build_mwm_policy


def _build_policy_with_cost_prefix(
    model: Any,
    metadata: dict[str, Any],
    cfg: Any,
    device: torch.device,
    action_space: Any,
    process: dict[str, Any],
) -> Any:
    prefix = cfg.planner.get("research_cost_prefix_k", None)
    if prefix is not None:
        model.get_cost_with_fidelity = _cost_with_prefix(model, int(prefix))
    return _ORIGINAL_BUILD_POLICY(model, metadata, cfg, device, action_space, process)


eval_execution.build_mwm_policy = _build_policy_with_cost_prefix


def build_config(
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    episodes: int,
    num_envs: int,
    eval_config: Path,
    manifest: Path,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for role, level_idx, cost_prefix in ROLE_SPECS:
        planner: dict[str, Any] = {
            "scheduler": {
                "enabled": True,
                "mpc": {"mode": "fixed", "level": level_idx},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            }
        }
        if cost_prefix is not None:
            planner["research_cost_prefix_k"] = cost_prefix
        runs.append(
            {
                "name": role,
                "role": role,
                "checkpoint": str(checkpoint),
                "planner": planner,
                "eval": {"episodes": episodes, "num_envs": num_envs, "seed": seed},
            }
        )
    return {
        "output_dir": str(output_dir),
        "title": f"Dense TwoRoom LR-horizon epoch screen: {checkpoint.name}",
        "env_id": "swm/TwoRoom-v1",
        "seed": seed,
        "eval_config": str(eval_config),
        "manifest": {
            "group": f"tworoom_dense_lr_horizon_seed{seed}_n{episodes}",
            "path": str(manifest),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one saved dense TwoRoom epoch.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("configs/eval/research_tworoom_exact_seed42_n50_20260913.yaml"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("rollouts/manifests/tworoom_paper_seed42.json"))
    parser.add_argument("--write-config-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.num_envs > args.episodes:
        raise ValueError("Require 1 <= num_envs <= episodes.")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.eval_config.is_file():
        raise FileNotFoundError(args.eval_config)

    config_path = args.output_dir.parent / f"{args.output_dir.name}_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        OmegaConf.to_yaml(
            build_config(
                args.checkpoint,
                args.output_dir,
                args.seed,
                args.episodes,
                args.num_envs,
                args.eval_config,
                args.manifest,
            )
        ),
        encoding="utf-8",
    )
    if not args.write_config_only:
        run_benchmark_matrix(str(config_path))


if __name__ == "__main__":
    main()
