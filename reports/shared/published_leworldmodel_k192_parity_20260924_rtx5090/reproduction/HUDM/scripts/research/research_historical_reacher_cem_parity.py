from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MWM fixed-fidelity CEM directly with the paper-era Stable-WM solver."
    )
    parser.add_argument("--stable-worldmodel-source", required=True)
    parser.add_argument(
        "--output",
        default="reports/research/single_level_reacher_parity_20260806/historical_cem_parity.json",
    )
    args = parser.parse_args()

    source_root = Path(args.stable_worldmodel_source).resolve()
    historical_cem_path = source_root / "stable_worldmodel" / "solver" / "cem.py"
    if not historical_cem_path.is_file():
        raise FileNotFoundError(historical_cem_path)

    # Import the historical package first so this gate executes that source,
    # rather than the current environment's Stable-WM installation.
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
    from stable_worldmodel.solver.cem import CEMSolver as HistoricalCEMSolver

    imported_path = Path(sys.modules[HistoricalCEMSolver.__module__].__file__).resolve()
    if imported_path != historical_cem_path:
        raise RuntimeError(f"Imported {imported_path}, expected {historical_cem_path}")

    from mwm.planning.scheduled_cem import MWMScheduledCEMSolver

    class CostModel(torch.nn.Module):
        K = (192,)
        D = 192
        min_k = 192
        num_levels = 1
        supports_arbitrary_k = False

        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        @staticmethod
        def _cost(info: dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
            return (candidates - info["target"]).square().sum(dim=(-1, -2))

        def get_cost(self, info: dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
            return self._cost(info, candidates)

        def get_cost_with_fidelity(
            self, info: dict[str, torch.Tensor], candidates: torch.Tensor, decision: object
        ) -> torch.Tensor:
            del decision
            return self._cost(info, candidates)

    model = CostModel()
    n_envs = 2
    action_space = Box(low=-10.0, high=10.0, shape=(n_envs, 2), dtype=np.float32)
    plan_config = SimpleNamespace(horizon=3, receding_horizon=1, action_block=2, warm_start=True)
    target = torch.tensor(
        [
            [[0.1, -0.2, 0.3, -0.4], [0.2, -0.1, 0.4, -0.3], [0.3, -0.4, 0.1, -0.2]],
            [[-0.3, 0.4, -0.1, 0.2], [-0.4, 0.3, -0.2, 0.1], [-0.1, 0.2, -0.3, 0.4]],
        ],
        dtype=torch.float32,
    )
    common = {
        "model": model,
        "batch_size": 1,
        "num_samples": 32,
        "var_scale": 1.0,
        "n_steps": 6,
        "topk": 8,
        "device": "cpu",
        "seed": 42,
    }
    historical = HistoricalCEMSolver(**common)
    mwm = MWMScheduledCEMSolver(
        **common,
        scheduler={
            "enabled": True,
            "mpc": {"mode": "fixed", "level": 0},
            "cem": {"mode": "fixed", "level": "base"},
            "rollout": {"mode": "fixed", "level": "base"},
        },
        std_unbiased=True,
    )
    historical.configure(action_space=action_space, n_envs=n_envs, config=plan_config)
    mwm.configure(action_space=action_space, n_envs=n_envs, config=plan_config)
    historical_result = historical.solve({"target": target.clone()})
    mwm_result = mwm.solve({"target": target.clone()})

    comparisons = {
        "actions_bitwise_equal": bool(torch.equal(historical_result["actions"], mwm_result["actions"])),
        "mean_bitwise_equal": bool(torch.equal(historical_result["mean"][0], mwm_result["mean"][0])),
        "var_bitwise_equal": bool(torch.equal(historical_result["var"][0], mwm_result["var"][0])),
        "costs_bitwise_equal": historical_result["costs"] == mwm_result["costs"],
        "actions_max_abs": float(
            (historical_result["actions"] - mwm_result["actions"]).abs().max().item()
        ),
        "mean_max_abs": float(
            (historical_result["mean"][0] - mwm_result["mean"][0]).abs().max().item()
        ),
        "var_max_abs": float(
            (historical_result["var"][0] - mwm_result["var"][0]).abs().max().item()
        ),
        "costs_max_abs": float(
            np.max(np.abs(np.asarray(historical_result["costs"]) - np.asarray(mwm_result["costs"])))
        ),
    }
    if not all(value for key, value in comparisons.items() if key.endswith("bitwise_equal")):
        raise RuntimeError(f"Historical CEM parity failed: {comparisons}")

    output = {
        "status": "pass",
        "historical_stable_worldmodel_commit": "2096f6a17498f6283881e141d1c0579536815485",
        "historical_cem_path": "stable_worldmodel/solver/cem.py",
        "historical_cem_sha256": sha256(historical_cem_path),
        "mwm_cem_path": "mwm/planning/scheduled_cem.py",
        "mwm_cem_sha256": sha256(Path(__file__).resolve().parents[2] / "mwm/planning/scheduled_cem.py"),
        "settings": {
            "n_envs": n_envs,
            "batch_size": common["batch_size"],
            "num_samples": common["num_samples"],
            "n_steps": common["n_steps"],
            "topk": common["topk"],
            "horizon": plan_config.horizon,
            "action_block": plan_config.action_block,
            "seed": common["seed"],
            "std_unbiased": True,
        },
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
