from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

from mwm.checkpoint_io import load_world_model_from_checkpoint


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the initial LeWM commit's local JEPA inference graph with the released Stable-WM object."
    )
    parser.add_argument("--historical-lewm-source", required=True)
    parser.add_argument(
        "--raw-checkpoint",
        default="/vast/home/e/ethanyu/.stable_worldmodel/checkpoints/models--quentinll--lewm-reacher",
    )
    parser.add_argument(
        "--mwm-checkpoint",
        default="checkpoints_mwm/upstream_lewm_reacher",
    )
    parser.add_argument(
        "--output",
        default="reports/research/single_level_reacher_parity_20260806/historical_lewm_inference_parity.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the released-model inference parity gate")
    device = torch.device("cuda")
    checkpoint = Path(args.raw_checkpoint)
    source = Path(args.historical_lewm_source).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=False)

    # Instantiate the released post-export class before making the historical
    # top-level jepa.py/module.py importable.
    released = instantiate(config)
    released.load_state_dict(state, strict=True)

    sys.path.insert(0, str(source))
    historical_config = json.loads(json.dumps(config))
    historical_config["_target_"] = "jepa.JEPA"
    historical_config["predictor"]["_target_"] = "module.ARPredictor"
    historical_config["action_encoder"]["_target_"] = "module.Embedder"
    historical_config["projector"]["_target_"] = "module.MLP"
    historical_config["pred_proj"]["_target_"] = "module.MLP"
    historical = instantiate(historical_config)
    historical.load_state_dict(state, strict=True)
    mwm, _, _ = load_world_model_from_checkpoint(args.mwm_checkpoint, None, device)
    mwm.set_planning_rollout_semantics("upstream_lewm_historical")

    released_state = released.state_dict()
    historical_state = historical.state_dict()
    if released_state.keys() != historical_state.keys():
        raise RuntimeError("Historical and released state keys differ")
    weight_max_abs = max(max_abs(released_state[name], historical_state[name]) for name in released_state)
    if weight_max_abs != 0.0:
        raise RuntimeError(f"Historical and released weights differ: {weight_max_abs}")

    released = released.to(device).eval()
    historical = historical.to(device).eval()
    mwm = mwm.to(device).eval()
    torch.manual_seed(20260806)
    pixels = torch.randn(1, 3, 3, 224, 224, device=device)
    actions = torch.randn(1, 3, 10, device=device)
    rollout_pixels = pixels[:, None].expand(-1, 4, -1, -1, -1, -1).clone()
    candidates = torch.randn(1, 4, 6, 10, device=device)
    goal = torch.randn(1, 4, 1, 3, 224, 224, device=device)

    with torch.inference_mode():
        released_encoded = released.encode({"pixels": pixels.clone(), "action": actions.clone()})
        historical_encoded = historical.encode({"pixels": pixels.clone(), "action": actions.clone()})
        encode_max_abs = max_abs(released_encoded["emb"], historical_encoded["emb"])
        action_encode_max_abs = max_abs(released_encoded["act_emb"], historical_encoded["act_emb"])
        released_pred = released.predict(released_encoded["emb"], released_encoded["act_emb"])
        historical_pred = historical.predict(historical_encoded["emb"], historical_encoded["act_emb"])
        predict_max_abs = max_abs(released_pred, historical_pred)
        released_rollout = released.rollout(
            {"pixels": rollout_pixels.clone()}, candidates.clone()
        )["predicted_emb"]
        historical_rollout = historical.rollout(
            {"pixels": rollout_pixels.clone()}, candidates.clone()
        )["predicted_emb"]
        rollout_max_abs = max_abs(released_rollout, historical_rollout)
        released_cost = released.get_cost(
            {"pixels": rollout_pixels.clone(), "goal": goal.clone(), "action": candidates.clone()},
            candidates.clone(),
        )
        historical_cost = historical.get_cost(
            {"pixels": rollout_pixels.clone(), "goal": goal.clone(), "action": candidates.clone()},
            candidates.clone(),
        )
        cost_max_abs = max_abs(released_cost, historical_cost)
        cost_order_equal = bool(torch.equal(torch.argsort(released_cost), torch.argsort(historical_cost)))
        mwm_encoded = mwm.encode(
            {"pixels": pixels.clone(), "action": actions.clone()},
            already_preprocessed=True,
        )
        historical_mwm_encode_max_abs = max_abs(historical_encoded["emb"], mwm_encoded["emb"])
        historical_mwm_action_encode_max_abs = max_abs(historical_encoded["act_emb"], mwm_encoded["act_emb"])
        mwm_rollout = mwm.rollout_at_level(
            {"pixels": rollout_pixels.clone()}, candidates.clone(), level_idx=0
        )["predicted_emb"]
        historical_mwm_rollout_max_abs = max_abs(historical_rollout, mwm_rollout)
        decision = type(
            "Decision",
            (),
            {
                "base_level_idx": 0,
                "rollout_level_indices": [0] * int(candidates.shape[2]),
                "flop_accounting": "none",
            },
        )()
        mwm_cost = mwm.get_cost_with_fidelity(
            {"pixels": rollout_pixels.clone(), "goal": goal.clone(), "action": candidates.clone()},
            candidates.clone(),
            decision,
        )
        historical_mwm_cost_max_abs = max_abs(historical_cost, mwm_cost)
        historical_mwm_cost_order_equal = bool(
            torch.equal(torch.argsort(historical_cost), torch.argsort(mwm_cost))
        )

    metrics = {
        "weight_max_abs": weight_max_abs,
        "encode_max_abs": encode_max_abs,
        "action_encode_max_abs": action_encode_max_abs,
        "predict_max_abs": predict_max_abs,
        "rollout_max_abs": rollout_max_abs,
        "cost_max_abs": cost_max_abs,
        "cost_order_equal": cost_order_equal,
        "historical_mwm_encode_max_abs": historical_mwm_encode_max_abs,
        "historical_mwm_action_encode_max_abs": historical_mwm_action_encode_max_abs,
        "historical_mwm_rollout_max_abs": historical_mwm_rollout_max_abs,
        "historical_mwm_cost_max_abs": historical_mwm_cost_max_abs,
        "historical_mwm_cost_order_equal": historical_mwm_cost_order_equal,
    }
    # The historical rollout repeatedly re-embeds the growing action prefix;
    # the released implementation embeds the complete action sequence once.
    # They are mathematically equivalent but accumulate float32 reductions in
    # a different order, so cost differs at roughly 3e-5 on this adversarial
    # random probe while candidate ordering remains identical.
    tolerance = 1e-4
    if max(value for key, value in metrics.items() if key.endswith("max_abs")) > tolerance:
        raise RuntimeError(f"Historical inference parity exceeded tolerance {tolerance}: {metrics}")
    if not cost_order_equal:
        raise RuntimeError(f"Historical inference changed candidate ordering: {metrics}")
    historical_mwm_metrics = [
        historical_mwm_encode_max_abs,
        historical_mwm_action_encode_max_abs,
        historical_mwm_rollout_max_abs,
        historical_mwm_cost_max_abs,
    ]
    if max(historical_mwm_metrics) > 1e-6 or not historical_mwm_cost_order_equal:
        raise RuntimeError(f"MWM historical rollout semantics did not reproduce initial LeWM: {metrics}")

    output = {
        "status": "pass",
        "historical_lewm_commit": "83f97d72ad067855bc89a1b74b4aff11d4dfdf0c",
        "historical_jepa_sha256": hashlib.sha256((source / "jepa.py").read_bytes()).hexdigest(),
        "historical_module_sha256": hashlib.sha256((source / "module.py").read_bytes()).hexdigest(),
        "released_weights_sha256": hashlib.sha256((checkpoint / "weights.pt").read_bytes()).hexdigest(),
        "mwm_checkpoint": str(Path(args.mwm_checkpoint).resolve()),
        "parameter_count": sum(parameter.numel() for parameter in released.parameters()),
        "metrics": metrics,
        "tolerance": tolerance,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
