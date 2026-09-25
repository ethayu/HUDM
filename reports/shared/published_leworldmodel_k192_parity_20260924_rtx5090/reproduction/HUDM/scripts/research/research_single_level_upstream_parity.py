from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
from hydra.utils import instantiate

from mwm.adapters.builder import build_mwm_from_stable_config
from mwm.adapters.stable_config import load_stable_wm_config, stable_config_sha256
from mwm.checkpoint_io import load_world_model_from_checkpoint
from mwm.training.stable_wm_lightning import (
    stable_wm_parameter_partitions,
    stable_wm_upstream_lewm_parameter_order,
)


RAW_TO_MWM_PREFIXES = (
    ("encoder.", "encoder."),
    ("predictor.", "transitions.0.predictor."),
    ("action_encoder.", "transitions.0.action_encoder."),
    ("projector.", "projector."),
    ("pred_proj.", "transitions.0.pred_proj."),
)


def _mapped_name(raw_name: str) -> str:
    for raw_prefix, mwm_prefix in RAW_TO_MWM_PREFIXES:
        if raw_name.startswith(raw_prefix):
            return mwm_prefix + raw_name[len(raw_prefix) :]
    raise KeyError(raw_name)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu().item())


def _assert_close(
    label: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float = 2e-6,
    rtol: float = 1e-5,
) -> float:
    value = _max_abs(left, right)
    torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=label)
    return value


def _load_raw(checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    model = instantiate(config).to(device)
    state = torch.load(checkpoint_dir / "weights.pt", map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    return model


def _fresh_initialization_parity(checkpoint_dir: Path) -> tuple[int, float]:
    """Verify that the extra decoder is initialized only after an identical world model."""
    source_config, config_path = load_stable_wm_config(checkpoint_dir)
    seed = 3072
    torch.manual_seed(seed)
    raw = instantiate(source_config)
    torch.manual_seed(seed)
    mwm = build_mwm_from_stable_config(
        family="lewm",
        source_config=source_config,
        source_config_sha256=stable_config_sha256(config_path),
        training_recipe={
            "history_size": 3,
            "num_preds": 1,
            "action_preprocessing": "standard_scaler",
            "loss_scope": {
                "regularizers": "shared_latent",
                "reconstructor_detach_encoder": True,
                "reconstructor_contributes_to_encoder_loss": False,
            },
        },
        K=(192,),
        action_dim=10,
        expected_D=192,
        action_block=5,
        image_shape=(224, 224),
        normalize_imagenet=True,
        component_policy={
            "shared": ["latent_producer"],
            "per_level": ["transition"],
            "reconstructor": ["decoder"],
        },
    )
    raw_state = raw.state_dict()
    mwm_state = mwm.state_dict()
    maximum = 0.0
    for raw_name, tensor in raw_state.items():
        maximum = max(
            maximum,
            _assert_close(
                f"fresh init {raw_name}",
                tensor,
                mwm_state[_mapped_name(raw_name)],
                atol=0.0,
                rtol=0.0,
            ),
        )
    count = sum(parameter.numel() for parameter in raw.parameters())
    if count != sum(parameter.numel() for name, parameter in mwm.named_parameters() if not name.startswith("decoders.")):
        raise RuntimeError("Fresh raw and MWM world parameter counts differ.")
    del raw, mwm
    return count, maximum


def _world_named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if not name.startswith("decoders.")}


def _clip(parameters: Iterable[torch.nn.Parameter], value: float) -> float:
    norm = torch.nn.utils.clip_grad_norm_(tuple(parameters), value)
    return float(norm.detach().float().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove full-width single-level MWM parity with raw upstream LeWM.")
    parser.add_argument(
        "--raw-checkpoint",
        default="/vast/home/e/ethanyu/.stable_worldmodel/checkpoints/models--quentinll--lewm-reacher",
    )
    parser.add_argument("--mwm-checkpoint", default="checkpoints_mwm/upstream_lewm_reacher")
    parser.add_argument(
        "--output",
        default="reports/research/single_level_upstream_parity_20260806/report.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required because upstream SIGReg samples projections on device='cuda'.")
    # Compare the two graphs under identical deterministic kernels. Flash and
    # memory-efficient SDPA backward use atomic reductions, so two otherwise
    # identical graphs can differ at the last few bits solely because their
    # allocations launch a different reduction schedule.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    fresh_count, fresh_init_max_abs = _fresh_initialization_parity(Path(args.raw_checkpoint))
    torch.manual_seed(20260806)
    raw = _load_raw(Path(args.raw_checkpoint), device)
    mwm, _, _ = load_world_model_from_checkpoint(Path(args.mwm_checkpoint), None, device)
    raw.eval()
    mwm.eval()

    raw_state = raw.state_dict()
    mwm_state = mwm.state_dict()
    weight_max_abs = 0.0
    for raw_name, tensor in raw_state.items():
        weight_max_abs = max(
            weight_max_abs,
            _assert_close(
                raw_name,
                tensor,
                mwm_state[_mapped_name(raw_name)],
                atol=0.0,
                rtol=0.0,
            ),
        )

    raw_params = dict(raw.named_parameters())
    mwm_params = _world_named_parameters(mwm)
    raw_parameter_layout = [
        {
            "index": index,
            "raw_name": raw_name,
            "mwm_name": _mapped_name(raw_name),
            "shape": list(parameter.shape),
        }
        for index, (raw_name, parameter) in enumerate(raw_params.items())
    ]
    raw_module_boundaries = [
        row
        for index, row in enumerate(raw_parameter_layout)
        if index == 0
        or row["raw_name"].split(".", 1)[0]
        != raw_parameter_layout[index - 1]["raw_name"].split(".", 1)[0]
    ]
    mapped_names = {_mapped_name(name) for name in raw_params}
    if mapped_names != set(mwm_params):
        raise RuntimeError(
            f"World parameter mapping differs: missing={sorted(set(mwm_params) - mapped_names)[:8]}, "
            f"extra={sorted(mapped_names - set(mwm_params))[:8]}"
        )
    raw_count = sum(param.numel() for param in raw_params.values())
    mwm_count = sum(param.numel() for param in mwm_params.values())
    if raw_count != mwm_count:
        raise RuntimeError(f"World parameter counts differ: raw={raw_count}, mwm={mwm_count}")

    batch_size, timesteps, action_dim = 2, 4, 10
    pixels = torch.randn(batch_size, timesteps, 3, 224, 224, device=device)
    actions = torch.randn(batch_size, timesteps, action_dim, device=device)
    with torch.no_grad():
        raw_encoded = raw.encode({"pixels": pixels.clone(), "action": actions.clone()})
        mwm_encoded = mwm.encode(
            {"pixels": pixels.clone(), "action": actions.clone()},
            already_preprocessed=True,
        )
        encode_max_abs = _assert_close("encoded latent", raw_encoded["emb"], mwm_encoded["emb"])
        action_encode_max_abs = _assert_close("encoded action", raw_encoded["act_emb"], mwm_encoded["act_emb"])
        raw_pred = raw.predict(raw_encoded["emb"][:, :3], raw_encoded["act_emb"][:, :3])
        mwm_pred = mwm._predict_prefix(0, mwm_encoded["emb"][:, :3], actions[:, :3])
        predict_max_abs = _assert_close("prediction", raw_pred, mwm_pred)

        samples, horizon = 3, 6
        rollout_pixels = pixels[:, None, :3].expand(-1, samples, -1, -1, -1, -1).clone()
        candidates = torch.randn(batch_size, samples, horizon, action_dim, device=device)
        raw_rollout = raw.rollout({"pixels": rollout_pixels.clone()}, candidates.clone())["predicted_emb"]
        mwm_rollout = mwm.rollout_at_level({"pixels": rollout_pixels.clone()}, candidates.clone(), 0)[
            "predicted_emb"
        ]
        rollout_max_abs = _assert_close("rollout", raw_rollout, mwm_rollout)

        # Raw LeWM's goal broadcast only supports the official evaluation
        # contract batch_size=1. MWM also supports larger batches, but parity
        # here must compare the actual upstream-supported planner path.
        cost_pixels = rollout_pixels[:1].clone()
        cost_candidates = candidates[:1].clone()
        goal = torch.randn(1, samples, 1, 3, 224, 224, device=device)
        info = {"pixels": cost_pixels, "goal": goal, "action": cost_candidates.clone()}
        raw_cost = raw.get_cost(
            {key: value.clone() for key, value in info.items()},
            cost_candidates.clone(),
        )
        decision = SimpleNamespace(
            base_level_idx=0,
            rollout_level_indices=[0] * horizon,
            flop_accounting="none",
        )
        mwm_cost = mwm.get_cost_with_fidelity(
            {key: value.clone() for key, value in info.items()},
            cost_candidates.clone(),
            decision,
        )
        cost_max_abs = _assert_close("planning cost", raw_cost, mwm_cost)

    from stable_worldmodel.wm.loss import SIGReg

    raw.train()
    mwm.train()
    raw.zero_grad(set_to_none=True)
    mwm.zero_grad(set_to_none=True)
    raw_optimizer = torch.optim.AdamW(raw.parameters(), lr=5e-5, weight_decay=1e-3)
    world_parameters, _ = stable_wm_parameter_partitions(mwm)
    world_parameters = stable_wm_upstream_lewm_parameter_order(mwm, world_parameters)
    mwm_optimizer = torch.optim.AdamW(world_parameters, lr=5e-5, weight_decay=1e-3)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    rng_state = torch.cuda.get_rng_state()
    raw_output = raw.encode({"pixels": pixels.clone(), "action": actions.clone()})
    raw_prediction = raw.predict(raw_output["emb"][:, :3], raw_output["act_emb"][:, :3])
    raw_pred_loss = (raw_prediction - raw_output["emb"][:, 1:]).pow(2).mean()
    raw_sigreg_loss = sigreg(raw_output["emb"].transpose(0, 1))
    raw_loss = raw_pred_loss + 0.09 * raw_sigreg_loss
    raw_loss.backward()

    torch.cuda.set_rng_state(rng_state)
    mwm_output = mwm.training_loss(
        {"pixels": pixels.clone(), "action": actions.clone()},
        decoder_training_enabled=False,
        sigreg=sigreg,
        sigreg_weight=0.09,
    )
    mwm_loss = mwm_output["loss"]
    mwm_loss.backward()

    pred_loss_max_abs = _assert_close("prediction loss", raw_pred_loss, mwm_output["pred_loss"])
    sigreg_loss_max_abs = _assert_close("SIGReg loss", raw_sigreg_loss, mwm_output["sigreg_loss"])
    loss_max_abs = _assert_close("world loss", raw_loss, mwm_loss)
    grad_max_abs = 0.0
    for raw_name, raw_param in raw_params.items():
        mwm_param = mwm_params[_mapped_name(raw_name)]
        if raw_param.grad is None or mwm_param.grad is None:
            raise RuntimeError(f"Missing gradient for {raw_name}")
        grad_max_abs = max(grad_max_abs, _assert_close(f"gradient {raw_name}", raw_param.grad, mwm_param.grad))

    raw_grad_norm = _clip(raw.parameters(), 1.0)
    mwm_grad_norm = _clip(world_parameters, 1.0)
    raw_optimizer.step()
    mwm_optimizer.step()
    update_max_abs = 0.0
    for raw_name, raw_param in raw_params.items():
        update_max_abs = max(
            update_max_abs,
            _assert_close(f"updated parameter {raw_name}", raw_param, mwm_params[_mapped_name(raw_name)]),
        )

    # Repeat the optimizer-scoped training comparison under the BF16 autocast
    # mode used by the canonical Lightning runs. Reload both models so this is
    # an independent one-step comparison from the exact released weights.
    raw_bf16 = _load_raw(Path(args.raw_checkpoint), device)
    mwm_bf16, _, _ = load_world_model_from_checkpoint(Path(args.mwm_checkpoint), None, device)
    raw_bf16.train()
    mwm_bf16.train()
    raw_bf16.zero_grad(set_to_none=True)
    mwm_bf16.zero_grad(set_to_none=True)
    raw_bf16_params = dict(raw_bf16.named_parameters())
    mwm_bf16_params = _world_named_parameters(mwm_bf16)
    raw_bf16_optimizer = torch.optim.AdamW(raw_bf16.parameters(), lr=5e-5, weight_decay=1e-3)
    mwm_bf16_world_parameters, _ = stable_wm_parameter_partitions(mwm_bf16)
    mwm_bf16_optimizer = torch.optim.AdamW(
        mwm_bf16_world_parameters, lr=5e-5, weight_decay=1e-3
    )
    bf16_sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    bf16_rng_state = torch.cuda.get_rng_state()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw_bf16_output = raw_bf16.encode(
            {"pixels": pixels.clone(), "action": actions.clone()}
        )
        raw_bf16_prediction = raw_bf16.predict(
            raw_bf16_output["emb"][:, :3], raw_bf16_output["act_emb"][:, :3]
        )
        raw_bf16_pred_loss = (
            raw_bf16_prediction - raw_bf16_output["emb"][:, 1:]
        ).pow(2).mean()
        raw_bf16_sigreg_loss = bf16_sigreg(raw_bf16_output["emb"].transpose(0, 1))
        raw_bf16_loss = raw_bf16_pred_loss + 0.09 * raw_bf16_sigreg_loss
    raw_bf16_loss.backward()

    torch.cuda.set_rng_state(bf16_rng_state)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mwm_bf16_output = mwm_bf16.training_loss(
            {"pixels": pixels.clone(), "action": actions.clone()},
            decoder_training_enabled=False,
            sigreg=bf16_sigreg,
            sigreg_weight=0.09,
        )
        mwm_bf16_loss = mwm_bf16_output["loss"]
    mwm_bf16_loss.backward()

    bf16_pred_loss_max_abs = _assert_close(
        "BF16 prediction loss", raw_bf16_pred_loss, mwm_bf16_output["pred_loss"]
    )
    bf16_sigreg_loss_max_abs = _assert_close(
        "BF16 SIGReg loss", raw_bf16_sigreg_loss, mwm_bf16_output["sigreg_loss"]
    )
    bf16_loss_max_abs = _assert_close(
        "BF16 world loss", raw_bf16_loss, mwm_bf16_loss
    )
    bf16_grad_max_abs = 0.0
    for raw_name, raw_param in raw_bf16_params.items():
        mwm_param = mwm_bf16_params[_mapped_name(raw_name)]
        if raw_param.grad is None or mwm_param.grad is None:
            raise RuntimeError(f"Missing BF16 gradient for {raw_name}")
        bf16_grad_max_abs = max(
            bf16_grad_max_abs,
            _assert_close(f"BF16 gradient {raw_name}", raw_param.grad, mwm_param.grad),
        )

    # MWM's module tree groups the shared projector before transition modules,
    # whereas raw LeWM registers predictor/action_encoder before projector.
    # Preserve both measurements: the current module-order norm diagnoses the
    # BF16 reduction-order effect, and the mapped tuple tests the exact
    # upstream parameter order proposed for strict-parity clipping.
    mwm_bf16_upstream_order = stable_wm_upstream_lewm_parameter_order(
        mwm_bf16, mwm_bf16_world_parameters
    )
    expected_mwm_bf16_order = tuple(
        mwm_bf16_params[_mapped_name(raw_name)] for raw_name in raw_bf16_params
    )
    if tuple(map(id, mwm_bf16_upstream_order)) != tuple(map(id, expected_mwm_bf16_order)):
        raise RuntimeError("Production strict-parity helper did not reproduce raw LeWM order.")
    raw_bf16_norm_preview = torch.nn.utils.get_total_norm(
        [param.grad for param in raw_bf16_params.values() if param.grad is not None]
    )
    mwm_bf16_module_order_norm_preview = torch.nn.utils.get_total_norm(
        [param.grad for param in mwm_bf16_world_parameters if param.grad is not None]
    )
    mwm_bf16_upstream_order_norm_preview = torch.nn.utils.get_total_norm(
        [param.grad for param in mwm_bf16_upstream_order if param.grad is not None]
    )
    bf16_module_order_norm_delta = abs(
        float(raw_bf16_norm_preview.detach().float().cpu().item())
        - float(mwm_bf16_module_order_norm_preview.detach().float().cpu().item())
    )
    bf16_upstream_order_norm_delta = abs(
        float(raw_bf16_norm_preview.detach().float().cpu().item())
        - float(mwm_bf16_upstream_order_norm_preview.detach().float().cpu().item())
    )
    raw_bf16_grad_norm = _clip(raw_bf16.parameters(), 1.0)
    mwm_bf16_grad_norm = _clip(mwm_bf16_upstream_order, 1.0)
    raw_bf16_optimizer.step()
    mwm_bf16_optimizer.step()
    bf16_update_max_abs = 0.0
    for raw_name, raw_param in raw_bf16_params.items():
        bf16_update_max_abs = max(
            bf16_update_max_abs,
            _assert_close(
                f"BF16 updated parameter {raw_name}",
                raw_param,
                mwm_bf16_params[_mapped_name(raw_name)],
            ),
        )

    payload = {
        "status": "pass",
        "raw_checkpoint": str(Path(args.raw_checkpoint).resolve()),
        "mwm_checkpoint": str(Path(args.mwm_checkpoint).resolve()),
        "world_parameter_count": raw_count,
        "fresh_world_parameter_count": fresh_count,
        "upstream_optimizer_parameter_layout": {
            "tensor_count": len(raw_parameter_layout),
            "module_boundaries": raw_module_boundaries,
            "last_tensor": raw_parameter_layout[-1],
        },
        "max_abs": {
            "fresh_initialization": fresh_init_max_abs,
            "weights": weight_max_abs,
            "encode": encode_max_abs,
            "action_encode": action_encode_max_abs,
            "predict": predict_max_abs,
            "rollout": rollout_max_abs,
            "planning_cost": cost_max_abs,
            "pred_loss": pred_loss_max_abs,
            "sigreg_loss": sigreg_loss_max_abs,
            "world_loss": loss_max_abs,
            "world_gradients": grad_max_abs,
            "post_adamw_update": update_max_abs,
            "preclip_gradient_norm": abs(raw_grad_norm - mwm_grad_norm),
        },
        "current_runtime_bf16": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "autocast_dtype": "torch.bfloat16",
            "max_abs": {
                "pred_loss": bf16_pred_loss_max_abs,
                "sigreg_loss": bf16_sigreg_loss_max_abs,
                "world_loss": bf16_loss_max_abs,
                "world_gradients": bf16_grad_max_abs,
                "preclip_gradient_norm": abs(
                    raw_bf16_grad_norm - mwm_bf16_grad_norm
                ),
                "module_order_preclip_gradient_norm": bf16_module_order_norm_delta,
                "upstream_order_preclip_gradient_norm": bf16_upstream_order_norm_delta,
                "post_adamw_update": bf16_update_max_abs,
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
