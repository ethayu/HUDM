from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from mwm.adapters.builder import build_mwm_from_stable_config
from mwm.benchmark.config import merged_run_config
from mwm.checkpoint_io import load_world_model_from_checkpoint, save_world_checkpoint, validate_checkpoint_directory
from mwm.diagnostics.flops import profile_dynamics_call
from mwm.eval.execution import combine_mwm_diagnostics
from mwm.eval.policy_builder import build_mwm_policy
from mwm.eval.runtime import _load_eval_config
from mwm.eval.review_trace import fidelity_trace_from_planning_trace
from mwm.fidelity import FidelityScheduler
from mwm.planning.scheduled_cem import (
    MWMScheduledCEMSolver,
    parse_k_selection,
    resolve_model_k_selection,
)
from mwm.training.stable_wm_lightning import stable_wm_adapter_forward
from mwm.training.stable_wm_model import build_trainable_stable_wm_adapter_model


class FakeLeWMEncoder(nn.Module):
    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = False) -> SimpleNamespace:
        del interpolate_pos_encoding
        pooled = x.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled).unsqueeze(1))


def _lewm_source_config(
    *,
    D: int,
    action_dim: int,
    history_size: int = 2,
    predictor_depth: int = 1,
    predictor_heads: int = 2,
    predictor_dim_head: int = 4,
    predictor_mlp_dim: int = 16,
    predictor_dropout: float = 0.0,
    projector_hidden_dim: int = 16,
) -> dict:
    return {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {
            "_target_": "tests.test_mwm_shared_slimmable.FakeLeWMEncoder",
            "out_dim": int(D),
        },
        "predictor": {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": int(history_size),
            "input_dim": int(D),
            "hidden_dim": int(D),
            "output_dim": int(D),
            "depth": int(predictor_depth),
            "heads": int(predictor_heads),
            "mlp_dim": int(predictor_mlp_dim),
            "dim_head": int(predictor_dim_head),
            "dropout": float(predictor_dropout),
            "emb_dropout": 0.0,
        },
        "action_encoder": {
            "_target_": "stable_worldmodel.wm.lewm.module.Embedder",
            "input_dim": int(action_dim),
            "emb_dim": int(D),
        },
        "projector": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": int(D),
            "output_dim": int(D),
            "hidden_dim": int(projector_hidden_dim),
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
        "pred_proj": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": int(D),
            "output_dim": int(D),
            "hidden_dim": int(projector_hidden_dim),
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
    }


SHARED_POLICY = {
    "shared": ["latent_producer", "transition"],
    "per_level": [],
    "reconstructor": ["decoder"],
}
SHARED_DYNAMICS = {
    "architecture": "slimmable_transformer_v1",
    "min_k": 1,
    "prefix_sampling": {
        "mode": "discrete_log_uniform_non_anchor",
        "samples_per_batch": 1,
    },
}


def _build_shared(*, K: tuple[int, ...] = (2, 4), D: int = 4):
    return build_mwm_from_stable_config(
        family="lewm",
        source_config=_lewm_source_config(
            D=D,
            action_dim=2,
            history_size=2,
            predictor_heads=1,
            predictor_dim_head=2,
            predictor_mlp_dim=8,
            predictor_dropout=0.0,
            projector_hidden_dim=8,
        ),
        source_config_sha256="shared-slimmable-test-source",
        training_recipe={
            "history_size": 2,
            "num_preds": 1,
            "loss_scope": {"regularizers": "shared_latent"},
        },
        K=K,
        action_dim=2,
        expected_D=D,
        action_block=1,
        image_shape=(8, 8),
        normalize_imagenet=False,
        component_policy=SHARED_POLICY,
        shared_dynamics=SHARED_DYNAMICS,
    )


def _build_legacy(*, D: int = 4):
    return build_mwm_from_stable_config(
        family="lewm",
        source_config=_lewm_source_config(
            D=D,
            action_dim=2,
            history_size=2,
            predictor_heads=1,
            predictor_dim_head=2,
            predictor_mlp_dim=8,
            predictor_dropout=0.0,
            projector_hidden_dim=8,
        ),
        source_config_sha256="shared-slimmable-test-source",
        training_recipe={"history_size": 2, "num_preds": 1},
        K=(D,),
        action_dim=2,
        expected_D=D,
        action_block=1,
        image_shape=(8, 8),
        normalize_imagenet=False,
    )


def _copy_linear(dst, src) -> None:
    dst.weight.data.copy_(src.weight.data)
    if dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def _copy_norm(dst, src) -> None:
    if getattr(dst, "weight", None) is not None:
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)
    if getattr(dst, "running_mean", None) is not None:
        dst.running_mean.data.copy_(src.running_mean.data)
        dst.running_var.data.copy_(src.running_var.data)
        dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)


def _copy_full_width_transition(shared, legacy) -> None:
    src_action = legacy.action_encoder
    dst_action = shared.action_encoder
    dst_action.patch_embed.load_state_dict(src_action.patch_embed.state_dict())
    _copy_linear(dst_action.fc1, src_action.embed[0])
    _copy_linear(dst_action.fc2, src_action.embed[2])

    src_predictor = legacy.predictor
    dst_predictor = shared.predictor
    dst_predictor.pos_embedding.data.copy_(src_predictor.pos_embedding.data)
    for dst_block, src_block in zip(dst_predictor.layers, src_predictor.transformer.layers):
        _copy_norm(dst_block.attn.norm, src_block.attn.norm)
        dst_block.attn.to_qkv.weight.data.copy_(src_block.attn.to_qkv.weight.data)
        _copy_linear(dst_block.attn.to_out, src_block.attn.to_out[0])
        _copy_norm(dst_block.mlp.norm, src_block.mlp.net[0])
        _copy_linear(dst_block.mlp.fc1, src_block.mlp.net[1])
        _copy_linear(dst_block.mlp.fc2, src_block.mlp.net[4])
        dst_block.modulation.load_state_dict(src_block.adaLN_modulation[1].state_dict())
    _copy_norm(dst_predictor.norm, src_predictor.transformer.norm)

    src_proj = legacy.pred_proj.net
    dst_proj = shared.pred_proj
    _copy_linear(dst_proj.fc1, src_proj[0])
    _copy_norm(dst_proj.norm, src_proj[1])
    _copy_linear(dst_proj.fc2, src_proj[3])


class SharedSlimmableDynamicsTests(unittest.TestCase):
    def test_single_shared_state_tree_and_arbitrary_widths(self) -> None:
        model = _build_shared()
        self.assertTrue(model.supports_arbitrary_k)
        self.assertEqual(model.architecture_version, "lewm_shared_slimmable_transformer_v1")
        self.assertEqual(model.metadata["architecture_version"], "lewm_shared_slimmable_transformer_v1")
        self.assertEqual(len(model.transitions), 0)
        state_keys = list(model.state_dict())
        self.assertTrue(any(key.startswith("shared_transition.") for key in state_keys))
        self.assertFalse(any(key.startswith("transitions.") for key in state_keys))

        action = torch.randn(3, 2, 2)
        for k in (1, 2, 3, 4):
            latent = torch.randn(3, 2, k, requires_grad=True)
            prediction = model.predict_at_k(latent, action)
            self.assertEqual(tuple(prediction.shape), (3, 2, k))
            prediction.square().mean().backward()
            self.assertIsNotNone(latent.grad)

        model.zero_grad(set_to_none=True)
        model.predict_at_k(torch.randn(3, 2, 3), action).square().mean().backward()
        positional_grad = model.shared_transition.predictor.pos_embedding.grad
        self.assertEqual(float(positional_grad[..., 3:].abs().sum()), 0.0)

    def test_inactive_parameter_regions_receive_no_gradient(self) -> None:
        model = _build_shared()
        model.zero_grad(set_to_none=True)
        action = torch.randn(3, 2, 2)
        with torch.no_grad():
            bias = model.shared_transition.predictor.layers[0].modulation.bias.reshape(6, 4)
            bias[2].fill_(1.0)
            bias[5].fill_(1.0)
        model.predict_at_k(torch.randn(3, 2, 2), action, k=2).square().mean().backward()
        block = model.shared_transition.predictor.layers[0]

        qkv_grad = block.attn.to_qkv.weight.grad.reshape(3, 2, 4)
        self.assertGreater(float(qkv_grad[:, :1, :2].abs().sum()), 0.0)
        self.assertEqual(float(qkv_grad[:, 1:, :].abs().sum()), 0.0)
        self.assertEqual(float(qkv_grad[:, :, 2:].abs().sum()), 0.0)

        modulation_grad = block.modulation.weight.grad.reshape(6, 4, 4)
        self.assertGreater(float(modulation_grad[:, :2, :2].abs().sum()), 0.0)
        self.assertEqual(float(modulation_grad[:, 2:, :].abs().sum()), 0.0)
        self.assertEqual(float(modulation_grad[:, :, 2:].abs().sum()), 0.0)

        projector = model.shared_transition.pred_proj
        self.assertGreater(float(projector.fc1.weight.grad[:4, :2].abs().sum()), 0.0)
        self.assertEqual(float(projector.fc1.weight.grad[4:, :].abs().sum()), 0.0)
        self.assertEqual(float(projector.fc1.weight.grad[:, 2:].abs().sum()), 0.0)
        self.assertEqual(float(projector.fc2.weight.grad[2:, :].abs().sum()), 0.0)
        self.assertEqual(float(projector.fc2.weight.grad[:, 4:].abs().sum()), 0.0)

    def test_full_width_matches_source_transition_after_weight_copy(self) -> None:
        legacy = _build_legacy().eval().transitions[0]
        shared = _build_shared().eval().shared_transition
        _copy_full_width_transition(shared, legacy)
        latent = torch.randn(3, 2, 4)
        action = torch.randn(3, 2, 2)
        with torch.no_grad():
            expected = legacy.predict(latent, action)
            actual = shared.predict(latent, action)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_action_conditioning_and_temporal_attention_are_causal(self) -> None:
        model = _build_shared().eval()
        with torch.no_grad():
            for block in model.shared_transition.predictor.layers:
                block.modulation.weight.normal_(mean=0.0, std=0.2)
                block.modulation.bias.normal_(mean=0.0, std=0.2)
        latent = torch.randn(2, 2, 4)
        actions = torch.randn(2, 2, 2)
        changed_actions = actions.clone()
        changed_actions[:, 0] += 3.0
        with torch.no_grad():
            baseline = model.predict_at_k(latent, actions, k=4)
            conditioned = model.predict_at_k(latent, changed_actions, k=4)
        self.assertFalse(torch.allclose(baseline, conditioned))

        changed_future_latent = latent.clone()
        changed_future_actions = actions.clone()
        changed_future_latent[:, -1] += 50.0
        changed_future_actions[:, -1] -= 50.0
        with torch.no_grad():
            changed_future = model.predict_at_k(
                changed_future_latent,
                changed_future_actions,
                k=4,
            )
        self.assertTrue(torch.allclose(baseline[:, 0], changed_future[:, 0], atol=1e-6, rtol=1e-6))

    def test_random_non_anchor_training_loss_and_encoder_gradient(self) -> None:
        torch.manual_seed(7)
        model = _build_shared()
        batch = {
            "pixels": torch.rand(2, 3, 3, 8, 8),
            "action": torch.randn(2, 3, 2),
        }
        output = model.training_loss(
            batch,
            random_prefix_weight=1.0,
            sample_random_prefixes=True,
        )
        self.assertIn(int(output["sampled_k"]), {1, 3})
        self.assertIn("pred_loss_random", output)
        output["loss"].backward()
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)

        deterministic = model.training_loss(
            {"pixels": batch["pixels"].detach(), "action": batch["action"].detach()},
            random_prefix_weight=1.0,
            sample_random_prefixes=False,
        )
        self.assertNotIn("sampled_k", deterministic)
        self.assertNotIn("pred_loss_random", deterministic)

    def test_random_prefix_sampling_is_seeded_and_disabled_in_eval_mode(self) -> None:
        model = _build_shared()
        batch = {
            "pixels": torch.rand(2, 3, 3, 8, 8),
            "action": torch.randn(2, 3, 2),
        }

        def sample_sequence() -> list[int]:
            return [
                int(
                    model.training_loss(
                        {key: value.clone() for key, value in batch.items()},
                        random_prefix_weight=1.0,
                        sample_random_prefixes=True,
                    )["sampled_k"]
                )
                for _ in range(8)
            ]

        torch.manual_seed(123)
        first = sample_sequence()
        torch.manual_seed(123)
        second = sample_sequence()
        self.assertEqual(first, second)
        self.assertTrue(set(first).issubset({1, 3}))

        model.eval()
        output = model.training_loss(
            {key: value.clone() for key, value in batch.items()},
            random_prefix_weight=1.0,
            sample_random_prefixes=True,
        )
        self.assertNotIn("sampled_k", output)
        self.assertNotIn("pred_loss_random", output)

    def test_stable_pretraining_fit_stage_enables_random_prefix_loss(self) -> None:
        model = _build_shared()
        logged: dict[str, torch.Tensor] = {}
        module = SimpleNamespace(
            model=model,
            sigreg=None,
            stable_wm_adapter_cfg=OmegaConf.create(
                {
                    "loss": {
                        "rollout_weight": 1.0,
                        "recon_latent_weight": 0.0,
                        "sigreg_weight": 0.0,
                        "sigreg_scope": "shared_latent",
                        "random_prefix_weight": 1.0,
                    }
                }
            ),
            log_dict=lambda values, **kwargs: logged.update(values),
        )
        batch = {
            "pixels": torch.rand(2, 3, 3, 8, 8),
            "action": torch.randn(2, 3, 2),
        }
        output = stable_wm_adapter_forward(module, batch, "fit")
        self.assertIn("pred_loss_random", output)
        self.assertIn("sampled_k", output)
        self.assertIn("fit/pred_loss_random", logged)
        self.assertIn("fit/sampled_k", logged)

        deterministic = stable_wm_adapter_forward(
            module,
            {key: value.clone() for key, value in batch.items()},
            "validate",
        )
        self.assertNotIn("pred_loss_random", deterministic)
        self.assertNotIn("sampled_k", deterministic)

    def test_arbitrary_k_rollout_preserves_inactive_suffix_and_rejects_increase(self) -> None:
        model = _build_shared().eval()
        infos = {"pixels": torch.rand(1, 2, 2, 3, 8, 8)}
        actions = torch.randn(1, 2, 4, 2)
        output = model.rollout_with_k_schedule(infos, actions, [3, 3, 2, 2])
        predicted = output["predicted_emb"]
        self.assertEqual(tuple(predicted.shape), (1, 2, 5, 4))
        self.assertTrue(torch.equal(predicted[:, :, -1, 2:], predicted[:, :, -2, 2:]))
        with self.assertRaisesRegex(ValueError, "cannot increase K"):
            model.rollout_with_k_schedule(
                {"pixels": torch.rand(1, 2, 2, 3, 8, 8)},
                actions,
                [2, 2, 3, 3],
            )

    def test_non_anchor_terminal_prefix_cost_and_latent_work(self) -> None:
        model = _build_shared().eval()
        infos = {
            "pixels": torch.rand(1, 2, 2, 3, 8, 8),
            "goal": torch.rand(1, 2, 2, 3, 8, 8),
        }
        candidates = torch.randn(1, 2, 4, 2)
        decision = SimpleNamespace(
            base_level_idx=None,
            rollout_level_indices=[None, None, 0, 0],
            base_k=3,
            rollout_ks=[3, 3, 2, 2],
            metadata={"mpc_k": 4},
        )
        cost = model.get_cost_with_fidelity(infos, candidates, decision)
        self.assertEqual(tuple(cost.shape), (1, 2))
        self.assertTrue(torch.isfinite(cost).all())
        self.assertEqual(model._last_cost_diagnostics["base_level_idx"], None)
        self.assertEqual(model._last_cost_diagnostics["terminal_level_idx"], 0)
        self.assertEqual(model._last_cost_diagnostics["terminal_k"], 2)
        self.assertEqual(model._last_cost_diagnostics["latent_work"], 14)

    def test_profiled_dynamics_flops_increase_with_k(self) -> None:
        model = _build_shared().eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        action = torch.randn(4, 2, 2)
        counts = []
        for k in (1, 2, 4):
            _, count, error = profile_dynamics_call(
                lambda k=k: model.predict_at_k(torch.randn(4, 2, k), action, k=k),
                enabled=True,
            )
            if error is not None:
                self.skipTest(error)
            counts.append(count)
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts, sorted(set(counts)))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count)

    def test_literal_k_scheduler_supports_non_anchor_decisions(self) -> None:
        scheduler = FidelityScheduler.from_config(
            {
                "enabled": True,
                "fidelity_unit": "k",
                "mpc": {"mode": "fixed", "k": "finest"},
                "cem": {"mode": "linear", "start_k": "coarsest", "end_k": "finest"},
                "rollout": {"mode": "fixed", "k": "base"},
            },
            num_levels=2,
            horizon=4,
            levels=(2, 4),
            min_k=1,
            max_k=4,
            supports_arbitrary_k=True,
        )
        early = scheduler.decision(cem_iter=0, n_iter=4)
        middle = scheduler.decision(cem_iter=2, n_iter=4)
        late = scheduler.decision(cem_iter=3, n_iter=4)
        self.assertEqual(early.base_k, 1)
        self.assertIsNone(early.base_level_idx)
        self.assertEqual(early.rollout_ks, [1, 1, 1, 1])
        self.assertEqual(middle.base_k, 3)
        self.assertIsNone(middle.base_level_idx)
        self.assertEqual(middle.rollout_ks, [3, 3, 3, 3])
        self.assertEqual(late.base_k, 4)
        self.assertEqual(late.base_level_idx, 1)
        with self.assertRaisesRegex(ValueError, "arbitrary-K"):
            FidelityScheduler.from_config(
                {"fidelity_unit": "k"},
                num_levels=2,
                horizon=4,
                levels=(2, 4),
                max_k=4,
                supports_arbitrary_k=False,
            )

    def test_inference_k_range_is_inclusive_and_requires_shared_slimmable_model(self) -> None:
        self.assertEqual(parse_k_selection(["2-4"]), ([2, 3, 4], True))
        self.assertEqual(parse_k_selection([2, 4]), ([2, 4], False))
        self.assertEqual(resolve_model_k_selection(_build_shared(), ["2-4"]), ([2, 3, 4], True))
        self.assertEqual(resolve_model_k_selection(_build_shared(), [1, 3, 4]), ([1, 3, 4], False))
        with self.assertRaisesRegex(ValueError, "range syntax requires architecture_version"):
            resolve_model_k_selection(_build_legacy(), ["2-4"])
        with self.assertRaisesRegex(ValueError, "match checkpoint anchors exactly"):
            resolve_model_k_selection(_build_legacy(), [2, 4])

    def test_k_range_shortcut_converts_old_level_scheduler_and_uses_each_width(self) -> None:
        model = _build_shared().eval()
        solver = MWMScheduledCEMSolver(
            model,
            scheduler={
                "K": ["2-4"],
                "enabled": True,
                "mpc": {"mode": "fixed", "level": "finest"},
                "cem": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
        )
        solver.configure(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=SimpleNamespace(horizon=2, action_block=1),
        )
        self.assertEqual(solver.scheduler.fidelity_unit, "k")
        self.assertEqual(solver.scheduler.selectable_ks, [2, 3, 4])
        self.assertEqual(
            [solver.scheduler.decision(cem_iter=i, n_iter=3).base_k for i in range(3)],
            [2, 3, 4],
        )

    def test_explicit_k_list_remains_discrete_for_shared_model(self) -> None:
        model = _build_shared().eval()
        solver = MWMScheduledCEMSolver(
            model,
            scheduler={
                "K": [2, 4],
                "enabled": True,
                "mpc": {"mode": "fixed", "level": "finest"},
                "cem": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
        )
        solver.configure(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=SimpleNamespace(horizon=2, action_block=1),
        )
        self.assertEqual(
            [solver.scheduler.decision(cem_iter=i, n_iter=2).base_k for i in range(2)],
            [2, 4],
        )

    def test_legacy_model_accepts_its_old_explicit_k_list_but_rejects_range(self) -> None:
        model = _build_legacy().eval()
        explicit = MWMScheduledCEMSolver(
            model,
            scheduler={
                "K": [4],
                "mpc": {"mode": "fixed", "level": "finest"},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
        )
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        config = SimpleNamespace(horizon=2, action_block=1)
        explicit.configure(action_space=action_space, n_envs=1, config=config)
        self.assertEqual(explicit.scheduler.fidelity_unit, "level")
        ranged = MWMScheduledCEMSolver(model, scheduler={"K": ["2-4"]})
        with self.assertRaisesRegex(ValueError, "range syntax requires architecture_version"):
            ranged.configure(action_space=action_space, n_envs=1, config=config)

    def test_literal_k_scheduled_cem_smoke_records_direct_widths(self) -> None:
        model = _build_shared().eval()
        solver = MWMScheduledCEMSolver(
            model,
            num_samples=3,
            n_steps=2,
            topk=1,
            std_unbiased=False,
            scheduler={
                "enabled": True,
                "fidelity_unit": "k",
                "mpc": {"mode": "fixed", "k": "finest"},
                "cem": {"mode": "linear", "start_k": "coarsest", "end_k": "finest"},
                "rollout": {"mode": "fixed", "k": "base"},
            },
        )
        solver.configure(
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=SimpleNamespace(horizon=4, action_block=1),
        )
        result = solver.solve(
            {
                "pixels": torch.rand(1, 2, 3, 8, 8),
                "goal": torch.rand(1, 2, 3, 8, 8),
            }
        )
        self.assertEqual(tuple(result["actions"].shape), (1, 4, 2))
        trace = result["mwm_diagnostics"]
        self.assertEqual([row["base_k"] for row in trace], [1, 4])
        self.assertEqual([row["base_level_idx"] for row in trace], [None, 1])
        self.assertEqual(trace[0]["rollout_ks"], [1, 1, 1, 1])

    def test_shared_checkpoint_round_trip(self) -> None:
        model = _build_shared()
        with tempfile.TemporaryDirectory() as tmp:
            save_world_checkpoint(model, tmp, metadata={"env_id": "swm/PushT-v1"})
            config, metadata = validate_checkpoint_directory(tmp)
            self.assertEqual(config["kwargs"]["shared_dynamics"], SHARED_DYNAMICS)
            self.assertEqual(metadata["supported_k"], {"min": 1, "max": 4, "arbitrary": True})
            loaded, loaded_metadata, _ = load_world_model_from_checkpoint(tmp, None, torch.device("cpu"))
            self.assertTrue(loaded.supports_arbitrary_k)
            self.assertEqual(loaded_metadata["architecture_version"], "lewm_shared_slimmable_transformer_v1")

    def test_absent_shared_builder_option_keeps_legacy_transition_implementation(self) -> None:
        model = build_mwm_from_stable_config(
            family="lewm",
            source_config=_lewm_source_config(D=4, action_dim=2),
            source_config_sha256="legacy-explicit-switch",
            training_recipe={"history_size": 2, "num_preds": 1, "shared_dynamics": SHARED_DYNAMICS},
            K=(2, 4),
            action_dim=2,
            expected_D=4,
            action_block=1,
            image_shape=(8, 8),
            normalize_imagenet=False,
        )
        self.assertFalse(model.supports_arbitrary_k)
        self.assertEqual(len(model.transitions), 2)
        self.assertNotIn("shared_dynamics", model.mwm_config["kwargs"])

    def test_training_config_builder_selects_shared_dynamics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps(_lewm_source_config(D=4, action_dim=2)),
                encoding="utf-8",
            )
            cfg = OmegaConf.create(
                {
                    "base": {"family": "lewm", "checkpoint": tmp},
                    "mwm": {
                        "component_policy": SHARED_POLICY,
                        "shared_dynamics": SHARED_DYNAMICS,
                        "loss_terms": {"regularizers": "shared_latent"},
                    },
                    "loss": {"rollout_weight": 1.0, "random_prefix_weight": 1.0},
                    "model": {"history_size": 2, "num_preds": 1},
                }
            )
            model = build_trainable_stable_wm_adapter_model(
                cfg,
                {
                    "D": 4,
                    "K": (2, 4),
                    "action_dim": 2,
                    "action_block": 1,
                    "image_shape": (8, 8),
                    "normalize_imagenet": False,
                },
            )
        self.assertTrue(model.supports_arbitrary_k)
        self.assertEqual(model.mwm_config["kwargs"]["shared_dynamics"], SHARED_DYNAMICS)

    def test_benchmark_merge_replaces_level_scheduler_with_literal_k_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(root / "configs/research/shared_slimmable_pusht_benchmark.yaml")
        run = next(item for item in cfg.runs if item.name == "shared_scheduled_k")
        _, merged = merged_run_config(cfg, run)
        scheduler = OmegaConf.to_container(merged.planner.scheduler, resolve=True)
        self.assertEqual(set(scheduler), {"enabled", "fidelity_unit", "mpc", "cem", "rollout"})
        self.assertEqual(scheduler["fidelity_unit"], "k")
        self.assertNotIn("level", scheduler["mpc"])
        self.assertEqual(merged.planner.flop_accounting, "dynamics_audit")

    def test_eval_config_loader_does_not_reinsert_level_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "eval.yaml")
            path.write_text(
                OmegaConf.to_yaml(
                    {
                        "planner": {
                            "scheduler": {
                                "enabled": True,
                                "fidelity_unit": "k",
                                "mpc": {"mode": "fixed", "k": 3},
                                "cem": {"mode": "fixed", "k": "base"},
                                "rollout": {"mode": "fixed", "k": "base"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            cfg = _load_eval_config(str(path))
        scheduler = OmegaConf.to_container(cfg.planner.scheduler, resolve=True)
        self.assertEqual(scheduler["fidelity_unit"], "k")
        self.assertEqual(scheduler["mpc"], {"mode": "fixed", "k": 3})
        self.assertFalse(any("level" in stage for stage in (scheduler["mpc"], scheduler["cem"], scheduler["rollout"])))

    def test_eval_root_k_range_is_forwarded_as_planner_shortcut(self) -> None:
        cfg = OmegaConf.create(
            {
                "K": ["2-4"],
                "eval": {"num_envs": 1, "budget": 2, "seed": 0},
                "planner": {
                    "horizon": 2,
                    "receding_horizon": 1,
                    "action_block": 1,
                    "batch_size": "auto",
                    "pop_size": 2,
                    "topk": 1,
                    "elite_frac": 0.5,
                    "n_iter": 2,
                    "init_std": 1.0,
                    "seed": 0,
                    "warm_start": False,
                    "clamp_actions": False,
                    "std_unbiased": False,
                    "scheduler": {
                        "enabled": True,
                        "mpc": {"mode": "fixed", "level": "finest"},
                        "cem": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    },
                },
            }
        )
        policy = build_mwm_policy(
            _build_shared().eval(),
            {"action_block": 1},
            cfg,
            torch.device("cpu"),
            gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            process={},
        )
        self.assertEqual(policy.solver.scheduler_spec["K"], ["2-4"])

    def test_literal_k_diagnostics_and_review_trace_preserve_non_anchor_k(self) -> None:
        decision = {
            "batch_start": 0,
            "batch_end": 1,
            "cem_iter": 0,
            "mpc_iter": 0,
            "base_level_idx": None,
            "mpc_level_idx": 1,
            "rollout_level_indices": [None, 0],
            "base_k": 3,
            "mpc_k": 4,
            "rollout_ks": [3, 2],
        }
        trace = fidelity_trace_from_planning_trace(
            planning_trace=[decision],
            batch_env=0,
            eval_budget=2,
            action_block=1,
            replan_interval=2,
            k_values=[2, 4],
        )
        self.assertEqual([row["K"] for row in trace], [3, 2])
        self.assertEqual([row["level_idx"] for row in trace], [None, 0])
        combined = combine_mwm_diagnostics(
            [
                {
                    "planning_diagnostics": {
                        "summary": {},
                        "trace": [decision],
                    }
                }
            ]
        )
        self.assertEqual(combined["schedule_k_counts"], {"3": 1})


if __name__ == "__main__":
    unittest.main()
