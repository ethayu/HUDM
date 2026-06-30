from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver

from mwm.eval.policy_builder import build_mwm_policy
from mwm.eval.action_preprocessing import available_stat_keys_for_action_process, uses_standardized_action_space
from mwm.adapters.builder import build_mwm_from_stable_config
from mwm.checkpoint_io import save_world_checkpoint
from mwm.data.transforms import MWMTrainSampleTransform, ZScoreScaler
from mwm.eval.policy import MWMWorldModelPolicy
from mwm.fidelity import FidelityScheduler
from mwm.models.common import MatryoshkaRuntimeModel
from mwm.models.lewm import LeWMMatryoshkaWorldModel
from mwm.models.decoders import ConvImageDecoder
from mwm.models.losses import latent_regularizer_loss, matryoshka_base_loss, weighted_level_mean
from mwm.models.transitions import TransitionPackage
from mwm.planning.scheduled_cem import MWMScheduledCEMSolver
from mwm.training.stable_wm import main as train_mwm_main
from mwm.training.stable_wm_callbacks import (
    AllLevelPlateauEarlyStopping,
    stable_wm_adapter_checkpoint_callback,
    select_stable_wm_adapter_export_checkpoint,
)
from mwm.training.stable_wm_export import load_stable_wm_adapter_lightning_state
from mwm.training.stable_wm_model import build_trainable_stable_wm_adapter_model
from mwm.training.stable_wm_runtime import (
    prepare_trainer_root,
    resolve_stable_wm_adapter_total_steps,
    resolve_lightning_trainer_runtime,
)


class FakeLeWMEncoder(nn.Module):
    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = False) -> SimpleNamespace:
        del interpolate_pos_encoding
        pooled = x.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled).unsqueeze(1))


class ShapeAgnosticLeWMEncoder(nn.Module):
    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = False) -> SimpleNamespace:
        del interpolate_pos_encoding
        pooled = x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=(pooled * self.scale).unsqueeze(1))


class FakeLeWMActionEncoder(nn.Module):
    def __init__(self, action_dim: int = 2, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, out_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.proj(action)


class FakeLeWMPredictor(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        return z + action_emb


class FakeStepDynamics(nn.Module):
    def __init__(self, action_dim: int = 2, out_dim: int = 4) -> None:
        super().__init__()
        self.action_encoder = FakeLeWMActionEncoder(action_dim=action_dim, out_dim=out_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + self.action_encoder(action)


class FakeActionScaler:
    def inverse_transform(self, action: np.ndarray) -> np.ndarray:
        return action * np.array([2.0, 3.0], dtype=np.float32) + np.array([10.0, 20.0], dtype=np.float32)

    def transform(self, action: np.ndarray) -> np.ndarray:
        return (action - np.array([10.0, 20.0], dtype=np.float32)) / np.array([2.0, 3.0], dtype=np.float32)


class CountingRegularizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(value.shape))
        return value.square().mean() * 0.0


def _grad_abs_sum(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().item())
    return total


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
        "encoder": {"_target_": "tests.test_mwm_core.FakeLeWMEncoder", "out_dim": int(D)},
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


def _stable_vit_lewm_source_config(model_cfg: dict) -> dict:
    d = int(model_cfg["D"])
    return {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {
            "_target_": "stable_pretraining.backbone.utils.vit_hf",
            "size": str(model_cfg.get("vit_size", "tiny")),
            "patch_size": int(model_cfg.get("vit_patch_size", 14)),
            "image_size": int(model_cfg.get("vit_image_size", 224)),
            "pretrained": bool(model_cfg.get("vit_pretrained", False)),
            "use_mask_token": bool(model_cfg.get("vit_use_mask_token", False)),
        },
        "predictor": {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": int(model_cfg.get("history_size", 3)),
            "input_dim": d,
            "hidden_dim": d,
            "output_dim": d,
            "depth": int(model_cfg.get("predictor_depth", 6)),
            "heads": int(model_cfg.get("predictor_heads", 16)),
            "mlp_dim": int(model_cfg.get("predictor_mlp_dim", 2048)),
            "dim_head": int(model_cfg.get("predictor_dim_head", 64)),
            "dropout": float(model_cfg.get("predictor_dropout", 0.1)),
            "emb_dropout": float(model_cfg.get("predictor_emb_dropout", 0.0)),
        },
        "action_encoder": {
            "_target_": "stable_worldmodel.wm.lewm.module.Embedder",
            "input_dim": int(model_cfg["action_dim"]),
            "emb_dim": d,
        },
        "projector": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": d,
            "output_dim": d,
            "hidden_dim": int(model_cfg.get("projector_hidden_dim", 2048)),
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
        "pred_proj": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": d,
            "output_dim": d,
            "hidden_dim": int(model_cfg.get("projector_hidden_dim", 2048)),
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_direct_lewm_reference(model_cfg: dict, cfg: Any) -> nn.Module:
    from stable_pretraining.backbone.utils import vit_hf
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

    d = int(model_cfg["D"])
    history_size = int(cfg.model.get("history_size", cfg.loss.get("history_size", 3)))
    encoder = vit_hf(
        size=str(model_cfg.get("vit_size", "tiny")),
        patch_size=int(model_cfg.get("vit_patch_size", 14)),
        image_size=int(model_cfg.get("vit_image_size", 224)),
        pretrained=bool(model_cfg.get("vit_pretrained", False)),
        use_mask_token=bool(model_cfg.get("vit_use_mask_token", False)),
    )
    return LeWM(
        encoder=encoder,
        predictor=Predictor(
            num_frames=history_size,
            input_dim=d,
            hidden_dim=d,
            output_dim=d,
            depth=int(model_cfg.get("predictor_depth", 6)),
            heads=int(model_cfg.get("predictor_heads", 16)),
            mlp_dim=int(model_cfg.get("predictor_mlp_dim", 2048)),
            dim_head=int(model_cfg.get("predictor_dim_head", 64)),
            dropout=float(model_cfg.get("predictor_dropout", 0.1)),
            emb_dropout=float(model_cfg.get("predictor_emb_dropout", 0.0)),
        ),
        action_encoder=Embedder(input_dim=int(model_cfg["action_dim"]), emb_dim=d),
        projector=MLP(
            input_dim=d,
            output_dim=d,
            hidden_dim=int(cfg.model.get("projector_hidden_dim", 2048)),
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=d,
            output_dim=d,
            hidden_dim=int(cfg.model.get("projector_hidden_dim", 2048)),
            norm_fn=nn.BatchNorm1d,
        ),
    )


def _lewm_matryoshka_model(
    *,
    K: tuple[int, ...] | list[int],
    D: int = 8,
    action_dim: int = 2,
    action_block: int = 1,
    image_shape: tuple[int, int] = (8, 8),
    normalize_imagenet: bool = False,
    history_size: int = 2,
    num_preds: int = 1,
    predictor_depth: int = 1,
    predictor_heads: int = 2,
    predictor_dim_head: int = 4,
    predictor_mlp_dim: int = 16,
    predictor_dropout: float = 0.0,
    projector_hidden_dim: int = 16,
) -> LeWMMatryoshkaWorldModel:
    return build_mwm_from_stable_config(
        family="lewm",
        source_config=_lewm_source_config(
            D=int(D),
            action_dim=int(action_dim),
            history_size=int(history_size),
            predictor_depth=int(predictor_depth),
            predictor_heads=int(predictor_heads),
            predictor_dim_head=int(predictor_dim_head),
            predictor_mlp_dim=int(predictor_mlp_dim),
            predictor_dropout=float(predictor_dropout),
            projector_hidden_dim=int(projector_hidden_dim),
        ),
        source_config_sha256="test-source-config",
        training_recipe={
            "history_size": int(history_size),
            "num_preds": int(num_preds),
            "loss_scope": {"regularizers": "shared_latent"},
        },
        K=tuple(int(k) for k in K),
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
    )


class FakeSolver:
    def __init__(self) -> None:
        self.solve_history = []
        self._n_envs = 0
        self._horizon = 1
        self._action_dim = 2

    def configure(self, **kwargs) -> None:
        self.configured = kwargs
        self._n_envs = int(kwargs["n_envs"])
        self._horizon = int(kwargs["config"].horizon)
        self._action_dim = 2

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return self._horizon

    def __call__(self, info_dict, init_action=None):
        return self.solve(info_dict, init_action=init_action)

    def solve(self, info_dict, init_action=None):
        del init_action
        batch = len(next(iter(info_dict.values())))
        actions = torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]], dtype=torch.float32)[:batch]
        self.solve_history.append({"solve_time_sec": 0.0, "mwm_diagnostics": []})
        return {"actions": actions, "costs": [0.0] * batch}

    def reset_history(self) -> None:
        self.solve_history = []


class FakeVectorEnv:
    num_envs = 2
    single_action_space = Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)
    action_space = Box(low=-100.0, high=100.0, shape=(2, 2), dtype=np.float32)


class FakeFidelityCostModel(nn.Module):
    K = [4]
    num_levels = 1

    def get_cost_with_fidelity(self, info_dict: dict, action_candidates: torch.Tensor, decision) -> torch.Tensor:
        del info_dict, decision
        return action_candidates.square().sum(dim=(2, 3))


class FakeCEMParityCostModel(nn.Module):
    K = [4]
    num_levels = 1

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        target = info_dict["target"]
        return (action_candidates - target).square().sum(dim=(2, 3))

    def get_cost_with_fidelity(self, info_dict: dict, action_candidates: torch.Tensor, decision) -> torch.Tensor:
        del decision
        return self.get_cost(info_dict, action_candidates)


class RecordingFidelityCostModel(nn.Module):
    K = [2, 4]
    num_levels = 2

    def __init__(self) -> None:
        super().__init__()
        self.decisions = []
        self._last_cost_diagnostics = {}

    def get_cost_with_fidelity(self, info_dict: dict, action_candidates: torch.Tensor, decision) -> torch.Tensor:
        del info_dict
        self.decisions.append(decision)
        flop_mode = dict(getattr(decision, "metadata", {})).get("flop_accounting", "none")
        self._last_cost_diagnostics = {
            "base_level_idx": int(decision.base_level_idx),
            "terminal_level_idx": int(decision.metadata.get("terminal_level_idx", decision.rollout_level_indices[-1])),
            "latent_work": int(action_candidates.shape[0] * action_candidates.shape[1]),
            "dynamics_flops": 123 if flop_mode == "dynamics_audit" else 0,
            "flop_accounting": flop_mode,
        }
        return action_candidates.square().sum(dim=(2, 3))


class MWMCoreTests(unittest.TestCase):
    def test_core_module_has_no_separate_runtime_model_class(self) -> None:
        import mwm.models.core as core

        self.assertEqual(core.__all__, [])
        self.assertFalse(hasattr(core, "MWMWorldModel"))

    def test_matryoshka_world_model_is_direct_nn_module_runtime(self) -> None:
        self.assertEqual(LeWMMatryoshkaWorldModel.__bases__, (MatryoshkaRuntimeModel,))

    def test_world_model_provides_matryoshka_loss_and_regularizer_routing(self) -> None:
        losses = [torch.tensor(2.0), torch.tensor(6.0)]

        total, logs = weighted_level_mean(losses, level_weights=[1.0, 3.0], log_prefix="pred_loss")

        self.assertTrue(torch.equal(total, torch.tensor(5.0)))
        self.assertEqual(set(logs), {"pred_loss_l0", "pred_loss_l1"})

        latents = torch.ones(2, 3, 8)
        shared_reg = CountingRegularizer()
        total_reg, reg_logs = latent_regularizer_loss(
            latents,
            K=[4, 8],
            regularizer=shared_reg,
            scope="shared_latent",
            level_weights=[1.0, 3.0],
            log_prefix="sigreg_loss",
        )

        self.assertTrue(torch.equal(total_reg, torch.tensor(0.0)))
        self.assertEqual(shared_reg.calls, 1)
        self.assertEqual(shared_reg.shapes, [(3, 2, 8)])
        self.assertEqual(set(reg_logs), {"sigreg_loss"})

        per_level_reg = CountingRegularizer()
        _, per_level_logs = latent_regularizer_loss(
            latents,
            K=[4, 8],
            regularizer=per_level_reg,
            scope="per_level_prefix",
            level_weights=[1.0, 3.0],
            log_prefix="sigreg_loss",
        )

        self.assertEqual(per_level_reg.calls, 2)
        self.assertEqual([shape[-1] for shape in per_level_reg.shapes], [4, 8])
        self.assertEqual(set(per_level_logs), {"sigreg_loss", "sigreg_loss_l0", "sigreg_loss_l1"})

    def test_world_model_builds_base_loss_from_adapter_level_terms(self) -> None:
        latents = torch.ones(2, 3, 8)
        reg = CountingRegularizer()

        logs = matryoshka_base_loss(
            [torch.tensor(2.0), torch.tensor(6.0)],
            latents=latents,
            K=[4, 8],
            level_weights=[1.0, 3.0],
            primary_log_prefix="pred_loss",
            primary_aliases=("pred_loss", "rollout_loss"),
            rollout_weight=2.0,
            regularizer=reg,
            regularizer_weight=0.5,
            regularizer_scope="shared_latent",
        )

        self.assertTrue(torch.equal(logs["pred_loss"], torch.tensor(5.0)))
        self.assertTrue(torch.equal(logs["rollout_loss"], torch.tensor(5.0)))
        self.assertTrue(torch.equal(logs["loss"], torch.tensor(10.0)))
        self.assertEqual(reg.calls, 1)
        self.assertEqual(set(logs), {"loss", "pred_loss", "pred_loss_l0", "pred_loss_l1", "rollout_loss", "sigreg_loss"})

    def test_canonical_checkpoint_export_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra_dir = root / "checkpoint"
            extra_dir.mkdir()
            (extra_dir / "notes.txt").write_text("not canonical", encoding="utf-8")
            model = _lewm_matryoshka_model(K=(4,), D=4, action_dim=2)
            with self.assertRaisesRegex(ValueError, "non-checkpoint files"):
                save_world_checkpoint(model, extra_dir, metadata={"env_id": "swm/PushT-v1"})

    def test_single_fidelity_k_equals_d_uses_adapter_owned_lewm_loss(self) -> None:
        model = _lewm_matryoshka_model(K=(8,), D=8, action_dim=2)
        self.assertIsInstance(model, LeWMMatryoshkaWorldModel)
        self.assertEqual(model.K, [model.D])
        self.assertEqual(len(model.decoders), 1)
        self.assertEqual(len(model.transitions), 1)
        self.assertEqual(model.metadata["action_spec"]["dim"], 2)
        self.assertEqual(model.metadata["preprocessing_spec"]["image"], "identity")
        self.assertEqual(model.metadata["architecture_version"], "lewm_base_adapter_v1")

        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}
        out = model.training_loss(batch)
        self.assertIn("pred_loss_l0", out)
        self.assertIn("recon_loss_l0", out)
        self.assertEqual(tuple(model.decode(0, torch.randn(2, 8)).shape), (2, 3, 8, 8))
        self.assertIn("loss", out)

    def test_lewm_matryoshka_head_scaling_and_k_may_omit_d(self) -> None:
        model = _lewm_matryoshka_model(
            K=(4, 8),
            D=8,
            action_dim=10,
            action_block=5,
            predictor_heads=2,
            predictor_dim_head=4,
            predictor_mlp_dim=16,
            projector_hidden_dim=16,
        )

        self.assertEqual(model.K, [4, 8])
        self.assertEqual(len(model.transitions), 2)
        self.assertEqual([h["predictor_heads"] for h in model.metadata["head_architectures"]], [1, 2])
        self.assertEqual([h["predictor_dim_head"] for h in model.metadata["head_architectures"]], [2, 4])
        self.assertEqual([h["predictor_mlp_dim"] for h in model.metadata["head_architectures"]], [8, 16])

        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 10)}
        out = model.training_loss(batch)
        self.assertIn("pred_loss_l0", out)
        self.assertIn("pred_loss_l1", out)
        self.assertIn("recon_loss_l0", out)
        self.assertIn("recon_loss_l1", out)
        self.assertNotIn("pred_loss_l2", out)

    def test_lewm_sigreg_is_shared_once_by_default(self) -> None:
        model = _lewm_matryoshka_model(K=(4, 8), D=8, action_dim=2)
        reg = CountingRegularizer()
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        out = model.training_loss(batch, sigreg=reg, sigreg_weight=0.5, sigreg_scope="shared_latent")

        self.assertIn("sigreg_loss", out)
        self.assertEqual(reg.calls, 1)
        self.assertEqual(reg.shapes[0][-1], 8)

    def test_lewm_sigreg_can_be_per_level_when_explicit(self) -> None:
        model = _lewm_matryoshka_model(K=(4, 8), D=8, action_dim=2)
        reg = CountingRegularizer()
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        model.training_loss(batch, sigreg=reg, sigreg_weight=0.5, sigreg_scope="per_level_prefix")

        self.assertEqual(reg.calls, 2)
        self.assertEqual([shape[-1] for shape in reg.shapes], [4, 8])

    def test_reconstruction_trains_decoders_without_latent_gradients_by_default(self) -> None:
        model = _lewm_matryoshka_model(K=(4, 8), D=8, action_dim=2)
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        out = model.training_loss(batch, rollout_weight=0.0, recon_latent_weight=0.0)
        model.zero_grad(set_to_none=True)
        out["loss"].backward()

        self.assertGreater(_grad_abs_sum(model.decoders), 0.0)
        self.assertEqual(_grad_abs_sum(model.encoder), 0.0)
        self.assertEqual(_grad_abs_sum(model.projector), 0.0)
        self.assertIn("recon_loss", out)
        self.assertNotIn("recon_latent_loss", out)

    def test_reconstruction_latent_weight_shapes_encoder_latents(self) -> None:
        model = _lewm_matryoshka_model(K=(4, 8), D=8, action_dim=2)
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        out = model.training_loss(batch, rollout_weight=0.0, recon_latent_weight=0.25)
        model.zero_grad(set_to_none=True)
        out["loss"].backward()

        self.assertGreater(_grad_abs_sum(model.decoders), 0.0)
        self.assertGreater(_grad_abs_sum(model.encoder), 0.0)
        self.assertGreater(_grad_abs_sum(model.projector), 0.0)
        self.assertIn("recon_latent_loss", out)

    def test_train_entrypoint_builds_from_stable_wm_base_config(self) -> None:
        source_config = _lewm_source_config(D=4, action_dim=2, predictor_heads=1, predictor_dim_head=2, predictor_mlp_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            (checkpoint_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
            cfg = OmegaConf.create(
                {
                    "base": {"family": "lewm", "checkpoint": str(checkpoint_dir)},
                    "mwm": {
                        "component_policy": {
                            "shared": ["latent_producer"],
                            "per_level": ["transition"],
                            "reconstructor": ["decoder"],
                        },
                        "loss_terms": {"regularizers": "shared_latent"},
                    },
                    "model": {"history_size": 2, "num_preds": 1},
                    "loss": {"sigreg_weight": 0.09},
                }
            )
            model_cfg = {
                "D": 4,
                "K": (4,),
                "action_dim": 2,
                "action_block": 1,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
            }

            model = build_trainable_stable_wm_adapter_model(cfg, model_cfg)

            self.assertIsInstance(model, LeWMMatryoshkaWorldModel)
            self.assertEqual(model.metadata["adapter_family"], "lewm")
            self.assertTrue(model.metadata["fresh_init"])
            self.assertEqual(model.metadata["component_policy"]["shared"], ["latent_producer"])
            self.assertEqual(model.metadata["loss_scope"]["regularizers"], "shared_latent")
            self.assertEqual(model.mwm_config["target"], "mwm.adapters.builder.build_mwm_from_stable_config")

    def test_train_entrypoint_rejects_configured_d_mismatch_with_base_config(self) -> None:
        source_config = _lewm_source_config(D=4, action_dim=2, predictor_heads=1, predictor_dim_head=2, predictor_mlp_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            (checkpoint_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
            cfg = OmegaConf.create(
                {
                    "base": {"family": "lewm", "checkpoint": str(checkpoint_dir)},
                    "mwm": {
                        "component_policy": {
                            "shared": ["latent_producer"],
                            "per_level": ["transition"],
                            "reconstructor": ["decoder"],
                        }
                    },
                    "model": {"history_size": 2, "num_preds": 1},
                    "loss": {},
                }
            )
            model_cfg = {
                "D": 8,
                "K": (4,),
                "action_dim": 2,
                "action_block": 1,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
            }

            with self.assertRaisesRegex(ValueError, "configured D=8.*base latent dimension D=4"):
                build_trainable_stable_wm_adapter_model(cfg, model_cfg)

    def test_k_equals_d_lewm_init_forward_grad_and_step_match_direct_backend(self) -> None:
        cfg = OmegaConf.create(
            {
                "model": {
                    "history_size": 2,
                    "num_preds": 1,
                    "projector_hidden_dim": 64,
                },
                "loss": {"history_size": 2, "num_preds": 1},
            }
        )
        model_cfg = {
            "encoder": "stable_vit",
            "D": 192,
            "K": (192,),
            "action_dim": 4,
            "action_block": 2,
            "image_shape": (224, 224),
            "freeze_encoder": False,
            "normalize_imagenet": True,
            "vit_size": "tiny",
            "vit_patch_size": 14,
            "vit_image_size": 224,
            "vit_pretrained": False,
            "vit_use_mask_token": False,
            "history_size": 2,
            "num_preds": 1,
            "predictor_depth": 1,
            "predictor_heads": 2,
            "predictor_dim_head": 8,
            "predictor_mlp_dim": 64,
            "predictor_dropout": 0.0,
            "projector_hidden_dim": 64,
        }

        # Keep the identity assertion about construction order, not cold-import RNG side effects.
        from stable_pretraining.backbone.utils import vit_hf as _vit_hf  # noqa: F401
        from stable_worldmodel.wm.lewm.lewm import LeWM as _LeWM  # noqa: F401
        from stable_worldmodel.wm.lewm.module import Embedder as _Embedder  # noqa: F401
        from stable_worldmodel.wm.lewm.module import MLP as _MLP  # noqa: F401
        from stable_worldmodel.wm.lewm.module import Predictor as _Predictor  # noqa: F401

        _seed_all(123)
        direct = _build_direct_lewm_reference(model_cfg, cfg)
        _seed_all(123)
        mwm = build_mwm_from_stable_config(
            family="lewm",
            source_config=_stable_vit_lewm_source_config(model_cfg),
            source_config_sha256="test-stable-vwm-config",
            training_recipe={
                "history_size": int(model_cfg["history_size"]),
                "num_preds": int(model_cfg["num_preds"]),
                "loss_scope": {"regularizers": "shared_latent"},
            },
            K=tuple(int(k) for k in model_cfg["K"]),
            action_dim=int(model_cfg["action_dim"]),
            action_block=int(model_cfg["action_block"]),
            image_shape=tuple(int(x) for x in model_cfg["image_shape"]),
            normalize_imagenet=bool(model_cfg["normalize_imagenet"]),
        )

        mapping = {
            "encoder.": "encoder.",
            "projector.": "projector.",
            "predictor.": "transitions.0.predictor.",
            "action_encoder.": "transitions.0.action_encoder.",
            "pred_proj.": "transitions.0.pred_proj.",
        }
        mwm_state = mwm.state_dict()
        aligned_mwm_state = dict(mwm_state)
        for direct_name, direct_tensor in direct.state_dict().items():
            for src, dst in mapping.items():
                if direct_name.startswith(src):
                    mwm_name = dst + direct_name[len(src) :]
                    self.assertIn(mwm_name, mwm_state, direct_name)
                    self.assertEqual(tuple(direct_tensor.shape), tuple(mwm_state[mwm_name].shape), direct_name)
                    if not direct_name.startswith("encoder."):
                        self.assertTrue(torch.equal(direct_tensor, mwm_state[mwm_name]), direct_name)
                    aligned_mwm_state[mwm_name] = direct_tensor.detach().clone()
                    break
        mwm.load_state_dict(aligned_mwm_state, strict=True)

        batch = {
            "pixels": torch.randn(2, 3, 3, 224, 224),
            "action": torch.randn(2, 3, 4),
        }

        def direct_loss() -> torch.Tensor:
            output = direct.encode(dict(batch))
            pred = direct.predict(output["emb"][:, :2], output["act_emb"][:, :2])
            return (pred - output["emb"][:, 1:]).pow(2).mean()

        direct.train()
        mwm.train()
        d_loss = direct_loss()
        m_out = mwm.training_loss(dict(batch))
        m_loss = m_out["loss"]
        self.assertTrue(torch.allclose(d_loss, m_out["pred_loss"], atol=0.0, rtol=0.0))
        self.assertIn("recon_loss", m_out)

        direct.zero_grad(set_to_none=True)
        mwm.zero_grad(set_to_none=True)
        d_loss.backward()
        m_loss.backward()
        self.assertTrue(torch.equal(direct.predictor.pos_embedding.grad, mwm.transitions[0].predictor.pos_embedding.grad))
        self.assertTrue(torch.equal(direct.projector.net[0].weight.grad, mwm.projector.net[0].weight.grad))

        d_opt = torch.optim.AdamW(direct.parameters(), lr=1e-4, weight_decay=1e-3)
        m_opt = torch.optim.AdamW(mwm.parameters(), lr=1e-4, weight_decay=1e-3)
        d_opt.step()
        m_opt.step()
        self.assertTrue(torch.allclose(direct.predictor.pos_embedding, mwm.transitions[0].predictor.pos_embedding))

    def test_lewm_column_scaler_matches_reference_sample_std(self) -> None:
        values = np.array([[1.0, 3.0], [3.0, 7.0], [np.nan, 1.0]], dtype=np.float32)
        scaler = ZScoreScaler().fit(values)

        self.assertTrue(np.allclose(scaler.mean, np.array([[2.0, 5.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(scaler.std, np.array([[np.sqrt(2.0), np.sqrt(8.0)]], dtype=np.float32)))

    def test_train_transform_preserves_frameskip_action_blocks(self) -> None:
        transform = MWMTrainSampleTransform(normalize_pixels=False)
        sample = {
            "pixels": torch.rand(3, 8, 8, 3),
            "action": torch.arange(30, dtype=torch.float32).reshape(15, 2),
        }
        out = transform(sample)

        self.assertEqual(tuple(out["x"].shape), (3, 3, 8, 8))
        self.assertEqual(tuple(out["a"].shape), (2, 10))
        self.assertTrue(torch.equal(out["a"][0], torch.arange(10, dtype=torch.float32)))

    def test_world_model_policy_inverse_transforms_normalized_actions(self) -> None:
        policy = MWMWorldModelPolicy(
            model=nn.Linear(1, 1),
            solver=FakeSolver(),
            config=PlanConfig(horizon=1, receding_horizon=1, action_block=1),
            process={"action": FakeActionScaler()},
            transform={},
        )
        policy.set_env(FakeVectorEnv())

        action = policy.get_action(
            {
                "pixels": np.zeros((2, 1, 8, 8, 3), dtype=np.uint8),
                "goal": np.zeros((2, 1, 8, 8, 3), dtype=np.uint8),
            }
        )

        self.assertTrue(
            np.allclose(
                action,
                np.array([[10.0, 23.0], [14.0, 29.0]], dtype=np.float32),
            )
        )

    def test_action_spec_distinguishes_base_and_block_dims(self) -> None:
        model = _lewm_matryoshka_model(K=(8,), D=8, action_dim=10, action_block=5)

        self.assertEqual(model.action_dim, 10)
        self.assertEqual(model.metadata["action_spec"], {"dim": 10, "base_dim": 2, "block": 5})

    def test_rollout_ignores_raw_history_actions_for_blocked_lewm_heads(self) -> None:
        model = LeWMMatryoshkaWorldModel(
            encoder=FakeLeWMEncoder(out_dim=4),
            projector=nn.Identity(),
            transitions=[
                TransitionPackage(
                    action_encoder=FakeLeWMActionEncoder(action_dim=10, out_dim=4),
                    predictor=FakeLeWMPredictor(),
                    pred_proj=nn.Identity(),
                )
            ],
            decoders=[ConvImageDecoder(latent_dim=4, image_shape=(8, 8))],
            K=[4],
            D=4,
            action_dim=10,
            action_block=5,
            image_shape=(8, 8),
            normalize_imagenet=False,
            history_size=3,
            num_preds=1,
            head_architectures=[{"K": 4}],
        )
        infos = {
            "pixels": torch.rand(1, 2, 3, 3, 8, 8),
            "action": torch.rand(1, 2, 3, 2),
        }
        candidates = torch.rand(1, 2, 5, 10)

        out = model.rollout_at_level(infos, candidates, level_idx=0)

        self.assertEqual(tuple(out["predicted_emb"].shape), (1, 2, 6, 4))

    def test_eval_policy_uses_explicit_topk_and_auto_batching(self) -> None:
        cfg = OmegaConf.create(
            {
                "eval": {"num_envs": 7},
                "planner": {
                    "horizon": 5,
                    "receding_horizon": 5,
                    "action_block": 5,
                    "batch_size": "auto",
                    "pop_size": 300,
                    "topk": 30,
                    "elite_frac": 0.01,
                    "n_iter": 30,
                    "init_std": 1.0,
                    "seed": 42,
                    "warm_start": True,
                    "clamp_actions": False,
                    "std_unbiased": True,
                    "scheduler": {
                        "enabled": True,
                        "mpc": {"mode": "fixed", "level": "finest"},
                        "cem": {"mode": "fixed", "level": "base"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    },
                },
            }
        )
        policy = build_mwm_policy(
            FakeFidelityCostModel(),
            {"action_block": 5},
            cfg,
            torch.device("cpu"),
            Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            process={},
        )

        self.assertEqual(policy.solver.batch_size, 7)
        self.assertEqual(policy.solver.num_samples, 300)
        self.assertEqual(policy.solver.topk, 30)

    def test_eval_policy_forwards_dynamic_pop_schedule_to_solver(self) -> None:
        cfg = OmegaConf.create(
            {
                "eval": {"num_envs": 1},
                "planner": {
                    "horizon": 2,
                    "receding_horizon": 1,
                    "action_block": 1,
                    "batch_size": "auto",
                    "pop_size": 64,
                    "topk": 8,
                    "elite_frac": 0.25,
                    "n_iter": 3,
                    "init_std": 1.0,
                    "seed": 42,
                    "warm_start": False,
                    "clamp_actions": False,
                    "std_unbiased": True,
                    "scheduler": {
                        "enabled": True,
                        "mpc": {"mode": "fixed", "level": "finest"},
                        "cem": {"mode": "fixed", "level": "base"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    },
                    "pop_schedule": {"start": 64, "end": 16},
                },
            }
        )

        policy = build_mwm_policy(
            FakeFidelityCostModel(),
            {"action_block": 1},
            cfg,
            torch.device("cpu"),
            Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            process={},
        )

        self.assertEqual(policy.solver.pop_schedule, {"start": 64, "end": 16})
        self.assertEqual(policy.solver.elite_frac, 0.25)

    def test_action_preprocessing_is_metadata_driven_not_upstream_role_driven(self) -> None:
        cfg = OmegaConf.create({"eval": {"action_preprocessing": "auto"}, "data": {"action_preprocessing": "auto"}})

        self.assertFalse(
            uses_standardized_action_space(
                object(),
                {"role": "upstream_lewm_converted"},
                cfg,
            )
        )
        self.assertTrue(
            uses_standardized_action_space(
                object(),
                {"role": "upstream_lewm_converted", "action_preprocessing": "standard_scaler"},
                cfg,
            )
        )

    def test_eval_action_process_stats_skip_missing_optional_columns(self) -> None:
        cfg = OmegaConf.create(
            {
                "data": {
                    "pixels_key": "pixels",
                    "action_key": "action",
                    "keys_to_cache": ["action", "proprio", "state"],
                }
            }
        )

        keys = available_stat_keys_for_action_process(cfg, ["episode_idx", "action", "proprio", "pos_agent"])

        self.assertEqual(keys, ["action", "proprio"])

    def test_scheduled_cem_matches_stable_worldmodel_cem_for_fixed_fidelity(self) -> None:
        model = FakeCEMParityCostModel()
        action_space = Box(low=-10.0, high=10.0, shape=(2, 2), dtype=np.float32)
        plan_cfg = PlanConfig(horizon=3, receding_horizon=1, action_block=2, warm_start=False)
        info_dict = {
            "target": torch.tensor(
                [
                    [[0.1, -0.2, 0.3, -0.4], [0.2, -0.1, 0.4, -0.3], [0.3, -0.4, 0.1, -0.2]],
                    [[-0.3, 0.4, -0.1, 0.2], [-0.4, 0.3, -0.2, 0.1], [-0.1, 0.2, -0.3, 0.4]],
                ],
                dtype=torch.float32,
            )
        }
        common = {
            "model": model,
            "batch_size": 2,
            "num_samples": 32,
            "var_scale": 1.0,
            "n_steps": 6,
            "topk": 8,
            "device": "cpu",
            "seed": 123,
        }
        upstream_solver = CEMSolver(**common)
        mwm_solver = MWMScheduledCEMSolver(
            **common,
            scheduler={
                "enabled": True,
                "mpc": {"mode": "fixed", "level": 0},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
            std_unbiased=True,
        )
        upstream_solver.configure(action_space=action_space, n_envs=2, config=plan_cfg)
        mwm_solver.configure(action_space=action_space, n_envs=2, config=plan_cfg)

        upstream = upstream_solver.solve(dict(info_dict))
        mwm = mwm_solver.solve(dict(info_dict))

        self.assertTrue(torch.allclose(mwm["actions"], upstream["actions"]))
        self.assertTrue(torch.allclose(mwm["mean"][0], upstream["mean"][0]))
        self.assertTrue(torch.allclose(mwm["var"][0], upstream["var"][0]))
        self.assertTrue(np.allclose(mwm["costs"], upstream["costs"]))
        self.assertEqual(len(mwm["mwm_diagnostics"]), common["n_steps"])
        self.assertEqual(mwm["mwm_diagnostics"][0]["batch_end"], 2)

    def test_scheduled_cem_uses_dynamic_population_schedule(self) -> None:
        solver = MWMScheduledCEMSolver(
            FakeFidelityCostModel(),
            batch_size=1,
            num_samples=8,
            var_scale=1.0,
            n_steps=3,
            topk=4,
            scheduler={
                "enabled": True,
                "mpc": {"mode": "fixed", "level": 0},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
            seed=0,
            std_unbiased=False,
            pop_schedule={"start": 8, "end": 2},
            elite_frac=0.5,
        )
        solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=PlanConfig(horizon=2, receding_horizon=1, action_block=1),
        )

        result = solver.solve({"pixels": torch.zeros(1, 1)})

        diagnostics = result["mwm_diagnostics"]
        self.assertEqual([entry["num_samples"] for entry in diagnostics], [8, 5, 2])
        self.assertEqual([entry["topk"] for entry in diagnostics], [4, 2, 1])
        self.assertEqual([entry["candidate_action_values"] for entry in diagnostics], [32, 20, 8])

    def test_scheduled_cem_rejects_legacy_get_cost_only_model(self) -> None:
        class LegacyCostOnly(nn.Module):
            K = [4]
            num_levels = 1

            def get_cost(self, info_dict, action_candidates):
                del info_dict
                return action_candidates.square().sum(dim=(2, 3))

        solver = MWMScheduledCEMSolver(
            LegacyCostOnly(),
            batch_size=1,
            num_samples=4,
            n_steps=1,
            topk=2,
            scheduler={
                "enabled": True,
                "mpc": {"mode": "fixed", "level": 0},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
            seed=0,
        )
        solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=PlanConfig(horizon=2, receding_horizon=1, action_block=1),
        )
        with self.assertRaisesRegex(TypeError, "get_cost_with_fidelity"):
            solver.solve({"pixels": torch.zeros(1, 1)})

    def test_trainer_root_cleanup_is_explicit_for_repeatable_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "logs" / "mwm_training" / "review_run" / "old.ckpt"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale", encoding="utf-8")

            cfg = OmegaConf.create({"train": {"clean_trainer_root": True}})
            trainer_root = prepare_trainer_root(root / "checkpoints" / "review_run", cfg, logs_root=root / "logs")
            self.assertEqual(trainer_root, stale.parent)
            self.assertFalse(stale.exists())

            keep = trainer_root / "keep.ckpt"
            keep.write_text("keep", encoding="utf-8")
            cfg.train.clean_trainer_root = False
            trainer_root = prepare_trainer_root(root / "checkpoints" / "review_run", cfg, logs_root=root / "logs")
            self.assertTrue((trainer_root / "keep.ckpt").is_file())

    def test_stable_wm_adapter_checkpoint_callback_can_save_within_large_epochs(self) -> None:
        cfg = OmegaConf.create({"train": {"checkpoint_every_n_train_steps": 1000}})
        callback = stable_wm_adapter_checkpoint_callback(cfg)

        self.assertEqual(callback._every_n_train_steps, 1000)
        self.assertEqual(callback._every_n_epochs, 0)
        self.assertTrue(callback.save_last)

    def test_stable_wm_adapter_checkpoint_callback_can_monitor_validation_metric(self) -> None:
        cfg = OmegaConf.create(
            {
                "train": {
                    "checkpoint_every_n_train_steps": 0,
                    "checkpoint_monitor": "validate/pred_loss_epoch",
                    "checkpoint_mode": "min",
                    "save_top_k": 2,
                }
            }
        )
        callback = stable_wm_adapter_checkpoint_callback(cfg)

        self.assertEqual(callback.monitor, "validate/pred_loss_epoch")
        self.assertEqual(callback.mode, "min")
        self.assertEqual(callback.save_top_k, 2)
        self.assertTrue(callback.save_last)

    def test_select_stable_wm_adapter_export_checkpoint_prefers_best_when_requested(self) -> None:
        cfg = OmegaConf.create({"train": {"export_checkpoint": "best"}})
        callback = SimpleNamespace(best_model_path="best.ckpt", last_model_path="last.ckpt")

        self.assertEqual(select_stable_wm_adapter_export_checkpoint(callback, cfg), "best.ckpt")

    def test_stable_wm_adapter_total_steps_can_decouple_lr_horizon_from_train_epochs(self) -> None:
        cfg = OmegaConf.create({"schedule": {"max_epochs": 80, "lr_max_epochs": 10}})
        loader = [object()] * 7

        self.assertEqual(resolve_stable_wm_adapter_total_steps(cfg, loader), 70)

    def test_all_level_plateau_stop_waits_until_no_level_improves(self) -> None:
        callback = AllLevelPlateauEarlyStopping(
            metrics=["validate/pred_loss_l0", "validate/pred_loss_l1"],
            patience=2,
            warmup_epochs=1,
            relative_min_delta=0.01,
        )
        trainer = SimpleNamespace(
            callback_metrics={
                "validate/pred_loss_l0": torch.tensor(1.0),
                "validate/pred_loss_l1": torch.tensor(2.0),
            },
            current_epoch=0,
            should_stop=False,
            sanity_checking=False,
        )

        callback.on_validation_epoch_end(trainer, None)
        self.assertFalse(trainer.should_stop)

        trainer.current_epoch = 1
        trainer.callback_metrics = {
            "validate/pred_loss_l0": torch.tensor(0.995),
            "validate/pred_loss_l1": torch.tensor(1.5),
        }
        callback.on_validation_epoch_end(trainer, None)
        self.assertFalse(trainer.should_stop)

        trainer.current_epoch = 2
        trainer.callback_metrics = {
            "validate/pred_loss_l0": torch.tensor(0.994),
            "validate/pred_loss_l1": torch.tensor(1.49),
        }
        callback.on_validation_epoch_end(trainer, None)
        self.assertFalse(trainer.should_stop)

        trainer.current_epoch = 3
        trainer.callback_metrics = {
            "validate/pred_loss_l0": torch.tensor(0.993),
            "validate/pred_loss_l1": torch.tensor(1.489),
        }
        callback.on_validation_epoch_end(trainer, None)
        self.assertTrue(trainer.should_stop)

    def test_lightning_runtime_defaults_to_single_gpu_when_cuda_available(self) -> None:
        cfg = OmegaConf.create({"train": {"no_cuda": False}})
        with unittest.mock.patch("torch.cuda.is_available", return_value=True):
            runtime = resolve_lightning_trainer_runtime(cfg)

        self.assertEqual(runtime["accelerator"], "gpu")
        self.assertEqual(runtime["devices"], 1)
        self.assertEqual(runtime["strategy"], "auto")
        self.assertEqual(runtime["num_nodes"], 1)
        self.assertFalse(runtime["sync_batchnorm"])
        self.assertTrue(runtime["use_distributed_sampler"])

    def test_lightning_runtime_allows_opt_in_multi_gpu(self) -> None:
        cfg = OmegaConf.create(
            {
                "train": {
                    "no_cuda": False,
                    "devices": 4,
                    "strategy": "ddp",
                    "num_nodes": 2,
                    "sync_batchnorm": True,
                    "use_distributed_sampler": False,
                }
            }
        )
        with unittest.mock.patch("torch.cuda.is_available", return_value=True):
            runtime = resolve_lightning_trainer_runtime(cfg)

        self.assertEqual(runtime["accelerator"], "gpu")
        self.assertEqual(runtime["devices"], 4)
        self.assertEqual(runtime["strategy"], "ddp")
        self.assertEqual(runtime["num_nodes"], 2)
        self.assertTrue(runtime["sync_batchnorm"])
        self.assertFalse(runtime["use_distributed_sampler"])

    def test_lightning_runtime_uses_cpu_devices_when_cuda_disabled(self) -> None:
        cfg = OmegaConf.create({"train": {"no_cuda": True, "devices": 4, "cpu_devices": 1}})
        with unittest.mock.patch("torch.cuda.is_available", return_value=True):
            runtime = resolve_lightning_trainer_runtime(cfg)

        self.assertEqual(runtime["accelerator"], "cpu")
        self.assertEqual(runtime["devices"], 1)

    def test_stable_wm_adapter_lightning_state_loader_strips_model_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.ckpt"
            expected = nn.Linear(3, 2)
            torch.save(
                {
                    "epoch": 7,
                    "state_dict": {
                        "model.weight": expected.weight.detach().clone(),
                        "model.bias": expected.bias.detach().clone(),
                    },
                },
                path,
            )

            actual = nn.Linear(3, 2)
            checkpoint = load_stable_wm_adapter_lightning_state(actual, path)

            self.assertEqual(checkpoint["epoch"], 7)
            self.assertTrue(torch.equal(actual.weight, expected.weight))
            self.assertTrue(torch.equal(actual.bias, expected.bias))

    def test_train_entrypoint_rejects_non_adapter_lewm_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "train.yaml"
            cfg_path.write_text(
                """
model:
  D: 4
  K: [4]
  dynamics: lewm
train:
  backend: stable_pretraining
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "adapter-owned Stable-WM base architecture"):
                train_mwm_main(str(cfg_path))

    def test_multi_fidelity_scheduler_resolves_mpc_cem_and_rollout_in_order(self) -> None:
        scheduler = FidelityScheduler.from_config(
            {
                "enabled": True,
                "mpc": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                "cem": {"mode": "linear", "start_level": "base", "end_level": "finest"},
                "rollout": {"mode": "linear", "start_level": "base", "end_level": "coarsest"},
            },
            num_levels=4,
            horizon=4,
        )

        early = scheduler.decision(cem_iter=0, n_iter=5, mpc_progress=0.0)
        late = scheduler.decision(cem_iter=4, n_iter=5, mpc_progress=1.0)

        self.assertEqual(early.metadata["mpc_level_idx"], 0)
        self.assertEqual(early.base_level_idx, 0)
        self.assertEqual(early.rollout_level_indices, [0, 0, 0, 0])
        self.assertEqual(early.metadata["terminal_level_idx"], 0)
        self.assertEqual(late.metadata["mpc_level_idx"], 3)
        self.assertEqual(late.base_level_idx, 3)
        self.assertEqual(late.rollout_level_indices, [3, 2, 1, 0])
        self.assertEqual(late.metadata["terminal_level_idx"], 0)

    def test_multi_fidelity_scheduler_rejects_legacy_schema_and_bad_rollout(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy.*planner.scheduler"):
            FidelityScheduler.from_config(
                {"policy": "linear_cem", "start_level": "coarsest", "end_level": "finest"},
                num_levels=4,
                horizon=3,
            )

        bad = FidelityScheduler.from_config(
            {
                "enabled": True,
                "mpc": {"mode": "fixed", "level": 1},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "linear", "start_level": "coarsest", "end_level": "base"},
            },
            num_levels=3,
            horizon=2,
        )
        with self.assertRaisesRegex(ValueError, "lower to higher"):
            bad.decision(cem_iter=0, n_iter=1)

        mpc_base = FidelityScheduler.from_config(
            {
                "enabled": True,
                "mpc": {"mode": "fixed", "level": "base"},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
            num_levels=3,
            horizon=2,
        )
        with self.assertRaisesRegex(ValueError, "mpc.*base"):
            mpc_base.decision(cem_iter=0, n_iter=1)

    def test_scheduled_cem_tracks_mpc_progress_across_replans(self) -> None:
        model = RecordingFidelityCostModel()
        solver = MWMScheduledCEMSolver(
            model,
            batch_size=1,
            num_samples=4,
            n_steps=1,
            topk=2,
            scheduler={
                "enabled": True,
                "mpc": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
            max_replans=3,
            seed=0,
            std_unbiased=False,
        )
        solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=PlanConfig(horizon=2, receding_horizon=1, action_block=1),
        )

        solver.solve({"pixels": torch.zeros(1, 1)})
        solver.solve({"pixels": torch.zeros(1, 1)})
        solver.solve({"pixels": torch.zeros(1, 1)})

        self.assertEqual([round(d.mpc_progress, 2) for d in model.decisions], [0.0, 0.5, 1.0])
        self.assertEqual([d.base_level_idx for d in model.decisions], [0, 0, 1])
        self.assertEqual([h["mwm_diagnostics"][0]["mpc_progress"] for h in solver.solve_history], [0.0, 0.5, 1.0])

        solver.reset_history()
        model.decisions.clear()
        solver.solve({"pixels": torch.zeros(1, 1)})
        self.assertEqual(model.decisions[-1].mpc_progress, 0.0)

    def test_scheduled_cem_forwards_dynamics_flop_audit_mode(self) -> None:
        model = RecordingFidelityCostModel()
        solver = MWMScheduledCEMSolver(
            model,
            batch_size=1,
            num_samples=4,
            n_steps=1,
            topk=2,
            scheduler={
                "enabled": True,
                "mpc": {"mode": "fixed", "level": "finest"},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base"},
            },
            flop_accounting="dynamics_audit",
            seed=0,
            std_unbiased=False,
        )
        solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=PlanConfig(horizon=2, receding_horizon=1, action_block=1),
        )

        result = solver.solve({"pixels": torch.zeros(1, 1)})

        diagnostics = result["mwm_diagnostics"][0]
        self.assertEqual(model.decisions[-1].metadata["flop_accounting"], "dynamics_audit")
        self.assertEqual(diagnostics["model_dynamics_flops"], 123)
        self.assertEqual(diagnostics["model_flop_accounting"], "dynamics_audit")

    def test_lewm_dynamic_rollout_scores_final_active_level(self) -> None:
        model = _lewm_matryoshka_model(K=(2, 4), D=4, action_dim=2, history_size=2)
        infos = {
            "pixels": torch.zeros(1, 2, 2, 3, 8, 8),
            "goal": torch.zeros(1, 2, 2, 3, 8, 8),
        }
        candidates = torch.zeros(1, 2, 4, 2)
        decision = SimpleNamespace(base_level_idx=1, rollout_level_indices=[1, 0, 0, 0])

        cost = model.get_cost_with_fidelity(infos, candidates, decision)

        self.assertEqual(tuple(cost.shape), (1, 2))
        self.assertTrue(torch.isfinite(cost).all())
        self.assertEqual(tuple(infos["predicted_emb"].shape), (1, 2, 5, 4))
        self.assertEqual(model._last_cost_diagnostics["base_level_idx"], 1)
        self.assertEqual(model._last_cost_diagnostics["terminal_level_idx"], 0)
        self.assertEqual(model._last_cost_diagnostics["terminal_k"], 2)

    def test_lewm_dynamics_flop_audit_profiles_active_rollout(self) -> None:
        model = _lewm_matryoshka_model(K=(2, 4), D=4, action_dim=2, history_size=2)
        infos = {
            "pixels": torch.zeros(1, 1, 2, 3, 8, 8),
            "goal": torch.zeros(1, 1, 2, 3, 8, 8),
        }
        candidates = torch.zeros(1, 1, 3, 2)
        decision = SimpleNamespace(
            base_level_idx=1,
            rollout_level_indices=[1, 0, 0],
            metadata={"flop_accounting": "dynamics_audit"},
        )

        model.get_cost_with_fidelity(infos, candidates, decision)

        self.assertGreater(model._last_cost_diagnostics["dynamics_flops"], 0)
        self.assertEqual(model._last_cost_diagnostics["flop_accounting"], "dynamics_audit")


if __name__ == "__main__":
    unittest.main()
