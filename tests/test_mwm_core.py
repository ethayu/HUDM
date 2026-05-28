from __future__ import annotations

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

from eval_mwm import _build_mwm_policy
from mwm.adapters.lewm import LeWMMatryoshkaWorldModel, LeWMObjectImporter, build_mwm_lewm
from mwm.checkpoints import CHECKPOINT_FORMAT, load_world_model_from_checkpoint, save_world_checkpoint
from mwm.data.stable_wm import MWMTrainSampleTransform
from mwm.eval.policy import MWMWorldModelPolicy
from mwm.fidelity import FidelityScheduler
from mwm.models.world_model import MWMWorldModel, mwm_prediction_loss
from mwm.planning.scheduled_cem import MWMScheduledCEMSolver
from train_mwm import _exact_lewm_checkpoint_callback, _load_exact_lewm_lightning_state, _prepare_trainer_root
from train_mwm import _build_exact_lewm_object, main as train_mwm_main


class FakeLeWMEncoder(nn.Module):
    def __init__(self, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = False) -> SimpleNamespace:
        del interpolate_pos_encoding
        pooled = x.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled).unsqueeze(1))


class FakeLeWMActionEncoder(nn.Module):
    def __init__(self, action_dim: int = 2, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(action_dim, out_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.proj(action)


class FakeLeWMPredictor(nn.Module):
    def forward(self, z: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        return z + action_emb


class FakeLeWMObject(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeLeWMEncoder()
        self.action_encoder = FakeLeWMActionEncoder()
        self.predictor = FakeLeWMPredictor()


class FakeCostLeWMObject(FakeLeWMObject):
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        del info_dict
        return action_candidates.square().sum(dim=(2, 3))


class FakeGoalEmbCostLeWMObject(FakeLeWMObject):
    def encode(self, info_dict: dict) -> dict:
        pixels = info_dict["pixels"]
        b, t = int(pixels.shape[0]), int(pixels.shape[1])
        return {**info_dict, "emb": torch.ones(b, t, 4, device=pixels.device, dtype=pixels.dtype)}

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        goal_emb = info_dict["goal_emb"]
        if goal_emb.ndim != 4:
            raise AssertionError(f"goal_emb must be 4D for batched upstream Le-WM cost, got {tuple(goal_emb.shape)}")
        if int(goal_emb.shape[0]) != int(action_candidates.shape[0]):
            raise AssertionError("goal_emb batch dimension must match candidate batch dimension")
        return goal_emb.expand(action_candidates.shape[0], action_candidates.shape[1], goal_emb.shape[2], goal_emb.shape[3]).sum(dim=(2, 3))


class BrokenLeWMObject(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeLeWMEncoder()


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


class MWMCoreTests(unittest.TestCase):
    def test_lewm_object_import_roundtrips_canonical_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "lewm_object.pt"
            torch.save(FakeLeWMObject(), source_path)

            model = LeWMObjectImporter(
                str(source_path),
                D=4,
                K=(4,),
                action_dim=2,
                action_block=1,
                image_shape=(8, 8),
                normalize_imagenet=False,
                expected_class_name=FakeLeWMObject.__name__,
            ).import_model()
            self.assertIsInstance(model, MWMWorldModel)
            self.assertEqual(model.K, [4])

            out_dir = root / "checkpoint"
            save_world_checkpoint(
                model,
                out_dir,
                metadata={
                    "env_id": "swm/PushT-v1",
                    "restore_spec": "pusht_state_goal_state",
                    "image_shape": [8, 8],
                    "action_dim": 2,
                    "action_block": 1,
                    "levels": [4],
                    "dataset": {"pixels_key": "pixels", "action_key": "action"},
                },
            )
            self.assertEqual(
                sorted(path.name for path in out_dir.iterdir()),
                ["config.json", "weights.pt", "world_metadata.json"],
            )

            loaded, metadata, epoch = load_world_model_from_checkpoint(out_dir, None, device=torch.device("cpu"))
            self.assertIsInstance(loaded, MWMWorldModel)
            self.assertEqual(metadata["format"], CHECKPOINT_FORMAT)
            self.assertEqual(epoch, 0)
            self.assertTrue(metadata["artifacts"]["weights"]["sha256"])
            self.assertEqual(metadata["action_spec"]["dim"], 2)
            self.assertEqual(loaded.metadata["action_spec"]["dim"], 2)
            self.assertEqual(loaded.metadata["preprocessing_spec"]["image"], "identity")

            (out_dir / "extra.txt").write_text("bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-checkpoint files"):
                load_world_model_from_checkpoint(out_dir, None, device=torch.device("cpu"))

    def test_imported_lewm_cost_delegates_to_source_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "lewm_object.pt"
            torch.save(FakeCostLeWMObject(), source_path)

            model = LeWMObjectImporter(
                str(source_path),
                D=4,
                K=(4,),
                action_dim=2,
                action_block=1,
                image_shape=(8, 8),
                normalize_imagenet=False,
                expected_class_name=FakeCostLeWMObject.__name__,
            ).import_model()
            decision = FidelityScheduler.from_config(
                {"policy": "fixed", "level": 0, "rollout_level": 0},
                num_levels=1,
                horizon=3,
            ).decision(cem_iter=0, n_iter=1)
            infos = {
                "pixels": torch.rand(2, 5, 1, 3, 8, 8),
                "goal": torch.rand(2, 5, 1, 3, 8, 8),
                "action": torch.zeros(2, 5, 1, 2),
            }
            actions = torch.randn(2, 5, 3, 2)

            expected = model.source_model.get_cost(dict(infos), actions)
            actual = model.get_cost_with_fidelity(dict(infos), actions, decision)

            self.assertTrue(torch.allclose(actual, expected))
            self.assertTrue(model._last_cost_diagnostics["delegated_source_cost"])

    def test_imported_lewm_precomputes_batched_goal_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "lewm_object.pt"
            torch.save(FakeGoalEmbCostLeWMObject(), source_path)

            model = LeWMObjectImporter(
                str(source_path),
                D=4,
                K=(4,),
                action_dim=2,
                action_block=1,
                image_shape=(8, 8),
                normalize_imagenet=False,
                expected_class_name=FakeGoalEmbCostLeWMObject.__name__,
            ).import_model()
            decision = FidelityScheduler.from_config(
                {"policy": "fixed", "level": 0, "rollout_level": 0},
                num_levels=1,
                horizon=3,
            ).decision(cem_iter=0, n_iter=1)
            infos = {
                "pixels": torch.rand(2, 5, 1, 3, 8, 8),
                "goal": torch.rand(2, 5, 1, 3, 8, 8),
                "action": torch.zeros(2, 5, 1, 2),
            }
            actions = torch.randn(2, 5, 3, 2)

            costs = model.get_cost_with_fidelity(infos, actions, decision)

            self.assertEqual(tuple(infos["goal_emb"].shape), (2, 1, 1, 4))
            self.assertEqual(tuple(costs.shape), (2, 5))
            self.assertTrue(model._last_cost_diagnostics["delegated_source_cost"])

    def test_canonical_checkpoint_export_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra_dir = root / "checkpoint"
            extra_dir.mkdir()
            (extra_dir / "notes.txt").write_text("not canonical", encoding="utf-8")
            model = build_mwm_lewm(
                {
                    "encoder": "cnn",
                    "D": 4,
                    "K": [4],
                    "action_dim": 2,
                    "image_shape": (8, 8),
                    "normalize_imagenet": False,
                    "dynamics": "mlp",
                }
            )
            with self.assertRaisesRegex(ValueError, "non-checkpoint files"):
                save_world_checkpoint(model, extra_dir, metadata={"env_id": "swm/PushT-v1"})

    def test_lewm_object_import_validates_required_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "broken.pt"
            torch.save(BrokenLeWMObject(), source)

            importer = LeWMObjectImporter(
                str(source),
                D=4,
                K=(4,),
                action_dim=2,
                image_shape=(8, 8),
                expected_class_name=BrokenLeWMObject.__name__,
            )
            with self.assertRaisesRegex(ValueError, "missing components"):
                importer.import_model()

    def test_lewm_object_import_validates_expected_class_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lewm_object.pt"
            torch.save(FakeLeWMObject(), source)

            importer = LeWMObjectImporter(
                str(source),
                D=4,
                K=(4,),
                action_dim=2,
                image_shape=(8, 8),
                expected_class_name="wrong.Module",
            )
            with self.assertRaisesRegex(ValueError, "expected"):
                importer.import_model()

    def test_single_fidelity_k_equals_d_uses_adapter_owned_lewm_loss(self) -> None:
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 8,
                "K": [8],
                "action_dim": 2,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "history_size": 2,
                "num_preds": 1,
                "predictor_depth": 1,
                "predictor_heads": 2,
                "predictor_dim_head": 4,
                "predictor_mlp_dim": 16,
                "projector_hidden_dim": 16,
            }
        )
        self.assertIsInstance(model, LeWMMatryoshkaWorldModel)
        self.assertEqual(model.K, [model.D])
        self.assertFalse(hasattr(model, "decoders"))
        self.assertEqual(len(model.transitions), 1)
        self.assertEqual(model.metadata["action_spec"]["dim"], 2)
        self.assertEqual(model.metadata["preprocessing_spec"]["image"], "identity")
        self.assertEqual(model.metadata["architecture_version"], "lewm_base_adapter_v1")

        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}
        out = model.training_loss(batch)
        self.assertIn("pred_loss_l0", out)
        self.assertIn("loss", out)

    def test_lewm_matryoshka_head_scaling_and_k_may_omit_d(self) -> None:
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 192,
                "K": [48, 96],
                "action_dim": 10,
                "action_block": 5,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "predictor_depth": 1,
                "predictor_heads": 16,
                "predictor_dim_head": 64,
                "predictor_mlp_dim": 2048,
                "projector_hidden_dim": 2048,
            }
        )

        self.assertEqual(model.K, [48, 96])
        self.assertEqual(len(model.transitions), 2)
        self.assertEqual([h["predictor_heads"] for h in model.metadata["head_architectures"]], [4, 8])
        self.assertEqual([h["predictor_dim_head"] for h in model.metadata["head_architectures"]], [16, 32])
        self.assertEqual([h["predictor_mlp_dim"] for h in model.metadata["head_architectures"]], [512, 1024])

        batch = {"pixels": torch.rand(2, 4, 3, 8, 8), "action": torch.randn(2, 4, 10)}
        out = model.training_loss(batch)
        self.assertIn("pred_loss_l0", out)
        self.assertIn("pred_loss_l1", out)
        self.assertNotIn("pred_loss_l2", out)

    def test_lewm_sigreg_is_shared_once_by_default(self) -> None:
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 8,
                "K": [4, 8],
                "action_dim": 2,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "history_size": 2,
                "num_preds": 1,
                "predictor_depth": 1,
                "predictor_heads": 2,
                "predictor_dim_head": 4,
                "predictor_mlp_dim": 16,
                "projector_hidden_dim": 16,
            }
        )
        reg = CountingRegularizer()
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        out = model.training_loss(batch, sigreg=reg, sigreg_weight=0.5, sigreg_scope="shared_latent")

        self.assertIn("sigreg_loss", out)
        self.assertEqual(reg.calls, 1)
        self.assertEqual(reg.shapes[0][-1], 8)

    def test_lewm_sigreg_can_be_per_level_when_explicit(self) -> None:
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 8,
                "K": [4, 8],
                "action_dim": 2,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "history_size": 2,
                "num_preds": 1,
                "predictor_depth": 1,
                "predictor_heads": 2,
                "predictor_dim_head": 4,
                "predictor_mlp_dim": 16,
                "projector_hidden_dim": 16,
            }
        )
        reg = CountingRegularizer()
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        model.training_loss(batch, sigreg=reg, sigreg_weight=0.5, sigreg_scope="per_level_prefix")

        self.assertEqual(reg.calls, 2)
        self.assertEqual([shape[-1] for shape in reg.shapes], [4, 8])

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

        torch.manual_seed(123)
        direct = _build_exact_lewm_object(model_cfg, cfg)
        torch.manual_seed(123)
        mwm = build_mwm_lewm(model_cfg)

        mapping = {
            "encoder.": "encoder.",
            "projector.": "projector.",
            "predictor.": "transitions.0.predictor.",
            "action_encoder.": "transitions.0.action_encoder.",
            "pred_proj.": "transitions.0.pred_proj.",
        }
        mwm_state = mwm.state_dict()
        for direct_name, direct_tensor in direct.state_dict().items():
            for src, dst in mapping.items():
                if direct_name.startswith(src):
                    mwm_name = dst + direct_name[len(src) :]
                    self.assertTrue(torch.equal(direct_tensor, mwm_state[mwm_name]), direct_name)
                    break

        batch = {
            "pixels": torch.randn(2, 3, 3, 224, 224),
            "action": torch.randn(2, 3, 4),
        }

        def direct_loss() -> torch.Tensor:
            output = direct.encode(dict(batch))
            pred = direct.predict(output["emb"][:, :2], output["act_emb"][:, :2])
            return (pred - output["emb"][:, 1:].detach()).pow(2).mean()

        direct.train()
        mwm.train()
        d_loss = direct_loss()
        m_loss = mwm.training_loss(dict(batch))["loss"]
        self.assertTrue(torch.allclose(d_loss, m_loss, atol=0.0, rtol=0.0))

        direct.zero_grad(set_to_none=True)
        mwm.zero_grad(set_to_none=True)
        d_loss.backward()
        m_loss.backward()
        self.assertTrue(torch.equal(direct.predictor.pos_embedding.grad, mwm.transitions[0].predictor.pos_embedding.grad))

        d_opt = torch.optim.AdamW(direct.parameters(), lr=1e-4, weight_decay=1e-3)
        m_opt = torch.optim.AdamW(mwm.parameters(), lr=1e-4, weight_decay=1e-3)
        d_opt.step()
        m_opt.step()
        self.assertTrue(torch.allclose(direct.predictor.pos_embedding, mwm.transitions[0].predictor.pos_embedding))

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
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 8,
                "K": [8],
                "action_dim": 10,
                "action_block": 5,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "dynamics": "mlp",
            }
        )

        self.assertEqual(model.action_dim, 10)
        self.assertEqual(model.metadata["action_spec"], {"dim": 10, "base_dim": 2, "block": 5})

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
                    "scheduler": {"policy": "fixed", "level": "finest", "rollout_level": "base"},
                },
            }
        )
        policy = _build_mwm_policy(
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
            scheduler={"policy": "fixed", "level": 0, "rollout_level": 0},
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
            scheduler={"policy": "fixed", "level": 0, "rollout_level": 0},
            seed=0,
        )
        solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            n_envs=1,
            config=PlanConfig(horizon=2, receding_horizon=1, action_block=1),
        )
        with self.assertRaisesRegex(TypeError, "get_cost_with_fidelity"):
            solver.solve({"pixels": torch.zeros(1, 1)})

    def test_trainer_root_cleanup_is_explicit_for_repeatable_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "logs" / "mwm_training" / "review_run" / "old.ckpt"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale", encoding="utf-8")

            cfg = OmegaConf.create({"train": {"clean_trainer_root": True}})
            trainer_root = _prepare_trainer_root(root / "checkpoints" / "review_run", cfg, logs_root=root / "logs")
            self.assertEqual(trainer_root, stale.parent)
            self.assertFalse(stale.exists())

            keep = trainer_root / "keep.ckpt"
            keep.write_text("keep", encoding="utf-8")
            cfg.train.clean_trainer_root = False
            trainer_root = _prepare_trainer_root(root / "checkpoints" / "review_run", cfg, logs_root=root / "logs")
            self.assertTrue((trainer_root / "keep.ckpt").is_file())

    def test_exact_lewm_checkpoint_callback_can_save_within_large_epochs(self) -> None:
        cfg = OmegaConf.create({"train": {"checkpoint_every_n_train_steps": 1000}})
        callback = _exact_lewm_checkpoint_callback(cfg)

        self.assertEqual(callback._every_n_train_steps, 1000)
        self.assertEqual(callback._every_n_epochs, 0)
        self.assertTrue(callback.save_last)

    def test_exact_lewm_lightning_state_loader_strips_model_prefix(self) -> None:
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
            checkpoint = _load_exact_lewm_lightning_state(actual, path)

            self.assertEqual(checkpoint["epoch"], 7)
            self.assertTrue(torch.equal(actual.weight, expected.weight))
            self.assertTrue(torch.equal(actual.bias, expected.bias))

    def test_train_entrypoint_rejects_generic_single_level_lewm_backend(self) -> None:
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

            with self.assertRaisesRegex(ValueError, "Trainable Le-WM MWM"):
                train_mwm_main(str(cfg_path))

    def test_multi_fidelity_scheduler_monotonicity_and_no_low_to_high_rollout(self) -> None:
        scheduler = FidelityScheduler.from_config(
            {
                "policy": "linear_cem",
                "start_level": "coarsest",
                "end_level": "finest",
                "rollout_level": "base",
            },
            num_levels=4,
            horizon=3,
        )
        decisions = [scheduler.decision(cem_iter=i, n_iter=5) for i in range(5)]
        bases = [d.base_level_idx for d in decisions]
        self.assertEqual(bases, sorted(bases))
        self.assertEqual(bases[0], 0)
        self.assertEqual(bases[-1], 3)

        bad = FidelityScheduler.from_config(
            {"policy": "fixed", "level": 1, "rollout_levels": [0, 1]},
            num_levels=3,
            horizon=2,
        )
        with self.assertRaisesRegex(ValueError, "lower to higher"):
            bad.decision(cem_iter=0, n_iter=1)


if __name__ == "__main__":
    unittest.main()
