from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from typing import Any

import lightning as pl
import stable_pretraining as spt
import torch
from torch.utils.data import DataLoader, Dataset

from mwm.training.stable_wm_lightning import (
    OptimizerIsolatedSPTModule,
    WorldOnlySPTModule,
    stable_wm_parameter_partitions,
    stable_wm_upstream_lewm_parameter_order,
)
from tests.test_mwm_core import _lewm_matryoshka_model


_PARITY_STEPS = 101


class _ParityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 5)
        self.transition = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(5, 2),
        )
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(5, 2)])

    def losses(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.tanh(self.encoder(batch["input"]))
        world_loss = (self.transition(latent) - batch["world_target"]).square().mean()
        decoder_loss = (self.decoders[0](latent.detach()) - batch["decoder_target"]).square().mean()
        return world_loss, decoder_loss


class _TransitionPackage(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Deliberately use MWM's natural registration order, which differs
        # from raw LeWM's predictor-before-action order.
        self.action_encoder = torch.nn.Linear(2, 2)
        self.predictor = torch.nn.Linear(2, 2)
        self.pred_proj = torch.nn.Linear(2, 2)


class _SingleLevelLeWMOrderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.projector = torch.nn.Linear(2, 2)
        self.transitions = torch.nn.ModuleList([_TransitionPackage()])
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


class _FixedBatches(Dataset):
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.batches[index].items()}


def _parameter_partitions(
    model: _ParityModel,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    decoder_ids = {id(param) for param in model.decoders.parameters()}
    world = tuple(param for param in model.parameters() if id(param) not in decoder_ids)
    decoder = tuple(param for param in model.parameters() if id(param) in decoder_ids)
    assert {id(param) for param in world}.isdisjoint(decoder_ids)
    assert {id(param) for param in world + decoder} == {id(param) for param in model.parameters()}
    return world, decoder


def _scheduler_config() -> dict[str, Any]:
    return {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": max(1, int(0.01 * _PARITY_STEPS)),
        "max_steps": _PARITY_STEPS,
    }


def _multi_optimizer_config() -> dict[str, Any]:
    return {
        "decoder_opt": {
            "modules": r"^model\.decoders(?:\.|$)",
            "optimizer": {"type": "AdamW", "lr": 7e-4, "weight_decay": 3e-3},
            "scheduler": _scheduler_config(),
            "interval": "epoch",
        },
        "world_opt": {
            "modules": r"^model(?:\.|$)",
            "optimizer": {"type": "AdamW", "lr": 3e-4, "weight_decay": 1e-3},
            "scheduler": _scheduler_config(),
            "interval": "epoch",
        },
    }


def _world_optimizer_config() -> dict[str, Any]:
    return {
        "world_opt": {
            "modules": r"^model\.(?!decoders(?:\.|$))",
            "optimizer": {"type": "AdamW", "lr": 3e-4, "weight_decay": 1e-3},
            "scheduler": _scheduler_config(),
            "interval": "epoch",
        }
    }


def _as_optimizers(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


class _TraceMixin:
    trace: list[dict[str, Any]]

    def _initialize_trace(self) -> None:
        self.trace = []

    def _optimizer_name(self, optimizer: Any) -> str:
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        optimizer_ids = {id(param) for group in raw_optimizer.param_groups for param in group["params"]}
        decoder_ids = {id(param) for param in self.model.decoders.parameters()}
        return "decoder_opt" if optimizer_ids and optimizer_ids.issubset(decoder_ids) else "world_opt"

    def _named_gradients(
        self,
        *,
        parameter_ids: set[int] | None = None,
        include_none: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        result: dict[str, torch.Tensor | None] = {}
        for name, parameter in self.model.named_parameters():
            if parameter_ids is not None and id(parameter) not in parameter_ids:
                continue
            if parameter.grad is None and not include_none:
                continue
            result[name] = None if parameter.grad is None else parameter.grad.detach().clone()
        return result

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        optimizers = _as_optimizers(self.optimizers())
        self.trace.append(
            {
                "losses": {},
                "raw_gradients": {},
                "clipped_gradients": {},
                "learning_rates_before": {
                    self._optimizer_name(optimizer): tuple(group["lr"] for group in optimizer.param_groups)
                    for optimizer in optimizers
                },
            }
        )

    def _record_losses(self, world_loss: torch.Tensor, decoder_loss: torch.Tensor | None) -> None:
        decoder_value = None if decoder_loss is None else decoder_loss.detach().clone()
        joint_loss = world_loss if decoder_loss is None else world_loss + decoder_loss
        self.trace[-1]["losses"] = {
            "world": world_loss.detach().clone(),
            "decoder": decoder_value,
            "joint": joint_loss.detach().clone(),
        }

    def _record_raw_gradients(self) -> None:
        self.trace[-1]["raw_gradients"] = self._named_gradients()

    def clip_gradients(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        super().clip_gradients(optimizer, *args, **kwargs)
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        parameter_ids = {id(param) for group in raw_optimizer.param_groups for param in group["params"]}
        self.trace[-1]["clipped_gradients"][self._optimizer_name(optimizer)] = self._named_gradients(
            parameter_ids=parameter_ids,
            include_none=False,
        )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        optimizers = _as_optimizers(self.optimizers())
        self.trace[-1]["learning_rates_after"] = {
            self._optimizer_name(optimizer): tuple(group["lr"] for group in optimizer.param_groups)
            for optimizer in optimizers
        }
        self.trace[-1]["parameters"] = {
            name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
        }


class _ReferenceCustomModule(_TraceMixin, pl.LightningModule):
    """Test-only snapshot of the pre-refactor custom optimizer loop."""

    def __init__(self, model: _ParityModel, *, decoder_enabled: bool) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.decoder_enabled = decoder_enabled
        self.world_parameters, self.decoder_parameters = _parameter_partitions(model)
        if not decoder_enabled:
            for parameter in self.decoder_parameters:
                parameter.requires_grad_(False)
            self.decoder_parameters = ()
        self._initialize_trace()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        world_loss, decoder_loss = self.model.losses(batch)
        active_decoder_loss = decoder_loss if self.decoder_enabled else None
        self._record_losses(world_loss, active_decoder_loss)
        result = {"loss": world_loss}
        if active_decoder_loss is not None:
            result["decoder_loss"] = active_decoder_loss
        return result

    def configure_optimizers(self):
        from stable_pretraining.optim import create_scheduler

        optimizers = [torch.optim.AdamW(self.world_parameters, lr=3e-4, weight_decay=1e-3)]
        if self.decoder_enabled:
            optimizers.append(torch.optim.AdamW(self.decoder_parameters, lr=7e-4, weight_decay=3e-3))
        schedulers = [
            {
                "scheduler": create_scheduler(optimizer, _scheduler_config(), module=self),
                "interval": "step",
                "frequency": 1,
            }
            for optimizer in optimizers
        ]
        return optimizers, schedulers

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        output = self(batch)
        optimizers = _as_optimizers(self.optimizers())
        schedulers = _as_optimizers(self.lr_schedulers())
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        joint_loss = output["loss"]
        if self.decoder_enabled:
            joint_loss = joint_loss + output["decoder_loss"]
        self.manual_backward(joint_loss)
        self._record_raw_gradients()

        world_optimizer = optimizers[0]
        self.clip_gradients(world_optimizer, gradient_clip_val=0.08, gradient_clip_algorithm="norm")
        world_optimizer.step()
        schedulers[0].step()
        if self.decoder_enabled:
            decoder_optimizer = optimizers[1]
            self.clip_gradients(decoder_optimizer, gradient_clip_val=0.03, gradient_clip_algorithm="norm")
            decoder_optimizer.step()
            schedulers[1].step()
        return output


def _spt_forward(module: Any, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    world_loss, decoder_loss = module.model.losses(batch)
    active_decoder_loss = decoder_loss if module.decoder_enabled else None
    module._record_losses(world_loss, active_decoder_loss)
    return {"loss": world_loss if active_decoder_loss is None else world_loss + active_decoder_loss}


class _TracedHistoricalSPTModule(_TraceMixin, WorldOnlySPTModule):
    def __init__(self, model: _ParityModel) -> None:
        world_parameters, _ = _parameter_partitions(model)
        super().__init__(
            model=model,
            forward=_spt_forward,
            optim=_world_optimizer_config(),
            world_parameters=world_parameters,
        )
        self.decoder_enabled = False
        self._initialize_trace()

    def after_manual_backward(self) -> None:
        self._record_raw_gradients()


class _TracedIsolatedSPTModule(_TraceMixin, OptimizerIsolatedSPTModule):
    def __init__(self, model: _ParityModel) -> None:
        world_parameters, decoder_parameters = _parameter_partitions(model)
        super().__init__(
            model=model,
            forward=_spt_forward,
            optim=_multi_optimizer_config(),
            world_parameters=world_parameters,
            decoder_parameters=decoder_parameters,
            world_gradient_clip_val=0.08,
            decoder_gradient_clip_val=0.03,
        )
        self.decoder_enabled = True
        self._initialize_trace()

    def after_manual_backward(self) -> None:
        OptimizerIsolatedSPTModule.after_manual_backward(self)
        self._record_raw_gradients()


def _actual_dense_forward(
    module: Any, batch: dict[str, torch.Tensor], stage: str
) -> dict[str, torch.Tensor]:
    del stage
    output = module.model.training_loss(
        batch,
        sigreg=None,
        sigreg_weight=0.0,
        decoder_training_enabled=True,
    )
    return {"loss": output["loss"] + output["decoder_loss"]}


class _ActualDenseIsolatedSPTModule(OptimizerIsolatedSPTModule):
    def __init__(self, model: torch.nn.Module) -> None:
        world_parameters, decoder_parameters = stable_wm_parameter_partitions(model)
        super().__init__(
            model=model,
            forward=_actual_dense_forward,
            optim=_multi_optimizer_config(),
            world_parameters=world_parameters,
            decoder_parameters=decoder_parameters,
            world_gradient_clip_val=0.08,
            decoder_gradient_clip_val=0.03,
        )


def _run(module: pl.LightningModule, batches: list[dict[str, torch.Tensor]], *, clip_val: float | None) -> None:
    with tempfile.TemporaryDirectory() as trainer_root:
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_epochs=1,
            gradient_clip_val=clip_val,
            default_root_dir=trainer_root,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            deterministic=True,
        )
        trainer.fit(module, train_dataloaders=DataLoader(_FixedBatches(batches), batch_size=None, shuffle=False))


class StablePretrainingOptimizerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generator = torch.Generator().manual_seed(9127)
        cls.batches = [
            {
                "input": torch.randn(4, 3, generator=generator),
                "world_target": torch.randn(4, 2, generator=generator),
                "decoder_target": torch.randn(4, 2, generator=generator),
            }
            for _ in range(_PARITY_STEPS)
        ]

    def assertTensorTreeEqual(self, left: Any, right: Any, path: str = "trace") -> None:
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assertTensorTreeEqual(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                self.assertTensorTreeEqual(left_value, right_value, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def _compare(self, *, decoder_enabled: bool) -> None:
        torch.manual_seed(441)
        initial = _ParityModel()
        reference = _ReferenceCustomModule(copy.deepcopy(initial), decoder_enabled=decoder_enabled)
        if decoder_enabled:
            candidate: pl.LightningModule = _TracedIsolatedSPTModule(copy.deepcopy(initial))
            trainer_clip = None
        else:
            candidate = _TracedHistoricalSPTModule(copy.deepcopy(initial))
            trainer_clip = 0.08

        _run(reference, self.batches, clip_val=None)
        _run(candidate, self.batches, clip_val=trainer_clip)

        self.assertEqual(len(reference.trace), _PARITY_STEPS)
        self.assertEqual(len(candidate.trace), _PARITY_STEPS)
        expected_order = ["world_opt", "decoder_opt"] if decoder_enabled else ["world_opt"]
        self.assertEqual(
            [reference._optimizer_name(optimizer) for optimizer in reference.trainer.optimizers],
            expected_order,
        )
        self.assertEqual(
            [candidate._optimizer_name(optimizer) for optimizer in candidate.trainer.optimizers],
            expected_order,
        )
        if not decoder_enabled:
            optimizer_ids = {
                id(parameter)
                for group in candidate.trainer.optimizers[0].param_groups
                for parameter in group["params"]
            }
            world_parameters, decoder_parameters = _parameter_partitions(candidate.model)
            self.assertEqual(optimizer_ids, {id(parameter) for parameter in world_parameters})
            self.assertTrue(optimizer_ids.isdisjoint({id(parameter) for parameter in decoder_parameters}))
        self.assertTensorTreeEqual(reference.trace, candidate.trace)

    def test_world_only_matches_pre_refactor_custom_path_for_101_steps(self) -> None:
        self._compare(decoder_enabled=False)

    def test_single_level_world_optimizer_uses_raw_lewm_parameter_order(self) -> None:
        model = _SingleLevelLeWMOrderModel()
        world_parameters, _ = stable_wm_parameter_partitions(model)
        ordered = stable_wm_upstream_lewm_parameter_order(model, world_parameters)
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        self.assertEqual(
            [parameter_names[id(parameter)].split(".")[0:3] for parameter in ordered],
            [
                ["encoder", "weight"],
                ["encoder", "bias"],
                ["transitions", "0", "predictor"],
                ["transitions", "0", "predictor"],
                ["transitions", "0", "action_encoder"],
                ["transitions", "0", "action_encoder"],
                ["projector", "weight"],
                ["projector", "bias"],
                ["transitions", "0", "pred_proj"],
                ["transitions", "0", "pred_proj"],
            ],
        )

        module = WorldOnlySPTModule(
            model=model,
            forward=lambda *_args, **_kwargs: {"loss": torch.zeros((), requires_grad=True)},
            optim={
                "world_opt": {
                    "modules": r"^model\.(?!decoders(?:\.|$))",
                    "optimizer": {"type": "AdamW", "lr": 3e-4},
                    "scheduler": _scheduler_config(),
                    "interval": "epoch",
                }
            },
            world_parameters=ordered,
        )
        optimizers, _ = module.configure_optimizers()
        self.assertEqual(
            [id(parameter) for parameter in optimizers[0].param_groups[0]["params"]],
            [id(parameter) for parameter in ordered],
        )

    def test_two_optimizer_matches_pre_refactor_custom_path_for_101_steps(self) -> None:
        self._compare(decoder_enabled=True)

    def test_two_optimizer_checkpoint_resume_matches_uninterrupted_training(self) -> None:
        batches = self.batches[:3]

        def fresh_module() -> _TracedIsolatedSPTModule:
            torch.manual_seed(441)
            return _TracedIsolatedSPTModule(_ParityModel())

        def trainer(root: Path, *, max_epochs: int) -> pl.Trainer:
            return pl.Trainer(
                accelerator="cpu",
                devices=1,
                precision="32-true",
                max_epochs=max_epochs,
                gradient_clip_val=None,
                default_root_dir=root,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                deterministic=True,
            )

        loader = DataLoader(_FixedBatches(batches), batch_size=None, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uninterrupted = fresh_module()
            uninterrupted_trainer = trainer(root / "uninterrupted", max_epochs=2)
            uninterrupted_trainer.fit(uninterrupted, train_dataloaders=loader)

            first_stage = fresh_module()
            first_trainer = trainer(root / "first", max_epochs=1)
            first_trainer.fit(first_stage, train_dataloaders=loader)
            checkpoint = root / "first" / "interrupted.ckpt"
            first_trainer.save_checkpoint(checkpoint)
            serialized = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(len(serialized["optimizer_states"]), 2)
            self.assertEqual(len(serialized["lr_schedulers"]), 2)

            resumed = fresh_module()
            resumed_trainer = trainer(root / "resumed", max_epochs=2)
            resumed_trainer.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)

            self.assertTensorTreeEqual(
                uninterrupted.model.state_dict(), resumed.model.state_dict(), "model_state"
            )
            self.assertTensorTreeEqual(
                [optimizer.state_dict() for optimizer in uninterrupted_trainer.optimizers],
                [optimizer.state_dict() for optimizer in resumed_trainer.optimizers],
                "optimizer_states",
            )
            self.assertTensorTreeEqual(
                [config.scheduler.state_dict() for config in uninterrupted_trainer.lr_scheduler_configs],
                [config.scheduler.state_dict() for config in resumed_trainer.lr_scheduler_configs],
                "scheduler_states",
            )

    def test_actual_dense_mwm_steps_both_optimizers_and_resumes_exactly(self) -> None:
        generator = torch.Generator().manual_seed(1771)
        batches = [
            {
                "pixels": torch.rand(2, 3, 3, 8, 8, generator=generator),
                "action": torch.randn(2, 3, 2, generator=generator),
            }
        ]

        def fresh_module() -> _ActualDenseIsolatedSPTModule:
            torch.manual_seed(771)
            return _ActualDenseIsolatedSPTModule(
                _lewm_matryoshka_model(
                    K=(4, 8),
                    D=8,
                    action_dim=2,
                    image_shape=(8, 8),
                    normalize_imagenet=False,
                )
            )

        def trainer(root: Path, *, max_epochs: int) -> pl.Trainer:
            return pl.Trainer(
                accelerator="cpu",
                devices=1,
                precision="32-true",
                max_epochs=max_epochs,
                gradient_clip_val=None,
                default_root_dir=root,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                deterministic=True,
            )

        loader = DataLoader(_FixedBatches(batches), batch_size=None, shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uninterrupted = fresh_module()
            uninterrupted_trainer = trainer(root / "uninterrupted", max_epochs=2)
            uninterrupted_trainer.fit(uninterrupted, train_dataloaders=loader)
            self.assertEqual(len(uninterrupted_trainer.optimizers), 2)
            self.assertTrue(all(optimizer.state for optimizer in uninterrupted_trainer.optimizers))

            first_stage = fresh_module()
            first_trainer = trainer(root / "first", max_epochs=1)
            first_trainer.fit(first_stage, train_dataloaders=loader)
            checkpoint = root / "first" / "actual_dense_interrupted.ckpt"
            first_trainer.save_checkpoint(checkpoint)
            serialized = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(len(serialized["optimizer_states"]), 2)
            self.assertEqual(len(serialized["lr_schedulers"]), 2)

            resumed = fresh_module()
            resumed_trainer = trainer(root / "resumed", max_epochs=2)
            resumed_trainer.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)
            self.assertTensorTreeEqual(
                uninterrupted.model.state_dict(), resumed.model.state_dict(), "dense_model_state"
            )
            self.assertTensorTreeEqual(
                [optimizer.state_dict() for optimizer in uninterrupted_trainer.optimizers],
                [optimizer.state_dict() for optimizer in resumed_trainer.optimizers],
                "dense_optimizer_states",
            )
            self.assertTensorTreeEqual(
                [config.scheduler.state_dict() for config in uninterrupted_trainer.lr_scheduler_configs],
                [config.scheduler.state_dict() for config in resumed_trainer.lr_scheduler_configs],
                "dense_scheduler_states",
            )


if __name__ == "__main__":
    unittest.main()
