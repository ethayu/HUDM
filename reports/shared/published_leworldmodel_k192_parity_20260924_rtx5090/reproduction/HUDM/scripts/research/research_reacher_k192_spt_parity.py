from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

# PyTorch requires this to be set before creating a CUDA context when
# deterministic cuBLAS execution is requested by Trainer(deterministic=True).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightning as pl
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from mwm.config_cli import load_config
from mwm.training.stable_wm_config import DEFAULTS, as_container
from mwm.training.stable_wm_data import close_dataset_handles, prepare_stable_wm_adapter_context
from mwm.training.stable_wm_lightning import (
    OptimizerIsolatedSPTModule,
    stable_wm_adapter_forward,
    stable_wm_parameter_partitions,
)
from mwm.training.stable_wm_model import build_trainable_stable_wm_adapter_model


WORLD_LOSS_KEYS = (
    "loss",
    "world_loss",
    "pred_loss",
    "pred_loss_l0",
    "rollout_loss",
    "sigreg_loss",
)

WORLD_PARITY_TRACE_KEYS = (
    "step",
    "batch_sha256",
    "rng_before",
    "learning_rate_before",
    "world_losses",
    "raw_world_gradients",
    "clipped_world_gradients",
    "learning_rate_after",
    "world_parameters",
)


def _sha256_bytes(parts: Iterable[bytes | memoryview]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    cpu = tensor.detach().contiguous().cpu()
    return memoryview(cpu.reshape(-1).view(torch.uint8).numpy())


def _tensor_digest(tensor: torch.Tensor) -> str:
    return _sha256_bytes((_tensor_bytes(tensor),))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_tensor_snapshot(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    *,
    gradients: bool,
) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []
    none_names: list[str] = []
    dtype: str | None = None
    for name, parameter in named_parameters:
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            none_names.append(name)
            continue
        tensor = tensor.detach().reshape(-1)
        if dtype is None:
            dtype = str(tensor.dtype)
        elif str(tensor.dtype) != dtype:
            raise RuntimeError(f"Mixed tensor dtypes in parity snapshot: {dtype} and {tensor.dtype}.")
        tensors.append(tensor)
    if not tensors:
        return {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "numel": 0,
            "dtype": None,
            "none_names": none_names,
            "l2_norm": 0.0,
            "max_abs": 0.0,
        }
    flat = torch.cat(tensors)
    flat_float = flat.float()
    snapshot = {
        "sha256": _tensor_digest(flat),
        "numel": int(flat.numel()),
        "dtype": dtype,
        "none_names": none_names,
        "l2_norm": float(torch.linalg.vector_norm(flat_float).detach().cpu().item()),
        "max_abs": float(flat_float.abs().max().detach().cpu().item()),
    }
    del flat, flat_float
    return snapshot


def _batch_digest(batch: dict[str, torch.Tensor]) -> str:
    parts: list[bytes | memoryview] = []
    for key in ("pixels", "action"):
        tensor = batch[key]
        parts.extend((key.encode(), str(tuple(tensor.shape)).encode(), str(tensor.dtype).encode(), _tensor_bytes(tensor)))
    return _sha256_bytes(parts)


def _rng_state() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def _set_rng_state(state: tuple[torch.Tensor, torch.Tensor]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)


def _rng_digest(state: tuple[torch.Tensor, torch.Tensor]) -> dict[str, str]:
    cpu_state, cuda_state = state
    return {"cpu": _tensor_digest(cpu_state), "cuda": _tensor_digest(cuda_state)}


def _loss_snapshot(output: dict[str, torch.Tensor], *, decoder_enabled: bool) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in WORLD_LOSS_KEYS:
        if key == "loss" and decoder_enabled:
            tensor = output["world_loss"]
        else:
            tensor = output.get(key)
        if tensor is None:
            continue
        detached = tensor.detach().cpu()
        values[key] = {
            "value": float(detached.float().item()),
            "dtype": str(detached.dtype),
            "sha256": _tensor_digest(detached),
        }
    return values


def _assert_equal(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: keys differ: {set(expected) ^ set(actual)}")
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: lengths differ: {len(expected)} != {len(actual)}")
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            _assert_equal(expected_value, actual_value, f"{path}[{index}]")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def _validate_trace_complete(trace: list[dict[str, Any]], *, steps: int, label: str) -> None:
    if len(trace) != steps:
        raise RuntimeError(f"{label} completed {len(trace)} steps, expected {steps}.")
    required = {
        "step",
        "batch_sha256",
        "rng_before",
        "rng_after_forward",
        "learning_rate_before",
        "world_losses",
        "raw_world_gradients",
        "clipped_world_gradients",
        "learning_rate_after",
        "world_parameters",
    }
    for step, record in enumerate(trace):
        missing = required - set(record)
        if missing:
            raise RuntimeError(f"{label} step {step} is missing trace fields: {sorted(missing)}.")
        if int(record["step"]) != step:
            raise RuntimeError(f"{label} trace is out of order at index {step}: {record['step']}.")
        for key in ("raw_world_gradients", "clipped_world_gradients", "world_parameters"):
            snapshot = record[key]
            if int(snapshot["numel"]) <= 0:
                raise RuntimeError(f"{label} step {step} captured no tensors for {key}.")


def _assert_world_parity_trace(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> None:
    if len(expected) != len(actual):
        raise AssertionError(f"trace: lengths differ: {len(expected)} != {len(actual)}")
    for step, (expected_record, actual_record) in enumerate(zip(expected, actual)):
        for key in WORLD_PARITY_TRACE_KEYS:
            _assert_equal(expected_record[key], actual_record[key], f"step[{step}].{key}")


def _parity_forward(module: Any, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    if stage != "fit":
        return stable_wm_adapter_forward(module, batch, stage)
    step = int(batch["batch_idx"])
    if module.parity_expected_trace is None:
        before_rng = _rng_state()
        module.parity_rng_states.append(before_rng)
    else:
        before_rng = module.parity_expected_rng_states[step]
        _set_rng_state(before_rng)
    record = {
        "step": step,
        "batch_sha256": _batch_digest(batch),
        "rng_before": _rng_digest(_rng_state()),
        "learning_rate_before": module._parity_world_lr(),
    }
    output = stable_wm_adapter_forward(module, batch, stage)
    record["rng_after_forward"] = _rng_digest(_rng_state())
    record["world_losses"] = _loss_snapshot(output, decoder_enabled=module.parity_decoder_enabled)
    module.parity_trace.append(record)
    if module.parity_expected_trace is not None:
        expected = module.parity_expected_trace[step]
        # Decoder-side work may legitimately advance RNG after the world graph
        # has been evaluated. Keep the post-forward state as a diagnostic, but
        # gate parity on the identical state restored before each world forward.
        for key in ("batch_sha256", "rng_before", "learning_rate_before", "world_losses"):
            _assert_equal(expected[key], record[key], f"step[{step}].{key}")
    return output


class _ParityTraceMixin:
    parity_decoder_enabled: bool
    parity_expected_trace: list[dict[str, Any]] | None
    parity_expected_rng_states: list[tuple[torch.Tensor, torch.Tensor]] | None

    def _initialize_parity_trace(
        self,
        *,
        decoder_enabled: bool,
        expected_trace: list[dict[str, Any]] | None = None,
        expected_rng_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.parity_decoder_enabled = decoder_enabled
        self.parity_expected_trace = expected_trace
        self.parity_expected_rng_states = expected_rng_states
        self.parity_trace: list[dict[str, Any]] = []
        self.parity_rng_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._parity_named_world_parameters: tuple[tuple[str, torch.nn.Parameter], ...] = ()
        self._parity_world_parameter_ids: set[int] = set()

    def on_train_start(self) -> None:
        super().on_train_start()
        world_parameters, decoder_parameters = stable_wm_parameter_partitions(self.model)
        self._parity_world_parameter_ids = {id(parameter) for parameter in world_parameters}
        self._parity_named_world_parameters = tuple(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if id(parameter) in self._parity_world_parameter_ids
        )
        if {id(parameter) for _, parameter in self._parity_named_world_parameters} != self._parity_world_parameter_ids:
            raise RuntimeError("Named world parameters do not match the asserted partition.")
        if self.parity_decoder_enabled and not decoder_parameters:
            raise RuntimeError("Current parity path has no decoder parameters.")

    def _parity_world_optimizer(self) -> Any:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        matches = []
        for optimizer in optimizers:
            raw_optimizer = getattr(optimizer, "optimizer", optimizer)
            optimizer_ids = {id(parameter) for group in raw_optimizer.param_groups for parameter in group["params"]}
            if self._parity_world_parameter_ids.issubset(optimizer_ids):
                matches.append(optimizer)
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one world optimizer, found {len(matches)}.")
        return matches[0]

    def _parity_world_lr(self) -> list[float]:
        optimizer = self._parity_world_optimizer()
        return [float(group["lr"]) for group in optimizer.param_groups]

    def after_manual_backward(self) -> None:
        super().after_manual_backward()
        step = len(self.parity_trace) - 1
        snapshot = _ordered_tensor_snapshot(self._parity_named_world_parameters, gradients=True)
        self.parity_trace[step]["raw_world_gradients"] = snapshot
        if self.parity_expected_trace is not None:
            _assert_equal(
                self.parity_expected_trace[step]["raw_world_gradients"],
                snapshot,
                f"step[{step}].raw_world_gradients",
            )

    def clip_gradients(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        super().clip_gradients(optimizer, *args, **kwargs)
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        optimizer_ids = {id(parameter) for group in raw_optimizer.param_groups for parameter in group["params"]}
        if not self._parity_world_parameter_ids.issubset(optimizer_ids):
            return
        step = len(self.parity_trace) - 1
        snapshot = _ordered_tensor_snapshot(self._parity_named_world_parameters, gradients=True)
        self.parity_trace[step]["clipped_world_gradients"] = snapshot
        if self.parity_expected_trace is not None:
            _assert_equal(
                self.parity_expected_trace[step]["clipped_world_gradients"],
                snapshot,
                f"step[{step}].clipped_world_gradients",
            )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        step = int(batch_idx)
        record = self.parity_trace[step]
        record["learning_rate_after"] = self._parity_world_lr()
        record["world_parameters"] = _ordered_tensor_snapshot(
            self._parity_named_world_parameters,
            gradients=False,
        )
        if self.parity_expected_trace is not None:
            for key in ("learning_rate_after", "world_parameters"):
                _assert_equal(self.parity_expected_trace[step][key], record[key], f"step[{step}].{key}")
        super().on_train_batch_end(outputs, batch, batch_idx)


class _HistoricalWorldOnlyModule(_ParityTraceMixin, spt.Module):
    pass


class _CurrentIsolatedDecoderModule(_ParityTraceMixin, OptimizerIsolatedSPTModule):
    pass


def _scheduler(total_steps: int) -> dict[str, Any]:
    return {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": max(1, int(0.01 * total_steps)),
        "max_steps": total_steps,
    }


def _historical_optim(cfg: Any, total_steps: int) -> dict[str, Any]:
    return {
        "model_opt": {
            "modules": "model",
            "optimizer": {
                "type": "AdamW",
                "lr": float(cfg.optim.lr),
                "weight_decay": float(cfg.optim.get("weight_decay", 0.0)),
            },
            "scheduler": _scheduler(total_steps),
            "interval": "epoch",
        }
    }


def _current_optim(cfg: Any, total_steps: int) -> dict[str, Any]:
    decoder = cfg.decoder_training
    return {
        "decoder_opt": {
            "modules": r"^model\.decoders(?:\.|$)",
            "optimizer": {
                "type": "AdamW",
                "lr": float(decoder.lr),
                "weight_decay": float(decoder.weight_decay),
            },
            "scheduler": _scheduler(total_steps),
            "interval": "epoch",
        },
        "world_opt": {
            "modules": r"^model(?:\.|$)",
            "optimizer": {
                "type": "AdamW",
                "lr": float(cfg.optim.lr),
                "weight_decay": float(cfg.optim.get("weight_decay", 0.0)),
            },
            "scheduler": _scheduler(total_steps),
            "interval": "epoch",
        },
    }


def _trainer(root: Path, *, gradient_clip_val: float | None) -> pl.Trainer:
    return pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=1,
        gradient_clip_val=gradient_clip_val,
        default_root_dir=str(root),
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
        use_distributed_sampler=False,
    )


def _validate_config(cfg: Any, steps: int) -> None:
    expected = {
        "seed": 3072,
        "env_id": "swm/ReacherDMControl-v0",
        "K": [192],
        "D": 192,
        "batch_size": 128,
        "sigreg_weight": 0.09,
        "sigreg_knots": 17,
        "sigreg_num_proj": 1024,
        "max_epochs": 10,
    }
    actual = {
        "seed": int(cfg.seed),
        "env_id": str(cfg.env_id),
        "K": [int(value) for value in cfg.model.K],
        "D": int(cfg.model.D),
        "batch_size": int(cfg.train.batch_size),
        "sigreg_weight": float(cfg.loss.sigreg_weight),
        "sigreg_knots": int(cfg.loss.sigreg_knots),
        "sigreg_num_proj": int(cfg.loss.sigreg_num_proj),
        "max_epochs": int(cfg.schedule.max_epochs),
    }
    if actual != expected:
        raise ValueError(f"Parity config mismatch: expected {expected}, got {actual}.")
    if not bool(cfg.decoder_training.enabled) or str(cfg.decoder_training.mode) != "separate_optimizer":
        raise ValueError("Current parity path requires enabled separate-optimizer decoder training.")
    if float(cfg.train.gradient_clip_val) != 1.0 or float(cfg.decoder_training.gradient_clip_val) != 1.0:
        raise ValueError("Parity requires world and decoder gradient clipping at 1.0.")
    if steps < 100:
        raise ValueError(f"Parity requires at least 100 steps, got {steps}.")


def _parameter_manifest(model: torch.nn.Module) -> list[dict[str, Any]]:
    world_parameters, _ = stable_wm_parameter_partitions(model)
    world_ids = {id(parameter) for parameter in world_parameters}
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
            "sha256": _tensor_digest(parameter),
        }
        for name, parameter in model.named_parameters()
        if id(parameter) in world_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Reacher K=192 SPT optimizer-isolation parity gate.")
    parser.add_argument("--config", default="configs/train/mwm_lewm_reacher_upstream.yaml")
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Reacher parity requires CUDA because SIGReg samples projections on CUDA.")
    cfg = load_config(DEFAULTS, args.config, [])
    _validate_config(cfg, args.steps)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(int(cfg.seed), workers=True)

    train_set, _, base_dataset, model_cfg, _ = prepare_stable_wm_adapter_context(cfg)
    batch_size = int(cfg.train.batch_size)
    full_steps_per_epoch = len(train_set) // batch_size
    total_steps = int(cfg.schedule.max_epochs) * full_steps_per_epoch
    required_samples = int(args.steps) * batch_size
    if required_samples > len(train_set):
        raise ValueError(
            f"Parity requires {required_samples} fixed samples for {args.steps} full batches, "
            f"but the Reacher train split contains only {len(train_set)}."
        )
    index_generator = torch.Generator().manual_seed(int(cfg.seed))
    fixed_indices = torch.randperm(len(train_set), generator=index_generator)[:required_samples].tolist()
    fixed_dataset = Subset(train_set, fixed_indices)

    def loader() -> DataLoader:
        return DataLoader(
            fixed_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=int(cfg.train.num_workers),
            persistent_workers=False,
            prefetch_factor=int(cfg.train.prefetch_factor),
            pin_memory=True,
        )

    from stable_worldmodel.wm.loss import SIGReg

    initial_model = build_trainable_stable_wm_adapter_model(cfg, model_cfg)
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
    initial_manifest = _parameter_manifest(initial_model)

    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.decoder_training.enabled = False
    reference = _HistoricalWorldOnlyModule(
        model=initial_model,
        sigreg=SIGReg(knots=int(cfg.loss.sigreg_knots), num_proj=int(cfg.loss.sigreg_num_proj)),
        forward=_parity_forward,
        optim=_historical_optim(cfg, total_steps),
        hparams=as_container(reference_cfg),
    )
    reference.stable_wm_adapter_cfg = reference_cfg
    reference._initialize_parity_trace(decoder_enabled=False)
    reference_trainer = _trainer(output_dir / "reference_trainer", gradient_clip_val=1.0)
    reference_trainer.fit(reference, train_dataloaders=loader())
    _validate_trace_complete(reference.parity_trace, steps=args.steps, label="Reference")
    reference_trace = copy.deepcopy(reference.parity_trace)
    reference_rng_states = [(cpu.clone(), cuda.clone()) for cpu, cuda in reference.parity_rng_states]
    del reference_trainer, reference, initial_model
    gc.collect()
    torch.cuda.empty_cache()

    pl.seed_everything(int(cfg.seed), workers=True)
    candidate_model = build_trainable_stable_wm_adapter_model(cfg, model_cfg)
    candidate_model.load_state_dict(initial_state, strict=True)
    candidate_manifest = _parameter_manifest(candidate_model)
    _assert_equal(initial_manifest, candidate_manifest, "initial_world_parameter_manifest")
    world_parameters, decoder_parameters = stable_wm_parameter_partitions(candidate_model)
    candidate = _CurrentIsolatedDecoderModule(
        model=candidate_model,
        sigreg=SIGReg(knots=int(cfg.loss.sigreg_knots), num_proj=int(cfg.loss.sigreg_num_proj)),
        forward=_parity_forward,
        optim=_current_optim(cfg, total_steps),
        hparams=as_container(cfg),
        world_parameters=world_parameters,
        decoder_parameters=decoder_parameters,
        world_gradient_clip_val=float(cfg.train.gradient_clip_val),
        decoder_gradient_clip_val=float(cfg.decoder_training.gradient_clip_val),
    )
    candidate.stable_wm_adapter_cfg = cfg
    candidate._initialize_parity_trace(
        decoder_enabled=True,
        expected_trace=reference_trace,
        expected_rng_states=reference_rng_states,
    )
    candidate_trainer = _trainer(output_dir / "candidate_trainer", gradient_clip_val=None)
    candidate_trainer.fit(candidate, train_dataloaders=loader())
    _validate_trace_complete(candidate.parity_trace, steps=args.steps, label="Candidate")
    _assert_world_parity_trace(reference_trace, candidate.parity_trace)

    report = {
        "status": "pass",
        "comparison": "exact",
        "stable_pretraining_version": importlib.metadata.version("stable-pretraining"),
        "stable_pretraining_module_source": inspect.getsourcefile(spt.Module),
        "stable_pretraining_module_sha256": _file_sha256(inspect.getsourcefile(spt.Module)),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _file_sha256(args.config),
        "config": as_container(cfg),
        "steps": int(args.steps),
        "seed": int(cfg.seed),
        "precision": "bf16-mixed",
        "dataset": str(cfg.data.path),
        "fixed_train_subset_indices": fixed_indices,
        "full_steps_per_epoch": full_steps_per_epoch,
        "scheduler_total_steps": total_steps,
        "scheduler_warmup_steps": max(1, int(0.01 * total_steps)),
        "world_parameter_count": sum(item["numel"] for item in initial_manifest),
        "world_parameter_manifest": initial_manifest,
        "trace": reference_trace,
        "candidate_rng_after_forward": [record["rng_after_forward"] for record in candidate.parity_trace],
    }
    (output_dir / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "PASS").write_text("exact real Reacher K=192 parity passed\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "comparison", "steps", "seed", "precision", "world_parameter_count", "scheduler_total_steps")}, indent=2))
    close_dataset_handles(base_dataset)


if __name__ == "__main__":
    main()
