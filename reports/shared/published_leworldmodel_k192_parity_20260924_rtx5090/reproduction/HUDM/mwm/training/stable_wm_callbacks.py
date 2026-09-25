from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


def stable_wm_adapter_checkpoint_callback(cfg: Any) -> ModelCheckpoint:
    checkpoint_steps = int(cfg.train.get("checkpoint_every_n_train_steps", 0) or 0)
    monitor = cfg.train.get("checkpoint_monitor", None)
    checkpoint_kwargs: dict[str, Any] = {"save_last": True}
    if monitor:
        if checkpoint_steps > 0:
            raise ValueError(
                "train.checkpoint_monitor uses validation metrics; set "
                "train.checkpoint_every_n_train_steps=0 so checkpointing runs after validation."
            )
        checkpoint_kwargs.update(
            {
                "monitor": str(monitor),
                "mode": str(cfg.train.get("checkpoint_mode", "min")),
                "save_top_k": int(cfg.train.get("save_top_k", 1)),
            }
        )
    else:
        checkpoint_kwargs["save_top_k"] = int(cfg.train.get("save_top_k", 0))
    if checkpoint_steps > 0:
        checkpoint_kwargs.update({"every_n_train_steps": checkpoint_steps, "every_n_epochs": 0})
    return ModelCheckpoint(**checkpoint_kwargs)


class AllLevelPlateauEarlyStopping(Callback):
    def __init__(
        self,
        *,
        metrics: list[str],
        patience: int,
        min_delta: float = 0.0,
        relative_min_delta: float = 0.0,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        if not metrics:
            raise ValueError("All-level convergence requires at least one metric.")
        if patience < 1:
            raise ValueError(f"patience must be positive, got {patience}.")
        self.metrics = list(metrics)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.relative_min_delta = float(relative_min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.best: dict[str, float] = {}
        self.wait_count = 0
        self.stopped_epoch: int | None = None

    @staticmethod
    def _to_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        if hasattr(value, "compute"):
            computed = value.compute()
            if torch.is_tensor(computed):
                return float(computed.detach().cpu().item())
            return float(computed)
        return float(value)

    @staticmethod
    def _metric_candidates(metric: str) -> list[str]:
        if metric.endswith("_epoch"):
            return [metric, metric[: -len("_epoch")]]
        return [metric, f"{metric}_epoch"]

    def _metric_value(self, callback_metrics: Any, metric: str) -> float | None:
        for candidate in self._metric_candidates(metric):
            if candidate in callback_metrics:
                return self._to_float(callback_metrics[candidate])
        return None

    def _improved(self, metric: str, value: float) -> bool:
        if metric not in self.best:
            return True
        threshold = max(self.min_delta, abs(self.best[metric]) * self.relative_min_delta)
        return value < self.best[metric] - threshold

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if bool(getattr(trainer, "sanity_checking", False)):
            return

        values: dict[str, float] = {}
        missing: list[str] = []
        for metric in self.metrics:
            value = self._metric_value(trainer.callback_metrics, metric)
            if value is None:
                missing.append(metric)
            else:
                values[metric] = value
        if missing:
            raise RuntimeError(f"Convergence early stopping missing validation metrics: {missing}")

        any_improved = False
        for metric, value in values.items():
            if self._improved(metric, value):
                self.best[metric] = value
                any_improved = True

        epoch = int(getattr(trainer, "current_epoch", 0))
        if epoch < self.warmup_epochs:
            self.wait_count = 0
            return
        if any_improved:
            self.wait_count = 0
            return
        self.wait_count += 1
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True


class TrainingHealthSummary(Callback):
    """Emit compact validation metrics for Slurm log scraping."""

    DEFAULT_SCALARS = (
        "validate/loss",
        "validate/rollout_loss",
        "validate/pred_loss",
        "validate/recon_loss",
        "validate/sigreg_loss",
    )

    def __init__(
        self,
        *,
        levels: list[int] | tuple[int, ...],
        metrics: list[str] | tuple[str, ...] | None = None,
        monitor: str = "validate/rollout_loss",
        prefix: str = "[mwm-health]",
    ) -> None:
        super().__init__()
        self.levels = [int(level) for level in levels]
        self.metrics = list(metrics) if metrics is not None else self._default_metrics()
        self.monitor = str(monitor)
        self.prefix = str(prefix)
        self.best_value: float | None = None

    def _default_metrics(self) -> list[str]:
        per_level = [f"validate/pred_loss_l{idx}" for idx, _ in enumerate(self.levels)]
        return [*self.DEFAULT_SCALARS[:3], *per_level, *self.DEFAULT_SCALARS[3:]]

    @staticmethod
    def _to_float(value: Any) -> float:
        return AllLevelPlateauEarlyStopping._to_float(value)

    @staticmethod
    def _metric_candidates(metric: str) -> list[str]:
        return AllLevelPlateauEarlyStopping._metric_candidates(metric)

    def _metric_value(self, callback_metrics: Any, metric: str) -> float | None:
        for candidate in self._metric_candidates(metric):
            if candidate in callback_metrics:
                return self._to_float(callback_metrics[candidate])
        return None

    def _metric_label(self, metric: str) -> str:
        prefix = "validate/pred_loss_l"
        if metric.startswith(prefix):
            suffix = metric[len(prefix) :]
            if suffix.isdigit():
                index = int(suffix)
                if index < len(self.levels):
                    return f"{metric}(K={self.levels[index]})"
        return metric

    def _monitor_value(self, callback_metrics: Any) -> float | None:
        for metric in (self.monitor, "validate/pred_loss", "validate/loss"):
            value = self._metric_value(callback_metrics, metric)
            if value is not None:
                return value
        return None

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if bool(getattr(trainer, "sanity_checking", False)):
            return

        callback_metrics = getattr(trainer, "callback_metrics", {})
        parts: list[str] = []
        for metric in self.metrics:
            value = self._metric_value(callback_metrics, metric)
            if value is not None:
                parts.append(f"{self._metric_label(metric)}={value:.6g}")
        if not parts:
            return

        monitor_value = self._monitor_value(callback_metrics)
        best = False
        if monitor_value is not None and (self.best_value is None or monitor_value < self.best_value):
            self.best_value = monitor_value
            best = True

        epoch = int(getattr(trainer, "current_epoch", 0))
        print(f"{self.prefix} epoch={epoch} best={str(best).lower()} {' '.join(parts)}", flush=True)


def stable_wm_adapter_callbacks(cfg: Any) -> list[Callback]:
    callbacks: list[Callback] = [stable_wm_adapter_checkpoint_callback(cfg)]
    health_cfg = cfg.train.get("health_summary", True)
    health_enabled = bool(health_cfg)
    if hasattr(health_cfg, "get"):
        health_enabled = bool(health_cfg.get("enabled", True))
    if health_enabled:
        health_monitor = "validate/rollout_loss"
        health_metrics = None
        if hasattr(health_cfg, "get"):
            health_monitor = str(health_cfg.get("monitor", health_monitor))
            configured_metrics = health_cfg.get("metrics", None)
            if configured_metrics is not None:
                health_metrics = list(configured_metrics)
        callbacks.append(
            TrainingHealthSummary(
                levels=list(cfg.model.K),
                metrics=health_metrics,
                monitor=health_monitor,
            )
        )
    convergence_cfg = cfg.schedule.get("convergence", None)
    if convergence_cfg is not None and bool(convergence_cfg.get("enabled", False)):
        default_metrics = [f"validate/pred_loss_l{idx}" for idx, _ in enumerate(cfg.model.K)]
        callbacks.append(
            AllLevelPlateauEarlyStopping(
                metrics=list(convergence_cfg.get("metrics", default_metrics)),
                patience=int(convergence_cfg.get("patience", 5)),
                min_delta=float(convergence_cfg.get("min_delta", 0.0)),
                relative_min_delta=float(convergence_cfg.get("relative_min_delta", 0.0)),
                warmup_epochs=int(convergence_cfg.get("warmup_epochs", 0)),
            )
        )
    return callbacks


def select_stable_wm_adapter_export_checkpoint(checkpoint_cb: Any, cfg: Any) -> str | None:
    policy = str(cfg.train.get("export_checkpoint", "last")).lower()
    best_path = str(getattr(checkpoint_cb, "best_model_path", "") or "")
    last_path = str(getattr(checkpoint_cb, "last_model_path", "") or "")
    if policy in {"last", "final"}:
        return last_path or None
    if policy == "best":
        if not best_path:
            raise ValueError("train.export_checkpoint=best requested, but no best checkpoint was saved.")
        return best_path
    if policy in {"best_if_available", "best-or-last", "best_or_last"}:
        return best_path or last_path or None
    raise ValueError(
        "train.export_checkpoint must be one of: last, final, best, best_if_available, best-or-last, best_or_last."
    )


__all__ = [
    "AllLevelPlateauEarlyStopping",
    "TrainingHealthSummary",
    "stable_wm_adapter_callbacks",
    "stable_wm_adapter_checkpoint_callback",
    "select_stable_wm_adapter_export_checkpoint",
]
