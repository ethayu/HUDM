from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mwm.models.world_model import MWMWorldModel, mwm_prediction_loss


def mwm_spt_forward(
    model: MWMWorldModel,
    batch: dict[str, torch.Tensor],
    *,
    level: int | None = None,
    loss_cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    cfg = dict(loss_cfg or {})
    if getattr(model, "eval_only", False):
        raise RuntimeError("Eval-only imported checkpoints cannot be used for MWM training.")
    if hasattr(model, "training_loss"):
        return model.training_loss(
            batch,
            level_weights=cfg.get("level_weights"),
            rollout_weight=float(cfg.get("rollout_weight", 1.0)),
            sigreg=cfg.get("sigreg_module"),
            sigreg_weight=float(cfg.get("sigreg_weight", 0.0)),
            sigreg_scope=str(cfg.get("sigreg_scope", cfg.get("regularizers", "shared_latent"))),
        )
    return mwm_prediction_loss(
        model,
        batch,
        level=level,
        level_weights=cfg.get("level_weights"),
        recon_weight=float(cfg.get("recon_weight", 0.0)),
        rollout_weight=float(cfg.get("rollout_weight", 1.0)),
        sigreg_weight=float(cfg.get("sigreg_weight", 0.0)),
        sigreg_knots=int(cfg.get("sigreg_knots", 17)),
        sigreg_num_proj=int(cfg.get("sigreg_num_proj", 1024)),
    )


class StablePretrainingMWMModule(nn.Module):
    """Tiny training wrapper with Stable-Pretraining-like step methods."""

    def __init__(self, model: MWMWorldModel, loss_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.model = model
        self.loss_cfg = dict(loss_cfg or {})

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return mwm_spt_forward(self.model, batch, loss_cfg=self.loss_cfg)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int = 0) -> torch.Tensor:
        del batch_idx
        return self.forward(batch)["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int = 0) -> dict[str, torch.Tensor]:
        del batch_idx
        return self.forward(batch)


def build_stable_pretraining_module(
    model: MWMWorldModel,
    *,
    loss_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Build an MWM training object while keeping Stable-Pretraining optional."""

    try:
        from stable_pretraining import Module as SPTModule  # type: ignore
    except Exception:
        return StablePretrainingMWMModule(model, loss_cfg=loss_cfg)

    class MWMSPTModule(SPTModule):  # pragma: no cover - depends on optional package API
        def __init__(self) -> None:
            super().__init__(**kwargs)
            self.model = model
            self.loss_cfg = dict(loss_cfg or {})

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return mwm_spt_forward(self.model, batch, loss_cfg=self.loss_cfg)

    return MWMSPTModule()


def build_stable_sigreg(model: MWMWorldModel, **kwargs: Any) -> Any:
    try:
        from stable_worldmodel.wm.loss import SIGReg  # type: ignore
    except Exception as exc:
        raise ImportError("stable-worldmodel SIGReg is not installed in this environment.") from exc
    del model
    return SIGReg(**kwargs)


__all__ = [
    "StablePretrainingMWMModule",
    "build_stable_pretraining_module",
    "build_stable_sigreg",
    "mwm_spt_forward",
]
