from __future__ import annotations

from typing import Any

import torch

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


__all__ = [
    "mwm_spt_forward",
]
