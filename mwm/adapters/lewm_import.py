from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.adapters.lewm_common import ImageNetPreprocess
from mwm.models.world_model import MWMWorldModel


class LeWMObjectEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x, interpolate_pos_encoding=True)
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return self.projector(out)


class LeWMObjectDynamics(nn.Module):
    def __init__(self, action_encoder: nn.Module, predictor: nn.Module, pred_proj: nn.Module | None = None) -> None:
        super().__init__()
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.pred_proj = pred_proj or nn.Identity()

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pred = self.predictor(z.unsqueeze(1), self.action_encoder(a.reshape(a.shape[0], 1, -1)))[:, -1]
        return self.pred_proj(pred)


class ImportedLeWMMWMWorldModel(MWMWorldModel):
    """MWM wrapper that preserves trusted upstream Le-WM inference semantics.

    The trainable MWM interface is single-step, but upstream Le-WM planning
    uses the object's own autoregressive ``get_cost`` implementation.  Delegating
    converted checkpoints through that path keeps the canonical checkpoint while
    avoiding a silent change in predictor context handling.
    """

    def __init__(
        self,
        source_model: nn.Module,
        *,
        K: Sequence[int],
        D: int,
        action_dim: int,
        action_block: int,
        image_shape: Sequence[int],
        normalize_imagenet: bool,
        metadata: dict[str, Any],
    ) -> None:
        preprocess = ImageNetPreprocess() if normalize_imagenet else None
        super().__init__(
            encoder=LeWMObjectEncoder(source_model.encoder, getattr(source_model, "projector", None)),
            K=tuple(int(k) for k in K),
            D=int(D),
            action_dim=int(action_dim),
            dynamics=[
                LeWMObjectDynamics(
                    source_model.action_encoder,
                    source_model.predictor,
                    getattr(source_model, "pred_proj", None),
                )
            ],
            preprocess=preprocess,
            metadata=metadata,
        )
        self.__dict__["source_model"] = source_model
        self.source_model.eval()
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval_only = True
        self.action_block = int(action_block)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.source_history_size = int(getattr(getattr(source_model, "predictor", None), "num_frames", 3))

    def train(self, mode: bool = True) -> "ImportedLeWMMWMWorldModel":
        if mode:
            raise RuntimeError("Imported upstream Le-WM checkpoints are eval-only; train a fresh MWM adapter instead.")
        return super().train(False)

    def _normalize_source_images_once(self, infos: dict[str, Any]) -> None:
        if not self.normalize_imagenet or infos.get("_mwm_lewm_source_normalized"):
            return
        ref = next(self.parameters())
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=ref.dtype, device=ref.device)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=ref.dtype, device=mean.device)
        for key in ("pixels", "goal"):
            value = infos.get(key)
            if not torch.is_tensor(value):
                continue
            x = value.to(device=mean.device, dtype=ref.dtype)
            if x.numel() and float(x.detach().min().cpu().item()) < -0.5:
                infos[key] = x
                continue
            if x.numel() and float(x.detach().max().cpu().item()) > 2.0:
                x = x / 255.0
            view_shape = [1] * x.ndim
            view_shape[-3] = 3
            infos[key] = (x - mean.view(*view_shape)) / std.view(*view_shape)
        infos["_mwm_lewm_source_normalized"] = True

    def _ensure_source_goal_emb(self, infos: dict[str, Any]) -> None:
        if "goal_emb" in infos or not hasattr(self.source_model, "encode"):
            return
        goal: dict[str, Any] = {}
        for key, value in infos.items():
            if torch.is_tensor(value):
                goal[key] = value[:, 0]
        if "goal" not in goal:
            return
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        encoded = self.source_model.encode(goal)
        emb = encoded["emb"]
        if emb.ndim == 2:
            emb = emb[:, None, None, :]
        elif emb.ndim == 3:
            emb = emb[:, None, :, :]
        elif emb.ndim != 4:
            raise ValueError(f"Source Le-WM goal embedding must be 2D, 3D, or 4D; got {tuple(emb.shape)}")
        infos["goal_emb"] = emb

    @torch.no_grad()
    def get_cost_with_fidelity(self, infos: dict[str, Any], candidates: torch.Tensor, decision: Any) -> torch.Tensor:
        can_delegate = (
            hasattr(self.source_model, "get_cost")
            and len(self.K) == 1
            and int(getattr(decision, "base_level_idx", 0)) == 0
            and all(int(x) == 0 for x in getattr(decision, "rollout_level_indices", [0] * int(candidates.shape[2])))
        )
        if not can_delegate:
            return super().get_cost_with_fidelity(infos, candidates, decision)
        self._normalize_source_images_once(infos)
        self._ensure_source_goal_emb(infos)
        costs = self.source_model.get_cost(infos, candidates)
        self._last_cost_diagnostics = {
            "base_level_idx": 0,
            "rollout_level_indices": [0 for _ in range(int(candidates.shape[2]))],
            "latent_work": int(candidates.shape[0] * candidates.shape[1] * candidates.shape[2] * self.D),
            "terminal_k": int(self.D),
            "delegated_source_cost": True,
            "source_history_size": int(self.source_history_size),
        }
        return costs


class LeWMObjectImporter:
    DEFAULT_EXPECTED_CLASS = "stable_worldmodel.wm.lewm.lewm.LeWM"

    def __init__(
        self,
        object_checkpoint: str,
        *,
        D: int = 192,
        K: Sequence[int] = (192,),
        action_dim: int,
        action_block: int = 1,
        image_shape: Sequence[int] = (224, 224),
        normalize_imagenet: bool = True,
        expected_class_name: str | None = DEFAULT_EXPECTED_CLASS,
    ) -> None:
        self.object_checkpoint = str(object_checkpoint)
        self.D = int(D)
        self.K = tuple(int(k) for k in K)
        self.action_dim = int(action_dim)
        self.action_block = int(action_block)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.expected_class_name = str(expected_class_name or self.DEFAULT_EXPECTED_CLASS)

    def _load_object(self) -> nn.Module:
        obj = torch.load(self.object_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise TypeError(f"Expected torch.nn.Module in {self.object_checkpoint}, got {type(obj).__name__}")
        missing = [name for name in ("encoder", "predictor", "action_encoder") if not hasattr(obj, name)]
        if missing:
            raise ValueError(f"Trusted Le-WM object checkpoint is missing components: {missing}")
        if self.expected_class_name:
            class_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
            allowed = {self.expected_class_name, self.expected_class_name.rsplit(".", 1)[-1]}
            if class_name not in allowed and type(obj).__name__ not in allowed:
                raise ValueError(
                    f"Trusted Le-WM object checkpoint has class {class_name!r}; expected {self.expected_class_name!r}."
                )
        return obj

    def import_model(self) -> MWMWorldModel:
        if self.K != (self.D,):
            raise ValueError("Trusted object import currently supports single-fidelity Le-WM checkpoints only.")
        obj = self._load_object()
        return mwm_from_lewm_object(
            obj,
            source_checkpoint=self.object_checkpoint,
            D=self.D,
            K=self.K,
            action_dim=self.action_dim,
            action_block=self.action_block,
            image_shape=self.image_shape,
            normalize_imagenet=self.normalize_imagenet,
            expected_class_name=self.expected_class_name,
        )


def mwm_from_lewm_object(
    obj: nn.Module,
    *,
    source_checkpoint: str,
    D: int,
    K: Sequence[int],
    action_dim: int,
    action_block: int = 1,
    image_shape: Sequence[int] = (224, 224),
    normalize_imagenet: bool = True,
    expected_class_name: str | None = None,
) -> MWMWorldModel:
    missing = [name for name in ("encoder", "predictor", "action_encoder") if not hasattr(obj, name)]
    if missing:
        raise ValueError(f"Trusted Le-WM object is missing components: {missing}")
    k_tuple = tuple(int(k) for k in K)
    d = int(D)
    if k_tuple != (d,):
        raise ValueError("Trusted object import currently supports single-fidelity Le-WM checkpoints only.")
    image_shape = tuple(int(x) for x in image_shape)
    metadata = {
        "adapter": "lewm",
        "source": "stable_worldmodel_object",
        "source_checkpoint": str(source_checkpoint),
        "source_class": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "expected_source_class": expected_class_name,
        "action_block": int(action_block),
        "image_shape": list(image_shape),
        "normalize_imagenet": bool(normalize_imagenet),
        "action_preprocessing": "standard_scaler",
        "source_history_size": int(getattr(getattr(obj, "predictor", None), "num_frames", 3)),
        "preprocessing_spec": {
            "image": "imagenet" if bool(normalize_imagenet) else "identity",
            "layout": "BCHW",
            "image_shape": list(image_shape),
        },
        "action_spec": {
            "dim": int(action_dim),
            "base_dim": int(action_dim) // max(1, int(action_block)),
            "block": int(action_block),
        },
    }
    model = ImportedLeWMMWMWorldModel(
        obj,
        K=k_tuple,
        D=d,
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=image_shape,
        normalize_imagenet=bool(normalize_imagenet),
        metadata=metadata,
    )
    model.mwm_config = {
        "target": "mwm.adapters.lewm.build_mwm_lewm_from_object",
        "kwargs": {
            "object_checkpoint": str(source_checkpoint),
            "D": d,
            "K": list(k_tuple),
            "action_dim": int(action_dim),
            "action_block": int(action_block),
            "image_shape": list(image_shape),
            "normalize_imagenet": bool(normalize_imagenet),
            "expected_class_name": expected_class_name,
        },
    }
    return model


def build_mwm_lewm_from_object(**kwargs: Any) -> MWMWorldModel:
    return LeWMObjectImporter(**kwargs).import_model()


__all__ = [
    "ImportedLeWMMWMWorldModel",
    "LeWMObjectDynamics",
    "LeWMObjectEncoder",
    "LeWMObjectImporter",
    "build_mwm_lewm_from_object",
    "mwm_from_lewm_object",
]
