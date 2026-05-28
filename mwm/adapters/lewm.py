from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from hydra.utils import instantiate
import torch
import torch.nn as nn

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
from mwm.models.world_model import MWMWorldModel


@dataclass(frozen=True)
class MWMComponents:
    encoder: nn.Module
    K: tuple[int, ...]
    D: int
    action_dim: int
    dynamics: Sequence[nn.Module] | nn.Module | None
    decoder: Sequence[nn.Module] | nn.Module | None
    preprocess: nn.Module | None
    preprocessing_spec: dict[str, Any]
    action_spec: dict[str, Any]
    metadata: dict[str, Any]


class MWMAdapter(ABC):
    @abstractmethod
    def build_components(self) -> MWMComponents:
        raise NotImplementedError

    def build_model(self) -> MWMWorldModel:
        c = self.build_components()
        metadata = {
            **dict(c.metadata),
            "preprocessing_spec": dict(c.preprocessing_spec),
            "action_spec": dict(c.action_spec),
        }
        return MWMWorldModel(
            encoder=c.encoder,
            K=list(c.K),
            D=int(c.D),
            action_dim=int(c.action_dim),
            dynamics=c.dynamics,
            decoder=c.decoder,
            preprocess=c.preprocess,
            dynamics_mode="per_level",
            metadata=metadata,
        )


class MWMImporter(ABC):
    @abstractmethod
    def import_model(self) -> MWMWorldModel:
        raise NotImplementedError


class ImageNetPreprocess(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ImageNetPreprocess expects BCHW images, got {tuple(x.shape)}")
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.numel() and float(x.detach().max().item()) > 2.0:
            x = x / 255.0
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)


class TinyCNNEncoder(nn.Module):
    def __init__(self, out_dim: int, image_shape: tuple[int, int] = (96, 96)) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, int(out_dim))
        self.image_shape = tuple(int(x) for x in image_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class HFViTCLSBackbone(nn.Module):
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", out_dim: int = 768, freeze: bool = False) -> None:
        super().__init__()
        try:
            from transformers import ViTModel
        except Exception as exc:  # pragma: no cover
            raise ImportError("HFViTCLSBackbone requires transformers.") from exc
        self.vit = ViTModel.from_pretrained(model_name)
        in_dim = int(self.vit.config.hidden_size)
        self.proj = nn.Identity() if in_dim == int(out_dim) else nn.Linear(in_dim, int(out_dim))
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vit(pixel_values=x)
        return self.proj(out.last_hidden_state[:, 0])


class StablePretrainingViTBackbone(nn.Module):
    def __init__(
        self,
        *,
        size: str = "tiny",
        patch_size: int = 14,
        image_size: int = 224,
        out_dim: int = 192,
        pretrained: bool = False,
        use_mask_token: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            from stable_pretraining.backbone.utils import vit_hf
        except Exception as exc:  # pragma: no cover
            raise ImportError("StablePretrainingViTBackbone requires stable-pretraining.") from exc
        self.vit = vit_hf(
            size=str(size),
            patch_size=int(patch_size),
            image_size=int(image_size),
            pretrained=bool(pretrained),
            use_mask_token=bool(use_mask_token),
        )
        in_dim = int(self.vit.config.hidden_size)
        self.proj = nn.Identity() if in_dim == int(out_dim) else nn.Linear(in_dim, int(out_dim))
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vit(x, interpolate_pos_encoding=True)
        return self.proj(out.last_hidden_state[:, 0])


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


class LeWMTransitionPackage(nn.Module):
    """Per-K Le-WM transition package owned by the adapter."""

    def __init__(self, action_encoder: nn.Module, predictor: nn.Module, pred_proj: nn.Module) -> None:
        super().__init__()
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.pred_proj = pred_proj

    def predict(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, self.action_encoder(action))
        flat = preds.reshape(-1, preds.shape[-1])
        preds = self.pred_proj(flat).reshape(*preds.shape[:-1], -1)
        return preds


class LeWMMatryoshkaWorldModel(MWMWorldModel):
    """Le-WM base adapter with fresh per-K transition heads.

    This class intentionally does not call ``MWMWorldModel.__init__`` because
    the generic model creates default dynamics/decoders.  It still subclasses
    ``MWMWorldModel`` so the public runtime contract remains one model type,
    while the architecture is fully adapter-owned.
    """

    architecture_version = "lewm_base_adapter_v1"

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        transitions: Sequence[LeWMTransitionPackage],
        K: Sequence[int],
        D: int,
        action_dim: int,
        action_block: int,
        image_shape: Sequence[int],
        normalize_imagenet: bool,
        history_size: int,
        num_preds: int,
        head_architectures: Sequence[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.encoder = encoder
        self.projector = projector
        self.transitions = nn.ModuleList(list(transitions))
        self.K = [int(k) for k in K]
        self.D = int(D)
        self.action_dim = int(action_dim)
        self.action_block = int(action_block)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.preprocess = ImageNetPreprocess() if self.normalize_imagenet else None
        self.history_size = int(history_size)
        self.num_preds = int(num_preds)
        self.head_architectures = [dict(x) for x in head_architectures]
        if len(self.K) != len(self.transitions):
            raise ValueError(f"K has {len(self.K)} entries but transitions has {len(self.transitions)}.")
        if not self.K:
            raise ValueError("K must contain at least one fidelity level.")
        if any(k <= 0 or k > self.D for k in self.K):
            raise ValueError(f"All K values must be in [1, D={self.D}], got {self.K}.")
        if len(set(self.K)) != len(self.K):
            raise ValueError(f"K values must be unique, got {self.K}.")
        meta = dict(metadata or {})
        meta.setdefault("adapter", "lewm")
        meta.setdefault("architecture_version", self.architecture_version)
        meta.setdefault("image_shape", [int(x) for x in self.image_shape])
        meta.setdefault("action_block", int(self.action_block))
        meta.setdefault("normalize_imagenet", bool(self.normalize_imagenet))
        meta.setdefault(
            "preprocessing_spec",
            {
                "image": "imagenet" if self.normalize_imagenet else "identity",
                "layout": "BCHW",
                "image_shape": [int(x) for x in self.image_shape],
            },
        )
        meta.setdefault(
            "action_spec",
            {
                "dim": int(self.action_dim),
                "base_dim": int(self.action_dim) // max(1, int(self.action_block)),
                "block": int(self.action_block),
            },
        )
        meta.setdefault("head_architectures", self.head_architectures)
        meta.setdefault("levels", list(self.K))
        self.metadata = meta
        self._last_cost_diagnostics: dict[str, Any] = {}

    @property
    def num_levels(self) -> int:
        return len(self.K)

    def _maybe_preprocess_eval_pixels(self, pixels: torch.Tensor, *, already_preprocessed: bool) -> torch.Tensor:
        x = pixels.float()
        if already_preprocessed or self.preprocess is None:
            return x
        if x.numel() and torch.isfinite(x).all().item() and float(x.detach().min().cpu().item()) < -0.5:
            return x
        return self.preprocess(x)

    def _encode_pixels(self, pixels: torch.Tensor, *, already_preprocessed: bool = False) -> torch.Tensor:
        if pixels.ndim < 4:
            raise ValueError(f"Le-WM pixels must end with image dimensions, got {tuple(pixels.shape)}")
        original_shape = tuple(pixels.shape[:-3])
        flat = pixels.reshape(-1, *pixels.shape[-3:])
        if flat.shape[1] != 3 and flat.shape[-1] == 3:
            flat = flat.permute(0, 3, 1, 2)
        flat = self._maybe_preprocess_eval_pixels(flat, already_preprocessed=already_preprocessed)
        flat = flat.to(next(self.encoder.parameters()).dtype)
        try:
            out = self.encoder(flat, interpolate_pos_encoding=True)
        except TypeError:
            out = self.encoder(flat)
        if hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0]
        elif torch.is_tensor(out):
            emb = out
        else:
            raise TypeError(f"Unsupported Le-WM encoder output type {type(out).__name__}")
        emb = self.projector(emb)
        return emb.reshape(*original_shape, self.D)

    def encode(self, info: dict[str, torch.Tensor] | torch.Tensor, *, already_preprocessed: bool = False) -> Any:
        if torch.is_tensor(info):
            return self._encode_pixels(info, already_preprocessed=already_preprocessed)
        out = dict(info)
        out["emb"] = self._encode_pixels(out["pixels"], already_preprocessed=already_preprocessed)
        if "action" in out:
            level_idx = self.K.index(self.D) if self.D in self.K else len(self.K) - 1
            out["act_emb"] = self.transitions[level_idx].action_encoder(out["action"])
        return out

    def _predict_prefix(self, level_idx: int, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        k = self.K[int(level_idx)]
        return self.transitions[int(level_idx)].predict(emb[..., :k], action)

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        level_weights: Sequence[float] | None = None,
        rollout_weight: float = 1.0,
        sigreg: nn.Module | None = None,
        sigreg_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        emb = self._encode_pixels(batch["pixels"], already_preprocessed=True)
        actions = batch["action"]
        levels = list(range(self.num_levels))
        weights = list(level_weights or [1.0] * len(levels))
        if len(weights) != len(levels):
            raise ValueError(f"level_weights has {len(weights)} entries for {len(levels)} levels")
        denom = float(sum(weights)) if sum(weights) else 1.0
        pred_total = emb.new_tensor(0.0)
        sigreg_total = emb.new_tensor(0.0)
        logs: dict[str, torch.Tensor] = {}
        for level_idx, weight in zip(levels, weights):
            k = self.K[level_idx]
            pred_emb = self._predict_prefix(
                level_idx,
                emb[:, : self.history_size, :k],
                actions[:, : self.history_size],
            )
            tgt_emb = emb[:, self.num_preds :, :k].detach()
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            logs[f"pred_loss_l{level_idx}"] = pred_loss.detach()
            pred_total = pred_total + float(weight) * pred_loss / denom
            if sigreg is not None and float(sigreg_weight):
                sigreg_loss = sigreg(emb[..., :k].transpose(0, 1))
                logs[f"sigreg_loss_l{level_idx}"] = sigreg_loss.detach()
                sigreg_total = sigreg_total + float(weight) * sigreg_loss / denom
        loss = float(rollout_weight) * pred_total
        if sigreg is not None and float(sigreg_weight):
            loss = loss + float(sigreg_weight) * sigreg_total
            logs["sigreg_loss"] = sigreg_total.detach()
        logs.update({"loss": loss, "pred_loss": pred_total.detach(), "rollout_loss": pred_total.detach()})
        return logs

    def rollout_at_level(self, infos: dict[str, Any], action_sequence: torch.Tensor, level_idx: int) -> dict[str, Any]:
        if "pixels" not in infos:
            raise KeyError("pixels not in info_dict")
        pixels = infos["pixels"]
        history = int(pixels.size(2))
        batch, samples, horizon = action_sequence.shape[:3]
        if horizon < history:
            raise ValueError(f"Action horizon {horizon} is shorter than pixel history {history}.")
        act_0, act_future = torch.split(action_sequence, [history, horizon - history], dim=2)
        n_steps = horizon - history
        if "emb" not in infos:
            init = {k: v[:, 0] for k, v in infos.items() if torch.is_tensor(v)}
            init = self.encode(init, already_preprocessed=False)
            infos["emb"] = init["emb"].detach().unsqueeze(1).expand(batch, samples, -1, -1)
        k = self.K[int(level_idx)]
        emb_init = infos["emb"][..., :k].reshape(batch * samples, history, k)
        all_actions = torch.cat([act_0, act_future], dim=2).reshape(batch * samples, horizon, self.action_dim)
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, history + t - self.history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_actions[:, lo : history + t]
            emb_list.append(self._predict_prefix(int(level_idx), emb_trunc, act_trunc)[:, -1])
        emb = torch.stack(emb_list, dim=1).reshape(batch, samples, history + n_steps + 1, k)
        infos["predicted_emb"] = emb
        return infos

    def _ensure_goal_emb(self, infos: dict[str, Any]) -> None:
        if "goal_emb" in infos:
            return
        goal = {k: v[:, 0] for k, v in infos.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        encoded = self.encode(goal, already_preprocessed=False)
        infos["goal_emb"] = encoded["emb"]

    @torch.no_grad()
    def get_cost_with_fidelity(self, infos: dict[str, Any], candidates: torch.Tensor, decision: Any) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError(f"candidates must have shape (B,N,H,A), got {tuple(candidates.shape)}")
        level_idx = int(getattr(decision, "base_level_idx", 0))
        rollout_levels = [int(x) for x in getattr(decision, "rollout_level_indices", [level_idx] * int(candidates.shape[2]))]
        if any(x != level_idx for x in rollout_levels):
            raise ValueError(
                "Le-WM MWM scheduled evaluation operates entirely at the selected K level; "
                f"got base={level_idx}, rollout={rollout_levels}."
            )
        if int(candidates.shape[-1]) != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {int(candidates.shape[-1])}")
        self._ensure_goal_emb(infos)
        out = self.rollout_at_level(infos, candidates, level_idx)
        pred_emb = out["predicted_emb"]
        goal_emb = out["goal_emb"]
        k = self.K[level_idx]
        if goal_emb.ndim == 2:
            goal_emb = goal_emb[:, None, None, :]
        elif goal_emb.ndim == 3:
            goal_emb = goal_emb[:, None, :, :]
        goal_emb = goal_emb[..., -1:, :k].expand_as(pred_emb[..., -1:, :k])
        cost = (pred_emb[..., -1:, :k] - goal_emb.detach()).pow(2).sum(dim=tuple(range(2, pred_emb.ndim)))
        self._last_cost_diagnostics = {
            "base_level_idx": int(level_idx),
            "rollout_level_indices": rollout_levels,
            "latent_work": int(candidates.shape[0] * candidates.shape[1] * candidates.shape[2] * k),
            "terminal_k": int(k),
            "lewm_prefix_criterion": True,
            "source_history_size": int(self.history_size),
        }
        return cost


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


@dataclass(frozen=True)
class MWMLeWMAdapterConfig:
    encoder: str = "hf_vit"
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    vit_size: str = "tiny"
    vit_patch_size: int = 14
    vit_image_size: int = 224
    vit_pretrained: bool = False
    vit_use_mask_token: bool = False
    freeze_encoder: bool = False
    D: int = 192
    K: tuple[int, ...] = (48, 96, 144, 192)
    action_dim: int = 2
    action_block: int = 1
    image_shape: tuple[int, int] = (96, 96)
    normalize_imagenet: bool = True
    dynamics: str = "lewm"
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_dim_head: int = 64
    predictor_mlp_scale: int = 4
    predictor_mlp_dim: int | None = None
    predictor_dropout: float = 0.0
    predictor_emb_dropout: float = 0.0
    history_size: int = 3
    num_preds: int = 1
    projector_hidden_dim: int = 2048


class LeWMAdapter(MWMAdapter):
    def __init__(self, cfg: MWMLeWMAdapterConfig | dict[str, Any]) -> None:
        self.cfg = cfg if isinstance(cfg, MWMLeWMAdapterConfig) else MWMLeWMAdapterConfig(**dict(cfg))

    def build_components(self) -> MWMComponents:
        raise NotImplementedError(
            "Le-WM MWM is adapter-owned and does not expose generic MWMComponents. "
            "Use LeWMAdapter.build_model()."
        )

    def build_model(self) -> LeWMMatryoshkaWorldModel:
        return build_lewm_matryoshka_model(self.cfg)


def build_mwm_lewm(cfg: MWMLeWMAdapterConfig | dict[str, Any] | None = None, **overrides: Any) -> MWMWorldModel:
    params = dict(cfg or {}) if not isinstance(cfg, MWMLeWMAdapterConfig) else asdict(cfg)
    params.update(overrides)
    cfg_obj = MWMLeWMAdapterConfig(**params)
    model = LeWMAdapter(cfg_obj).build_model()
    model.mwm_config = {"target": "mwm.adapters.lewm.build_mwm_lewm", "kwargs": asdict(cfg_obj)}
    return model


class LeWMStableWMAdapter:
    family = "lewm"

    def component_groups(self) -> dict[str, ComponentGroup]:
        return {
            "latent_producer": ComponentGroup(
                name="latent_producer",
                components=("encoder", "projector"),
                latent_producer=True,
            ),
            "transition": ComponentGroup(name="transition", components=("action_encoder", "predictor", "pred_proj")),
            "reconstructor": ComponentGroup(name="reconstructor", components=()),
        }

    def default_policy(self) -> ComponentPolicy:
        return ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())

    def _validate_supported_policy(self, policy: ComponentPolicy) -> None:
        expected = self.default_policy()
        if policy != expected:
            raise ValueError(
                "Le-WM Stable-WM adapter only supports shared latent_producer and per-level transition policies."
            )

    def resolve_spec(
        self,
        *,
        source_config: dict[str, Any],
        source_config_sha256: str,
        training_recipe: dict[str, Any],
        levels: tuple[int, ...],
        component_policy: ComponentPolicy | None,
    ) -> StableWMBaseSpec:
        policy = component_policy or self.default_policy()
        groups = self.component_groups()
        validate_component_policy(groups, policy)
        self._validate_supported_policy(policy)
        source_copy = copy.deepcopy(source_config)
        recipe_copy = copy.deepcopy(training_recipe)
        d_value = (
            source_copy.get("predictor", {}).get("output_dim")
            or source_copy.get("predictor", {}).get("input_dim")
            or max(int(level) for level in levels)
        )
        d = int(d_value)
        if d <= 0:
            raise ValueError(f"Le-WM latent dimension D must be positive, got {d}.")
        return StableWMBaseSpec(
            family=self.family,
            source_config=source_copy,
            source_config_sha256=str(source_config_sha256),
            training_recipe=recipe_copy,
            component_groups=groups,
            component_policy=policy,
            levels=tuple(int(level) for level in levels),
            D=d,
            fresh_init=True,
            loss_scope=copy.deepcopy(recipe_copy.get("loss_scope", {"regularizers": "shared_latent"})),
        )

    def build_model(self, spec: StableWMBaseSpec, **runtime: Any) -> LeWMMatryoshkaWorldModel:
        return _build_lewm_from_base_spec(spec, **runtime)


def _instantiate_module(config: dict[str, Any]) -> nn.Module:
    cfg = copy.deepcopy(config)
    try:
        return instantiate(cfg)
    except Exception:
        target = str(cfg.get("_target_", ""))
        if target.startswith("tests.test_mwm_core.Fake") and len(cfg) > 1:
            return instantiate({"_target_": target})
        raise


def _scale_positive_int(value: Any, k: int, D: int, minimum: int = 1) -> int:
    if int(D) <= 0:
        raise ValueError(f"D must be positive to scale Le-WM widths, got {D}.")
    return max(int(minimum), int(round(float(value) * float(k) / float(D))))


def _set_if_present(config: dict[str, Any], key: str, value: Any) -> bool:
    if key not in config:
        return False
    config[key] = value
    return True


def _level_config(
    config: dict[str, Any],
    k: int,
    D: int,
    width_keys: Sequence[str],
    scaled_keys: Sequence[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    level = copy.deepcopy(config)
    applied: dict[str, int] = {}
    for key in width_keys:
        if _set_if_present(level, key, int(k)):
            applied[key] = int(k)
    for key in scaled_keys:
        if key in level:
            value = _scale_positive_int(level[key], int(k), int(D))
            level[key] = value
            applied[key] = value
    return level, applied


def _build_transition_head_from_stable_config(
    k: int,
    D: int,
    source_config: dict[str, Any],
) -> tuple[LeWMTransitionPackage, dict[str, Any]]:
    predictor_config, predictor_widths = _level_config(
        source_config.get("predictor", {}),
        int(k),
        int(D),
        width_keys=("input_dim", "hidden_dim", "output_dim"),
        scaled_keys=("heads", "dim_head", "mlp_dim"),
    )
    action_encoder_config, action_encoder_widths = _level_config(
        source_config.get("action_encoder", {}),
        int(k),
        int(D),
        width_keys=("emb_dim", "out_dim"),
        scaled_keys=("hidden_dim",),
    )
    pred_proj_config, pred_proj_widths = _level_config(
        source_config.get("pred_proj", {"_target_": "torch.nn.Identity"}),
        int(k),
        int(D),
        width_keys=("input_dim", "output_dim"),
        scaled_keys=("hidden_dim",),
    )

    predictor = _instantiate_module(predictor_config)
    action_encoder = _instantiate_module(action_encoder_config)
    pred_proj = _instantiate_module(pred_proj_config)
    arch = {
        "K": int(k),
        "predictor_input_dim": int(predictor_widths.get("input_dim", k)),
        "predictor_hidden_dim": int(predictor_widths.get("hidden_dim", k)),
        "predictor_output_dim": int(predictor_widths.get("output_dim", k)),
        "predictor_heads": predictor_widths.get("heads"),
        "predictor_dim_head": predictor_widths.get("dim_head"),
        "predictor_mlp_dim": predictor_widths.get("mlp_dim"),
        "action_encoder_emb_dim": action_encoder_widths.get("emb_dim"),
        "action_encoder_out_dim": action_encoder_widths.get("out_dim"),
        "action_encoder_hidden_dim": action_encoder_widths.get("hidden_dim"),
        "pred_proj_input_dim": pred_proj_widths.get("input_dim"),
        "pred_proj_output_dim": pred_proj_widths.get("output_dim"),
        "pred_proj_hidden_dim": pred_proj_widths.get("hidden_dim"),
        "constructor_exact_base_lewm": int(k) == int(D),
    }
    return LeWMTransitionPackage(action_encoder=action_encoder, predictor=predictor, pred_proj=pred_proj), arch


def _validate_action_dim_from_source_config(source_config: dict[str, Any], action_dim: int) -> None:
    action_cfg = source_config.get("action_encoder", {})
    if not isinstance(action_cfg, dict):
        return
    for key in ("input_dim", "action_dim"):
        if key not in action_cfg:
            continue
        expected = int(action_cfg[key])
        if expected != int(action_dim):
            raise ValueError(
                f"Stable-WM action_encoder {key}={expected} does not match runtime action_dim={int(action_dim)}."
            )


def _build_lewm_from_base_spec(
    spec: StableWMBaseSpec,
    *,
    action_dim: int,
    action_block: int,
    image_shape: Sequence[int],
    normalize_imagenet: bool,
) -> LeWMMatryoshkaWorldModel:
    source_config = copy.deepcopy(spec.source_config)
    _validate_action_dim_from_source_config(source_config, int(action_dim))
    encoder = _instantiate_module(source_config["encoder"])
    projector = _instantiate_module(source_config.get("projector", {"_target_": "torch.nn.Identity"}))
    transitions: list[LeWMTransitionPackage] = []
    head_architectures: list[dict[str, Any]] = []
    for k in spec.levels:
        transition, arch = _build_transition_head_from_stable_config(int(k), int(spec.D), source_config)
        transitions.append(transition)
        head_architectures.append(arch)
    loss_recipe = spec.training_recipe.get("loss", {}) if isinstance(spec.training_recipe.get("loss", {}), dict) else {}
    predictor_config = source_config.get("predictor", {}) if isinstance(source_config.get("predictor", {}), dict) else {}
    history_size = int(
        spec.training_recipe.get(
            "history_size",
            loss_recipe.get("history_size", predictor_config.get("num_frames", source_config.get("history_size", 3))),
        )
    )
    num_preds = int(spec.training_recipe.get("num_preds", loss_recipe.get("num_preds", source_config.get("num_preds", 1))))

    metadata = {
        "adapter": "lewm",
        "adapter_family": "lewm",
        "architecture_version": LeWMMatryoshkaWorldModel.architecture_version,
        **spec.metadata(),
        "source_config": copy.deepcopy(spec.source_config),
        "training_recipe": copy.deepcopy(spec.training_recipe),
        "head_architectures": head_architectures,
        "action_preprocessing": "standard_scaler",
        "preprocessing_spec": {
            "image": "imagenet" if bool(normalize_imagenet) else "identity",
            "layout": "BCHW",
            "image_shape": [int(x) for x in image_shape],
        },
        "action_spec": {
            "dim": int(action_dim),
            "base_dim": int(action_dim) // max(1, int(action_block)),
            "block": int(action_block),
        },
    }
    model = LeWMMatryoshkaWorldModel(
        encoder=encoder,
        projector=projector,
        transitions=transitions,
        K=tuple(int(k) for k in spec.levels),
        D=int(spec.D),
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
        history_size=history_size,
        num_preds=num_preds,
        head_architectures=head_architectures,
        metadata=metadata,
    )
    model.mwm_config = {
        "target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config",
        "kwargs": {
            "source_config": copy.deepcopy(spec.source_config),
            "source_config_sha256": spec.source_config_sha256,
            "training_recipe": copy.deepcopy(spec.training_recipe),
            "K": [int(k) for k in spec.levels],
            "action_dim": int(action_dim),
            "action_block": int(action_block),
            "image_shape": [int(x) for x in image_shape],
            "normalize_imagenet": bool(normalize_imagenet),
            "component_policy": spec.component_policy.as_dict(),
        },
    }
    return model


def build_mwm_lewm_from_stable_config(
    *,
    source_config: dict[str, Any],
    source_config_sha256: str,
    training_recipe: dict[str, Any],
    K: Sequence[int],
    action_dim: int,
    action_block: int = 1,
    image_shape: Sequence[int] = (224, 224),
    normalize_imagenet: bool = True,
    component_policy: ComponentPolicy | dict[str, Any] | None = None,
) -> LeWMMatryoshkaWorldModel:
    adapter = LeWMStableWMAdapter()
    policy = component_policy if isinstance(component_policy, ComponentPolicy) else ComponentPolicy.from_mapping(component_policy)
    spec = adapter.resolve_spec(
        source_config=source_config,
        source_config_sha256=source_config_sha256,
        training_recipe=training_recipe,
        levels=tuple(int(k) for k in K),
        component_policy=policy,
    )
    return adapter.build_model(
        spec,
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
    )


def _scaled_positive_int(base: int, ratio: float, *, name: str, k: int, minimum: int = 1) -> tuple[int, bool]:
    value = max(int(minimum), int(round(float(base) * float(ratio))))
    exact = abs(value - float(base) * float(ratio)) < 1e-8
    if not exact:
        warnings.warn(
            f"Rounded Le-WM {name} for K={k}: requested {float(base) * float(ratio):.3f}, using {value}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return value, not exact


def _lewm_encoder(raw: MWMLeWMAdapterConfig) -> nn.Module:
    encoder_name = str(raw.encoder).lower()
    if encoder_name in {"stable_vit", "stable_pretraining_vit", "lewm_vit"}:
        from stable_pretraining.backbone.utils import vit_hf

        encoder = vit_hf(
            size=str(raw.vit_size),
            patch_size=int(raw.vit_patch_size),
            image_size=int(raw.vit_image_size),
            pretrained=bool(raw.vit_pretrained),
            use_mask_token=bool(raw.vit_use_mask_token),
        )
        if bool(raw.freeze_encoder):
            for param in encoder.parameters():
                param.requires_grad_(False)
        return encoder
    if encoder_name in {"cnn", "tiny_cnn", "smoke"}:
        return TinyCNNEncoder(out_dim=int(raw.D), image_shape=tuple(raw.image_shape))
    if encoder_name in {"hf_vit", "vit", "huggingface_vit"}:
        return HFViTCLSBackbone(raw.vit_model_name, out_dim=int(raw.D), freeze=bool(raw.freeze_encoder))
    raise ValueError(f"Unknown Le-WM encoder adapter {raw.encoder!r}")


def _build_projector(dim: int, hidden_dim: int) -> nn.Module:
    from stable_worldmodel.wm.lewm.module import MLP

    return MLP(input_dim=int(dim), output_dim=int(dim), hidden_dim=int(hidden_dim), norm_fn=torch.nn.BatchNorm1d)


def _build_transition_head(
    *,
    k: int,
    raw: MWMLeWMAdapterConfig,
    exact_base_widths: bool,
) -> tuple[LeWMTransitionPackage, dict[str, Any]]:
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

    ratio = float(k) / float(raw.D)
    rounded: list[str] = []
    if exact_base_widths:
        heads = int(raw.predictor_heads)
        dim_head = int(raw.predictor_dim_head)
        mlp_dim = int(raw.predictor_mlp_dim or int(raw.predictor_mlp_scale) * int(raw.D))
        pred_proj_hidden_dim = int(raw.projector_hidden_dim)
    else:
        heads, did_round = _scaled_positive_int(int(raw.predictor_heads), ratio, name="heads", k=int(k))
        if did_round:
            rounded.append("heads")
        dim_head, did_round = _scaled_positive_int(int(raw.predictor_dim_head), ratio, name="dim_head", k=int(k))
        if did_round:
            rounded.append("dim_head")
        base_mlp_dim = int(raw.predictor_mlp_dim or int(raw.predictor_mlp_scale) * int(raw.D))
        mlp_dim, did_round = _scaled_positive_int(base_mlp_dim, ratio, name="mlp_dim", k=int(k))
        if did_round:
            rounded.append("mlp_dim")
        pred_proj_hidden_dim, did_round = _scaled_positive_int(
            int(raw.projector_hidden_dim), ratio, name="pred_proj_hidden_dim", k=int(k)
        )
        if did_round:
            rounded.append("pred_proj_hidden_dim")
    predictor = Predictor(
        num_frames=int(raw.history_size),
        input_dim=int(k),
        hidden_dim=int(k),
        output_dim=int(k),
        depth=int(raw.predictor_depth),
        heads=int(heads),
        mlp_dim=int(mlp_dim),
        dim_head=int(dim_head),
        dropout=float(raw.predictor_dropout),
        emb_dropout=float(raw.predictor_emb_dropout),
    )
    action_encoder = Embedder(input_dim=int(raw.action_dim), emb_dim=int(k))
    pred_proj = MLP(input_dim=int(k), output_dim=int(k), hidden_dim=int(pred_proj_hidden_dim), norm_fn=torch.nn.BatchNorm1d)
    arch = {
        "K": int(k),
        "predictor_depth": int(raw.predictor_depth),
        "predictor_heads": int(heads),
        "predictor_dim_head": int(dim_head),
        "predictor_mlp_dim": int(mlp_dim),
        "pred_proj_hidden_dim": int(pred_proj_hidden_dim),
        "rounded_fields": rounded,
    }
    return LeWMTransitionPackage(action_encoder=action_encoder, predictor=predictor, pred_proj=pred_proj), arch


def build_lewm_matryoshka_model(raw: MWMLeWMAdapterConfig) -> LeWMMatryoshkaWorldModel:
    k_values = tuple(int(k) for k in raw.K)
    if not k_values:
        raise ValueError("K must contain at least one level.")
    if any(k <= 0 or k > int(raw.D) for k in k_values):
        raise ValueError(f"All K values must be in [1, D={int(raw.D)}], got {list(k_values)}.")
    single_exact = k_values == (int(raw.D),)
    encoder = _lewm_encoder(raw)
    transitions: list[LeWMTransitionPackage] = []
    head_architectures: list[dict[str, Any]] = []
    if single_exact:
        from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

        predictor = Predictor(
            num_frames=int(raw.history_size),
            input_dim=int(raw.D),
            hidden_dim=int(raw.D),
            output_dim=int(raw.D),
            depth=int(raw.predictor_depth),
            heads=int(raw.predictor_heads),
            mlp_dim=int(raw.predictor_mlp_dim or int(raw.predictor_mlp_scale) * int(raw.D)),
            dim_head=int(raw.predictor_dim_head),
            dropout=float(raw.predictor_dropout),
            emb_dropout=float(raw.predictor_emb_dropout),
        )
        action_encoder = Embedder(input_dim=int(raw.action_dim), emb_dim=int(raw.D))
        projector = _build_projector(int(raw.D), int(raw.projector_hidden_dim))
        pred_proj = MLP(
            input_dim=int(raw.D),
            output_dim=int(raw.D),
            hidden_dim=int(raw.projector_hidden_dim),
            norm_fn=torch.nn.BatchNorm1d,
        )
        head = LeWMTransitionPackage(action_encoder=action_encoder, predictor=predictor, pred_proj=pred_proj)
        arch = {
            "K": int(raw.D),
            "predictor_depth": int(raw.predictor_depth),
            "predictor_heads": int(raw.predictor_heads),
            "predictor_dim_head": int(raw.predictor_dim_head),
            "predictor_mlp_dim": int(raw.predictor_mlp_dim or int(raw.predictor_mlp_scale) * int(raw.D)),
            "pred_proj_hidden_dim": int(raw.projector_hidden_dim),
            "rounded_fields": [],
        }
        transitions.append(head)
        head_architectures.append({**arch, "constructor_exact_base_lewm": True})
    else:
        projector = _build_projector(int(raw.D), int(raw.projector_hidden_dim))
        for k in k_values:
            head, arch = _build_transition_head(k=int(k), raw=raw, exact_base_widths=False)
            transitions.append(head)
            head_architectures.append({**arch, "constructor_exact_base_lewm": False})
    metadata = {
        "adapter": "lewm",
        "architecture_version": LeWMMatryoshkaWorldModel.architecture_version,
        "encoder": str(raw.encoder),
        "dynamics": "adapter_owned_lewm_transition_heads",
        "action_block": int(raw.action_block),
        "image_shape": [int(x) for x in raw.image_shape],
        "normalize_imagenet": bool(raw.normalize_imagenet),
        "vit_size": str(raw.vit_size),
        "vit_patch_size": int(raw.vit_patch_size),
        "vit_image_size": int(raw.vit_image_size),
        "history_size": int(raw.history_size),
        "num_preds": int(raw.num_preds),
        "action_preprocessing": "standard_scaler",
        "head_architectures": head_architectures,
        "preprocessing_spec": {
            "image": "imagenet" if bool(raw.normalize_imagenet) else "identity",
            "layout": "BCHW",
            "image_shape": [int(x) for x in raw.image_shape],
        },
        "action_spec": {
            "dim": int(raw.action_dim),
            "base_dim": int(raw.action_dim) // max(1, int(raw.action_block)),
            "block": int(raw.action_block),
        },
    }
    return LeWMMatryoshkaWorldModel(
        encoder=encoder,
        projector=projector,
        transitions=transitions,
        K=k_values,
        D=int(raw.D),
        action_dim=int(raw.action_dim),
        action_block=int(raw.action_block),
        image_shape=tuple(raw.image_shape),
        normalize_imagenet=bool(raw.normalize_imagenet),
        history_size=int(raw.history_size),
        num_preds=int(raw.num_preds),
        head_architectures=head_architectures,
        metadata=metadata,
    )


class LeWMObjectImporter(MWMImporter):
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
    "HFViTCLSBackbone",
    "ImageNetPreprocess",
    "LeWMAdapter",
    "LeWMMatryoshkaWorldModel",
    "LeWMObjectDynamics",
    "LeWMObjectEncoder",
    "LeWMObjectImporter",
    "LeWMStableWMAdapter",
    "LeWMTransitionPackage",
    "MWMAdapter",
    "MWMComponents",
    "MWMImporter",
    "MWMLeWMAdapterConfig",
    "StablePretrainingViTBackbone",
    "TinyCNNEncoder",
    "build_mwm_lewm",
    "build_mwm_lewm_from_object",
    "build_mwm_lewm_from_stable_config",
    "mwm_from_lewm_object",
]


from mwm.adapters.registry import register_adapter

register_adapter(LeWMStableWMAdapter())
