from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.adapters.lewm_common import ImageNetPreprocess
from mwm.models.world_model import MWMWorldModel, matryoshka_base_loss


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
    """Le-WM base adapter with fresh per-K transition heads."""

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
        sigreg_scope: str = "shared_latent",
    ) -> dict[str, torch.Tensor]:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        emb = self._encode_pixels(batch["pixels"], already_preprocessed=True)
        actions = batch["action"]
        levels = list(range(self.num_levels))
        pred_losses: list[torch.Tensor] = []
        for level_idx in levels:
            k = self.K[level_idx]
            pred_emb = self._predict_prefix(
                level_idx,
                emb[:, : self.history_size, :k],
                actions[:, : self.history_size],
            )
            tgt_emb = emb[:, self.num_preds :, :k].detach()
            pred_losses.append((pred_emb - tgt_emb).pow(2).mean())

        return matryoshka_base_loss(
            pred_losses,
            latents=emb,
            K=self.K,
            level_weights=level_weights,
            primary_log_prefix="pred_loss",
            primary_aliases=("pred_loss", "rollout_loss"),
            rollout_weight=rollout_weight,
            regularizer=sigreg,
            regularizer_weight=sigreg_weight,
            regularizer_scope=sigreg_scope,
        )

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
            init.pop("action", None)
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


__all__ = ["LeWMMatryoshkaWorldModel", "LeWMTransitionPackage"]
