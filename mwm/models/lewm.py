from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.diagnostics.flops import (
    FLOP_ACCOUNTING_DYNAMICS_AUDIT,
    decision_flop_accounting,
    profile_dynamics_call,
)
from mwm.models.common import MatryoshkaRuntimeModel
from mwm.models.objectives import matryoshka_training_loss
from mwm.models.planning_costs import (
    active_rollout_levels,
    rollout_schedule_indices,
)
from mwm.models.transitions import TransitionPackage
from mwm.preprocessing.images import ImageNetPreprocess, image_tensor_to_bchw, maybe_apply_image_preprocess


class LeWMMatryoshkaWorldModel(MatryoshkaRuntimeModel):
    """Le-WM runtime with either legacy per-level or opt-in shared dynamics."""

    architecture_version = "lewm_base_adapter_v1"

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        transitions: Sequence[TransitionPackage],
        decoders: Sequence[nn.Module],
        K: Sequence[int],
        D: int,
        action_dim: int,
        action_block: int,
        image_shape: Sequence[int],
        normalize_imagenet: bool,
        history_size: int,
        num_preds: int,
        head_architectures: Sequence[dict[str, Any]],
        decoder_architectures: Sequence[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        architecture_version: str | None = None,
        shared_transition: nn.Module | None = None,
        shared_dynamics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.transitions = nn.ModuleList(list(transitions))
        if shared_transition is not None:
            self.shared_transition = shared_transition
        self.decoders = nn.ModuleList(list(decoders))
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
        self.decoder_architectures = [dict(x) for x in decoder_architectures or []]
        self.shared_dynamics = dict(shared_dynamics or {})
        self.supports_arbitrary_k = shared_transition is not None
        self.min_k = (
            int(self.shared_dynamics.get("min_k", 1))
            if self.supports_arbitrary_k
            else (min(self.K) if self.K else 1)
        )
        if self.supports_arbitrary_k and self.transitions:
            raise ValueError("Shared Le-WM dynamics cannot also register per-level transitions.")
        if not self.supports_arbitrary_k and len(self.K) != len(self.transitions):
            raise ValueError(f"K has {len(self.K)} entries but transitions has {len(self.transitions)}.")
        if len(self.K) != len(self.decoders):
            raise ValueError(f"K has {len(self.K)} entries but decoders has {len(self.decoders)}.")
        if not self.K:
            raise ValueError("K must contain at least one fidelity level.")
        if any(k <= 0 or k > self.D for k in self.K):
            raise ValueError(f"All K values must be in [1, D={self.D}], got {self.K}.")
        if len(set(self.K)) != len(self.K):
            raise ValueError(f"K values must be unique, got {self.K}.")
        if self.supports_arbitrary_k:
            if self.K != sorted(self.K):
                raise ValueError(f"Shared-dynamics K anchors must be sorted, got {self.K}.")
            if self.K[-1] != self.D:
                raise ValueError(f"Shared-dynamics K anchors must include D={self.D}, got {self.K}.")
            if self.min_k <= 0 or self.min_k > self.D:
                raise ValueError(f"shared_dynamics.min_k must be in [1, D={self.D}], got {self.min_k}.")
            if any(k < self.min_k for k in self.K):
                raise ValueError(
                    f"Shared-dynamics anchors must lie in [{self.min_k}, {self.D}], got {self.K}."
                )
        arch = str(architecture_version or self.architecture_version)
        self.architecture_version = arch
        meta = dict(metadata or {})
        meta.setdefault("architecture_version", arch)
        meta.setdefault("image_shape", [int(x) for x in self.image_shape])
        meta.setdefault("action_block", int(self.action_block))
        meta.setdefault("action_dim", int(self.action_dim) // max(1, int(self.action_block)))
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
        meta.setdefault("decoder_architectures", self.decoder_architectures)
        meta.setdefault("levels", list(self.K))
        if self.supports_arbitrary_k:
            meta.setdefault("shared_dynamics", dict(self.shared_dynamics))
            meta.setdefault("supported_k", {"min": int(self.min_k), "max": int(self.D), "arbitrary": True})
        self.metadata = meta
        self._last_cost_diagnostics: dict[str, Any] = {}

    def _maybe_preprocess_eval_pixels(self, pixels: torch.Tensor, *, already_preprocessed: bool) -> torch.Tensor:
        return maybe_apply_image_preprocess(
            pixels,
            self.preprocess,
            already_preprocessed=already_preprocessed,
        )

    def _encode_pixels(self, pixels: torch.Tensor, *, already_preprocessed: bool = False) -> torch.Tensor:
        if pixels.ndim < 4:
            raise ValueError(f"World model pixels must end with image dimensions, got {tuple(pixels.shape)}")
        original_shape = tuple(pixels.shape[:-3])
        flat = pixels.reshape(-1, *pixels.shape[-3:])
        flat = image_tensor_to_bchw(flat)
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
            raise TypeError(f"Unsupported encoder output type {type(out).__name__}")
        emb = self.projector(emb)
        return emb.reshape(*original_shape, self.D)

    def encode(self, info: dict[str, torch.Tensor] | torch.Tensor, *, already_preprocessed: bool = False) -> Any:
        if torch.is_tensor(info):
            return self._encode_pixels(info, already_preprocessed=already_preprocessed)
        out = dict(info)
        out["emb"] = self._encode_pixels(out["pixels"], already_preprocessed=already_preprocessed)
        if "action" in out:
            if self.supports_arbitrary_k:
                out["act_emb"] = self.shared_transition.action_encoder(out["action"], k=self.D)
            else:
                level_idx = self.K.index(self.D) if self.D in self.K else len(self.K) - 1
                out["act_emb"] = self.transitions[level_idx].action_encoder(out["action"])
        return out

    def _validated_k(self, k: int) -> int:
        value = int(k)
        if value < int(self.min_k) or value > self.D:
            raise ValueError(f"K={value} is outside supported range [{self.min_k}, {self.D}].")
        return value

    def predict_at_k(
        self,
        emb: torch.Tensor,
        action: torch.Tensor,
        k: int | None = None,
    ) -> torch.Tensor:
        if not self.supports_arbitrary_k:
            raise RuntimeError("predict_at_k requires the opt-in shared slimmable Le-WM dynamics architecture.")
        inferred = int(emb.shape[-1])
        active_k = self._validated_k(inferred if k is None else int(k))
        if inferred != active_k:
            raise ValueError(f"Latent input width {inferred} does not match requested K={active_k}.")
        return self.shared_transition.predict(emb, action)

    def _predict_prefix(self, level_idx: int, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        k = self.K[int(level_idx)]
        if self.supports_arbitrary_k:
            return self.predict_at_k(emb[..., :k], action, k=k)
        return self.transitions[int(level_idx)].predict(emb[..., :k], action)

    def decode(self, level_idx: int, latent: torch.Tensor) -> torch.Tensor:
        idx = int(level_idx)
        k = self.K[idx]
        return self.decoders[idx](latent[..., :k])

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        level_weights: Sequence[float] | None = None,
        rollout_weight: float = 1.0,
        recon_latent_weight: float = 0.0,
        sigreg: nn.Module | None = None,
        sigreg_weight: float = 0.0,
        sigreg_scope: str = "shared_latent",
        random_prefix_weight: float = 0.0,
        sample_random_prefixes: bool = False,
    ) -> dict[str, torch.Tensor]:
        return matryoshka_training_loss(
            self,
            batch,
            level_weights=level_weights,
            rollout_weight=rollout_weight,
            recon_latent_weight=recon_latent_weight,
            sigreg=sigreg,
            sigreg_weight=sigreg_weight,
            sigreg_scope=sigreg_scope,
            random_prefix_weight=random_prefix_weight,
            sample_random_prefixes=sample_random_prefixes,
        )

    def rollout_at_level(self, infos: dict[str, Any], action_sequence: torch.Tensor, level_idx: int) -> dict[str, Any]:
        if self.supports_arbitrary_k:
            return self.rollout_at_k(infos, action_sequence, self.K[int(level_idx)])
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

    def rollout_with_schedule(
        self,
        infos: dict[str, Any],
        action_sequence: torch.Tensor,
        rollout_levels: Sequence[int],
        *,
        flop_accounting: str = "none",
    ) -> dict[str, Any]:
        if self.supports_arbitrary_k:
            levels = [int(x) for x in rollout_levels]
            for idx in levels:
                if idx < 0 or idx >= self.num_levels:
                    raise ValueError(f"rollout level {idx} is outside [0, {self.num_levels - 1}].")
            return self.rollout_with_k_schedule(
                infos,
                action_sequence,
                [self.K[idx] for idx in levels],
                flop_accounting=flop_accounting,
            )
        if "pixels" not in infos:
            raise KeyError("pixels not in info_dict")
        pixels = infos["pixels"]
        history = int(pixels.size(2))
        batch, samples, horizon = action_sequence.shape[:3]
        if horizon < history:
            raise ValueError(f"Action horizon {horizon} is shorter than pixel history {history}.")
        active_levels = active_rollout_levels([int(x) for x in rollout_levels], horizon=int(horizon), history=history)
        if "emb" not in infos:
            init = {k: v[:, 0] for k, v in infos.items() if torch.is_tensor(v)}
            init.pop("action", None)
            init = self.encode(init, already_preprocessed=False)
            infos["emb"] = init["emb"].detach().unsqueeze(1).expand(batch, samples, -1, -1)
        emb_init = infos["emb"][..., : self.D].reshape(batch * samples, history, self.D)
        all_actions = action_sequence.reshape(batch * samples, int(horizon), self.action_dim)
        emb_list = list(emb_init.unbind(dim=1))
        profile_flops = str(flop_accounting) == FLOP_ACCOUNTING_DYNAMICS_AUDIT
        dynamics_flops = 0
        flop_errors: list[str] = []
        for step, level_idx in enumerate(active_levels):
            level_idx = int(level_idx)
            k = self.K[level_idx]
            pred_time = history + step
            lo = max(0, pred_time - self.history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_actions[:, lo:pred_time]
            pred, flop_count, flop_error = profile_dynamics_call(
                lambda: self._predict_prefix(level_idx, emb_trunc[..., :k], act_trunc),
                enabled=profile_flops,
            )
            dynamics_flops += int(flop_count)
            if flop_error is not None:
                flop_errors.append(flop_error)
            pred_k = pred[:, -1]
            next_emb = emb_list[-1].clone()
            next_emb[..., :k] = pred_k
            emb_list.append(next_emb)
        emb = torch.stack(emb_list, dim=1).reshape(batch, samples, history + len(active_levels), self.D)
        infos["predicted_emb"] = emb
        infos["_mwm_dynamics_flops"] = int(dynamics_flops)
        infos["_mwm_flop_accounting"] = str(flop_accounting)
        if flop_errors:
            infos["_mwm_flop_audit_error"] = "; ".join(flop_errors)
        return infos

    def _validate_k_schedule(self, rollout_ks: Sequence[int], *, horizon: int) -> list[int]:
        values = [self._validated_k(int(k)) for k in rollout_ks]
        if len(values) != int(horizon):
            raise ValueError(f"Expected rollout_ks to have horizon={int(horizon)} entries, got {len(values)}.")
        for previous, current in zip(values, values[1:]):
            if current > previous:
                raise ValueError(f"rollout cannot increase K within one rollout: {values}.")
        return values

    def rollout_at_k(
        self,
        infos: dict[str, Any],
        action_sequence: torch.Tensor,
        k: int,
        *,
        flop_accounting: str = "none",
    ) -> dict[str, Any]:
        if not self.supports_arbitrary_k:
            raise RuntimeError("rollout_at_k requires the opt-in shared slimmable Le-WM dynamics architecture.")
        horizon = int(action_sequence.shape[2])
        active_k = self._validated_k(int(k))
        return self.rollout_with_k_schedule(
            infos,
            action_sequence,
            [active_k] * horizon,
            flop_accounting=flop_accounting,
        )

    def rollout_with_k_schedule(
        self,
        infos: dict[str, Any],
        action_sequence: torch.Tensor,
        rollout_ks: Sequence[int],
        *,
        flop_accounting: str = "none",
    ) -> dict[str, Any]:
        if not self.supports_arbitrary_k:
            raise RuntimeError(
                "rollout_with_k_schedule requires the opt-in shared slimmable Le-WM dynamics architecture."
            )
        if "pixels" not in infos:
            raise KeyError("pixels not in info_dict")
        pixels = infos["pixels"]
        history = int(pixels.size(2))
        batch, samples, horizon = action_sequence.shape[:3]
        if horizon < history:
            raise ValueError(f"Action horizon {horizon} is shorter than pixel history {history}.")
        schedule = self._validate_k_schedule(rollout_ks, horizon=int(horizon))
        active_ks = active_rollout_levels(schedule, horizon=int(horizon), history=history)
        if "emb" not in infos:
            init = {key: value[:, 0] for key, value in infos.items() if torch.is_tensor(value)}
            init.pop("action", None)
            init = self.encode(init, already_preprocessed=False)
            infos["emb"] = init["emb"].detach().unsqueeze(1).expand(batch, samples, -1, -1)
        emb_init = infos["emb"][..., : self.D].reshape(batch * samples, history, self.D)
        all_actions = action_sequence.reshape(batch * samples, int(horizon), self.action_dim)
        emb_list = list(emb_init.unbind(dim=1))
        profile_flops = str(flop_accounting) == FLOP_ACCOUNTING_DYNAMICS_AUDIT
        dynamics_flops = 0
        flop_errors: list[str] = []
        for step, k in enumerate(active_ks):
            pred_time = history + step
            lo = max(0, pred_time - self.history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_actions[:, lo:pred_time]
            pred, flop_count, flop_error = profile_dynamics_call(
                lambda k=int(k): self.predict_at_k(emb_trunc[..., :k], act_trunc, k=k),
                enabled=profile_flops,
            )
            dynamics_flops += int(flop_count)
            if flop_error is not None:
                flop_errors.append(flop_error)
            next_emb = emb_list[-1].clone()
            next_emb[..., : int(k)] = pred[:, -1]
            emb_list.append(next_emb)
        infos["predicted_emb"] = torch.stack(emb_list, dim=1).reshape(
            batch, samples, history + len(active_ks), self.D
        )
        infos["_mwm_dynamics_flops"] = int(dynamics_flops)
        infos["_mwm_flop_accounting"] = str(flop_accounting)
        infos["_mwm_rollout_ks"] = [int(k) for k in schedule]
        if flop_errors:
            infos["_mwm_flop_audit_error"] = "; ".join(flop_errors)
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
        rollout_ks_value = getattr(decision, "rollout_ks", None)
        if self.supports_arbitrary_k and rollout_ks_value is not None:
            base_k = self._validated_k(int(getattr(decision, "base_k")))
            rollout_ks = self._validate_k_schedule(rollout_ks_value, horizon=int(candidates.shape[2]))
            if any(k > base_k for k in rollout_ks):
                raise ValueError(f"rollout K cannot exceed base K={base_k}: {rollout_ks}.")
            base_level_idx = self.K.index(base_k) if base_k in self.K else None
            rollout_levels = [self.K.index(k) if k in self.K else None for k in rollout_ks]
        else:
            legacy_base_idx, legacy_levels = rollout_schedule_indices(
                decision,
                int(candidates.shape[2]),
                num_levels=self.num_levels,
            )
            base_level_idx = int(legacy_base_idx)
            rollout_levels = [int(x) for x in legacy_levels]
            base_k = self.K[base_level_idx]
            rollout_ks = [self.K[idx] for idx in rollout_levels]
        if int(candidates.shape[-1]) != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {int(candidates.shape[-1])}")
        flop_accounting = decision_flop_accounting(decision)
        self._ensure_goal_emb(infos)
        if self.supports_arbitrary_k:
            out = self.rollout_with_k_schedule(infos, candidates, rollout_ks, flop_accounting=flop_accounting)
        else:
            out = self.rollout_with_schedule(infos, candidates, rollout_levels, flop_accounting=flop_accounting)
        pred_emb = out["predicted_emb"]
        goal_emb = out["goal_emb"]
        history = int(infos["pixels"].size(2))
        active_ks = active_rollout_levels(rollout_ks, horizon=int(candidates.shape[2]), history=history)
        k = int(active_ks[-1])
        terminal_idx = self.K.index(k) if k in self.K else None
        if goal_emb.ndim == 2:
            goal_emb = goal_emb[:, None, None, :]
        elif goal_emb.ndim == 3:
            goal_emb = goal_emb[:, None, :, :]
        goal_emb = goal_emb[..., -1:, :k].expand_as(pred_emb[..., -1:, :k])
        cost = (pred_emb[..., -1:, :k] - goal_emb.detach()).pow(2).sum(dim=tuple(range(2, pred_emb.ndim)))
        self._last_cost_diagnostics = {
            "base_level_idx": int(base_level_idx) if base_level_idx is not None else None,
            "terminal_level_idx": int(terminal_idx) if terminal_idx is not None else None,
            "rollout_level_indices": rollout_levels,
            "base_k": int(base_k),
            "terminal_k": int(k),
            "rollout_ks": [int(value) for value in rollout_ks],
            "latent_work": int(candidates.shape[0]) * int(candidates.shape[1]) * sum(int(value) for value in active_ks),
            "dynamics_flops": int(out.get("_mwm_dynamics_flops", 0)),
            "flop_accounting": str(out.get("_mwm_flop_accounting", flop_accounting)),
            "prefix_criterion": True,
            "history_size": int(self.history_size),
        }
        if "_mwm_flop_audit_error" in out:
            self._last_cost_diagnostics["flop_audit_error"] = str(out["_mwm_flop_audit_error"])
        return cost


__all__ = ["LeWMMatryoshkaWorldModel"]
