from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.fidelity import FidelityDecision


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


class MWMWorldModel(nn.Module):
    """Common multi-fidelity world model used by MWM train/eval wrappers."""

    def __init__(
        self,
        encoder: nn.Module,
        K: Sequence[int],
        D: int,
        action_dim: int,
        dynamics: nn.Module | Sequence[nn.Module] | None = None,
        decoder: nn.Module | Sequence[nn.Module] | None = None,
        preprocess: nn.Module | None = None,
        dynamics_mode: str = "per_level",
        decoder_mode: str = "per_level",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.K = [int(k) for k in K]
        self.D = int(D)
        self.action_dim = int(action_dim)
        self.preprocess = preprocess
        self.dynamics_mode = str(dynamics_mode)
        self.decoder_mode = str(decoder_mode)
        self.metadata = dict(metadata or {})
        if not self.K:
            raise ValueError("K must contain at least one fidelity level.")
        if any(k <= 0 or k > self.D for k in self.K):
            raise ValueError(f"All K values must be in [1, D={self.D}], got {self.K}.")
        if len(set(self.K)) != len(self.K):
            raise ValueError(f"K values must be unique, got {self.K}.")

        if dynamics is None:
            raise ValueError("MWMWorldModel requires explicit dynamics from a base adapter.")
        if isinstance(dynamics, nn.Module):
            self.dynamics = dynamics
        else:
            self.dynamics = nn.ModuleList(list(dynamics))
            self.dynamics_mode = "per_level"

        if decoder is None:
            self.decoder_mode = "none"
        elif isinstance(decoder, nn.Module):
            self.decoder = decoder
        else:
            self.decoders = nn.ModuleList(list(decoder))
            self.decoder_mode = "per_level"

        self._last_cost_diagnostics: dict[str, Any] = {}

    @property
    def num_levels(self) -> int:
        return len(self.K)

    def config_dict(self) -> dict[str, Any]:
        raise ValueError("Raw MWMWorldModel instances need an adapter/importer `mwm_config` before checkpoint export.")

    def _encoder_output(self, out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            z = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            z = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            z = out.last_hidden_state[:, 0]
        elif isinstance(out, (list, tuple)):
            z = out[0]
        else:
            raise TypeError(f"Unsupported encoder output type {type(out).__name__}")
        if z.ndim > 2:
            z = z.flatten(1)
        if z.shape[-1] < self.D:
            raise ValueError(f"Encoder returned dim {z.shape[-1]} smaller than D={self.D}")
        return z[..., : self.D]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self._encoder_output(self.encoder(x))

    def _pad_prefix(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if int(k) == self.D:
            return z
        out = torch.zeros(*z.shape[:-1], self.D, dtype=z.dtype, device=z.device)
        out[..., : int(k)] = z[..., : int(k)]
        return out

    def _dynamic_step(self, level_idx: int, z_prefix: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dyn = self.dynamics[int(level_idx)] if isinstance(self.dynamics, nn.ModuleList) else self.dynamics
        if hasattr(dyn, "step"):
            return dyn.step(z_prefix, action)
        return dyn(z_prefix, action)

    def predict_next(self, level: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        k = self.K[int(level)]
        if isinstance(self.dynamics, nn.ModuleList):
            return self._dynamic_step(int(level), z[..., :k], a)
        z_in = self._pad_prefix(z, k)
        z_next = self._dynamic_step(int(level), z_in, a)
        return z_next[..., :k]

    def decode(self, level: int, z: torch.Tensor) -> torch.Tensor:
        if self.decoder_mode == "none":
            raise NotImplementedError("This base adapter did not provide a decoder.")
        k = self.K[int(level)]
        if self.decoder_mode == "per_level":
            return self.decoders[int(level)](z[..., :k])
        return self.decoder(self._pad_prefix(z, k))

    def rollout(self, level: int, z: torch.Tensor, a_seq: torch.Tensor, detach_each_step: bool = False) -> torch.Tensor:
        cur = z
        k = self.K[int(level)]
        rows: list[torch.Tensor] = []
        for t in range(int(a_seq.shape[1])):
            if detach_each_step:
                cur = cur.detach()
            nxt = self.predict_next(int(level), cur, a_seq[:, t])
            cur = self._pad_prefix(nxt, k)
            rows.append(nxt)
        return torch.stack(rows, dim=1)

    def scheduled_rollout(self, z: torch.Tensor, a_seq: torch.Tensor, decision: FidelityDecision) -> torch.Tensor:
        cur = z
        rows: list[torch.Tensor] = []
        for t, level_idx in enumerate(decision.rollout_level_indices):
            k = self.K[int(level_idx)]
            nxt = self.predict_next(int(level_idx), cur, a_seq[:, t])
            cur = self._pad_prefix(nxt, k)
            rows.append(cur)
        return torch.stack(rows, dim=1)

    def _latent_from_infos(self, infos: dict[str, Any], key: str, image_key: str) -> torch.Tensor:
        value = infos.get(key)
        if value is not None:
            return value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        image = infos.get(image_key)
        if image is None:
            raise KeyError(f"Expected {key!r} or {image_key!r} in planning infos.")
        if not torch.is_tensor(image):
            image = torch.as_tensor(image, dtype=torch.float32)
        device = next(self.parameters()).device
        if image.ndim == 6:
            flat = image.reshape(-1, *image.shape[3:])
            if flat.shape[-1] == 3:
                flat = flat.permute(0, 3, 1, 2)
            z = self.encode(flat.to(device))
            return z.reshape(*image.shape[:3], self.D)
        if image.ndim == 5:
            flat = image.reshape(-1, *image.shape[2:])
            if flat.shape[-1] == 3:
                flat = flat.permute(0, 3, 1, 2)
            return self.encode(flat.to(device)).reshape(*image.shape[:2], self.D)
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        return self.encode(image.to(device))

    @torch.no_grad()
    def get_cost_with_fidelity(
        self,
        infos: dict[str, Any],
        candidates: torch.Tensor,
        decision: FidelityDecision,
    ) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError(f"candidates must have shape (B,N,H,A), got {tuple(candidates.shape)}")
        device = candidates.device
        dtype = candidates.dtype
        batch, samples, horizon, action_dim = candidates.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if len(decision.rollout_level_indices) != horizon:
            raise ValueError(
                f"Decision rollout length {len(decision.rollout_level_indices)} does not match horizon {horizon}"
            )
        z0 = self._planning_latents(
            self._latent_from_infos(infos, "z0", "pixels").to(device=device, dtype=dtype),
            batch=batch,
            samples=samples,
            name="z0",
        )
        z_goal = self._planning_latents(
            self._latent_from_infos(infos, "z_goal", "goal").to(device=device, dtype=dtype),
            batch=batch,
            samples=samples,
            name="z_goal",
        )
        flat_actions = candidates.reshape(batch * samples, horizon, action_dim)
        rollout = self.scheduled_rollout(z0, flat_actions, decision)
        terminal = rollout[:, -1]
        k = self.K[int(decision.base_level_idx)]
        costs = torch.linalg.vector_norm(terminal[:, :k] - z_goal[:, :k], dim=-1).reshape(batch, samples)
        self._last_cost_diagnostics = {
            "base_level_idx": int(decision.base_level_idx),
            "rollout_level_indices": [int(x) for x in decision.rollout_level_indices],
            "latent_work": int(batch * samples * sum(self.K[int(x)] for x in decision.rollout_level_indices)),
            "terminal_k": int(k),
        }
        return costs

    def _planning_latents(self, z: torch.Tensor, *, batch: int, samples: int, name: str) -> torch.Tensor:
        if z.shape[-1] != self.D:
            raise ValueError(f"{name} must end with D={self.D}, got {tuple(z.shape)}")
        if z.ndim == 4:
            if z.shape[0] != batch or z.shape[1] != samples:
                raise ValueError(f"{name} expected shape (B,N,T,D), got {tuple(z.shape)}")
            z = z[:, :, -1]
        elif z.ndim == 3:
            if z.shape[0] != batch:
                raise ValueError(f"{name} expected leading B={batch}, got {tuple(z.shape)}")
            if z.shape[1] == samples:
                pass
            else:
                z = z[:, -1].unsqueeze(1).expand(batch, samples, self.D)
        elif z.ndim == 2:
            if z.shape[0] == batch:
                z = z.unsqueeze(1).expand(batch, samples, self.D)
            elif z.shape[0] == batch * samples:
                z = z.reshape(batch, samples, self.D)
            else:
                raise ValueError(f"{name} expected B={batch} or B*N={batch * samples}, got {tuple(z.shape)}")
        else:
            raise ValueError(f"{name} must have shape (B,D), (B,T,D), (B,N,D), or (B,N,T,D), got {tuple(z.shape)}")
        return z.reshape(batch * samples, self.D)


class TransitionPackage(nn.Module):
    """Per-level base transition package used by base-adaptive MWM models."""

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


class MatryoshkaWorldModel(MWMWorldModel):
    """Base-adaptive MWM shell with shared latent production and per-level tails."""

    architecture_version = "base_adaptive_world_model_v1"

    def __init__(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        transitions: Sequence[TransitionPackage],
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
        architecture_version: str | None = None,
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
        arch = str(architecture_version or self.architecture_version)
        meta = dict(metadata or {})
        meta.setdefault("architecture_version", arch)
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
            raise ValueError(f"World model pixels must end with image dimensions, got {tuple(pixels.shape)}")
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
            raise TypeError(f"Unsupported encoder output type {type(out).__name__}")
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
        pred_losses: list[torch.Tensor] = []
        for level_idx in range(self.num_levels):
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
                "Base-adaptive MWM scheduled evaluation operates entirely at the selected K level; "
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
            "prefix_criterion": True,
            "history_size": int(self.history_size),
        }
        return cost


def weighted_level_mean(
    level_losses: Sequence[torch.Tensor],
    *,
    level_weights: Sequence[float] | None = None,
    log_prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate per-level objective terms with the shared MWM weighting rule."""

    losses = list(level_losses)
    if not losses:
        raise ValueError("weighted_level_mean requires at least one level loss.")
    weights = list(level_weights or [1.0] * len(losses))
    if len(weights) != len(losses):
        raise ValueError(f"level_weights has {len(weights)} entries for {len(losses)} levels")
    denom = float(sum(weights)) if sum(weights) else 1.0
    total = losses[0].new_tensor(0.0)
    logs: dict[str, torch.Tensor] = {}
    for level_idx, (loss, weight) in enumerate(zip(losses, weights)):
        logs[f"{log_prefix}_l{level_idx}"] = loss.detach()
        total = total + float(weight) * loss / denom
    return total, logs


def latent_regularizer_loss(
    latents: torch.Tensor,
    *,
    K: Sequence[int],
    regularizer: nn.Module,
    scope: str = "shared_latent",
    level_weights: Sequence[float] | None = None,
    log_prefix: str = "sigreg_loss",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a latent regularizer using MWM's shared/per-level scope policy."""

    if latents.ndim != 3:
        raise ValueError(f"latent_regularizer_loss expects (B,T,D) latents, got {tuple(latents.shape)}")
    if scope == "shared_latent":
        total = regularizer(latents.transpose(0, 1))
        return total, {log_prefix: total.detach()}
    if scope != "per_level_prefix":
        raise ValueError(f"Unknown regularizer scope {scope!r}")
    losses = [regularizer(latents[..., : int(k)].transpose(0, 1)) for k in K]
    total, logs = weighted_level_mean(losses, level_weights=level_weights, log_prefix=log_prefix)
    logs[log_prefix] = total.detach()
    return total, logs


def matryoshka_base_loss(
    level_losses: Sequence[torch.Tensor],
    *,
    latents: torch.Tensor | None = None,
    K: Sequence[int] | None = None,
    level_weights: Sequence[float] | None = None,
    primary_log_prefix: str = "pred_loss",
    primary_aliases: Sequence[str] = (),
    rollout_weight: float = 1.0,
    regularizer: nn.Module | None = None,
    regularizer_weight: float = 0.0,
    regularizer_scope: str = "shared_latent",
    regularizer_log_prefix: str = "sigreg_loss",
) -> dict[str, torch.Tensor]:
    """Aggregate base-provided level losses and optional shared regularization."""

    primary_loss, logs = weighted_level_mean(
        level_losses,
        level_weights=level_weights,
        log_prefix=primary_log_prefix,
    )
    for alias in primary_aliases:
        logs[str(alias)] = primary_loss.detach()
    loss = float(rollout_weight) * primary_loss
    if regularizer is not None and float(regularizer_weight):
        if latents is None:
            raise ValueError("matryoshka_base_loss requires latents when regularizer_weight is non-zero.")
        if K is None:
            raise ValueError("matryoshka_base_loss requires K when regularizer_weight is non-zero.")
        reg_loss, reg_logs = latent_regularizer_loss(
            latents,
            K=K,
            regularizer=regularizer,
            scope=regularizer_scope,
            level_weights=level_weights,
            log_prefix=regularizer_log_prefix,
        )
        loss = loss + float(regularizer_weight) * reg_loss
        logs.update(reg_logs)
    logs["loss"] = loss
    return logs


__all__ = [
    "ImageNetPreprocess",
    "MatryoshkaWorldModel",
    "MWMWorldModel",
    "TransitionPackage",
    "latent_regularizer_loss",
    "matryoshka_base_loss",
    "weighted_level_mean",
]
