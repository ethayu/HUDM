from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.fidelity import FidelityDecision


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
        if max(self.K) != self.D:
            raise ValueError(f"Largest K must equal D={self.D}, got {self.K}")

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
    "MWMWorldModel",
    "latent_regularizer_loss",
    "matryoshka_base_loss",
    "weighted_level_mean",
]
