from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mwm.fidelity import FidelityDecision


@dataclass(frozen=True)
class MWMActionSpec:
    dim: int
    low: list[float] | None = None
    high: list[float] | None = None


@dataclass(frozen=True)
class MWMComponentSpec:
    target: str
    kwargs: dict[str, Any]


class _DefaultDynamics(nn.Module):
    def __init__(self, k_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(k_dim) + int(action_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(k_dim)),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))


class _DefaultImageDecoder(nn.Module):
    def __init__(self, k_dim: int, image_shape: tuple[int, int], hidden_channels: int = 128) -> None:
        super().__init__()
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.fc = nn.Sequential(nn.Linear(int(k_dim), int(hidden_channels) * 7 * 7), nn.GELU())
        self.conv = nn.Sequential(
            nn.Conv2d(int(hidden_channels), 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).reshape(z.shape[0], -1, 7, 7)
        x = F.interpolate(x, size=self.image_shape, mode="bilinear", align_corners=False)
        return self.conv(x)


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
            self.dynamics = nn.ModuleList([_DefaultDynamics(k, self.action_dim) for k in self.K])
            self.dynamics_mode = "per_level"
        elif isinstance(dynamics, nn.Module):
            self.dynamics = dynamics
        else:
            self.dynamics = nn.ModuleList(list(dynamics))
            self.dynamics_mode = "per_level"

        image_shape = tuple(int(x) for x in self.metadata.get("image_shape", (96, 96)))
        if decoder is None:
            self.decoders = nn.ModuleList([_DefaultImageDecoder(k, image_shape) for k in self.K])
            self.decoder_mode = "per_level"
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


def _sigreg_loss(proj: torch.Tensor, *, knots: int = 17, num_proj: int = 1024) -> torch.Tensor:
    """CPU/GPU-safe Sketch Isotropic Gaussian regularizer on full latents."""

    if proj.ndim != 3:
        raise ValueError(f"SIGReg expects (T,B,D), got {tuple(proj.shape)}")
    t = torch.linspace(0, 3, int(knots), dtype=proj.dtype, device=proj.device)
    dt = 3 / max(1, int(knots) - 1)
    weights = torch.full((int(knots),), 2 * dt, dtype=proj.dtype, device=proj.device)
    weights[[0, -1]] = dt
    window = torch.exp(-t.square() / 2.0)
    weights = weights * window
    basis = torch.randn(proj.size(-1), int(num_proj), dtype=proj.dtype, device=proj.device)
    basis = basis / basis.norm(p=2, dim=0).clamp_min(1e-12)
    x_t = (proj @ basis).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - window).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def mwm_prediction_loss(
    model: MWMWorldModel,
    batch: dict[str, torch.Tensor],
    *,
    level: int | None = None,
    level_weights: Sequence[float] | None = None,
    recon_weight: float = 0.0,
    rollout_weight: float = 1.0,
    sigreg_weight: float = 0.0,
    sigreg_knots: int = 17,
    sigreg_num_proj: int = 1024,
) -> dict[str, torch.Tensor]:
    if getattr(model, "eval_only", False):
        raise RuntimeError("Eval-only imported checkpoints cannot be used for MWM training.")
    x = batch["x"]
    actions = batch["a"]
    if x.ndim != 5:
        raise ValueError(f"batch['x'] must have shape (B,T,C,H,W), got {tuple(x.shape)}")
    if actions.ndim < 3:
        actions = actions.reshape(actions.shape[0], actions.shape[1], -1)
    expected_actions = int(x.shape[1]) - 1
    if int(actions.shape[1]) < expected_actions:
        raise ValueError(f"Expected at least {expected_actions} action steps for {x.shape[1]} frames, got {actions.shape[1]}")
    if int(actions.shape[1]) > expected_actions:
        actions = actions[:, :expected_actions]
    z = model.encode(x.reshape(-1, *x.shape[2:])).reshape(x.shape[0], x.shape[1], model.D)
    levels = [int(level)] if level is not None else list(range(model.num_levels))
    weights = list(level_weights or [1.0] * len(levels))
    if len(weights) != len(levels):
        raise ValueError(f"level_weights has {len(weights)} entries for {len(levels)} levels")
    denom = float(sum(weights)) if sum(weights) else 1.0
    loss = z.new_tensor(0.0)
    logs: dict[str, torch.Tensor] = {}
    for level_idx, weight in zip(levels, weights):
        pred = model.rollout(level_idx, z[:, 0], actions)
        target = z[:, 1 : 1 + pred.shape[1], : model.K[level_idx]].detach()
        level_loss = F.mse_loss(pred, target)
        logs[f"rollout_loss_l{level_idx}"] = level_loss.detach()
        loss = loss + float(weight) * level_loss / denom
    rollout_loss = loss
    loss = float(rollout_weight) * rollout_loss
    logs.update({"loss": loss, "rollout_loss": rollout_loss})
    if float(sigreg_weight):
        sigreg = _sigreg_loss(z.transpose(0, 1), knots=int(sigreg_knots), num_proj=int(sigreg_num_proj))
        loss = loss + float(sigreg_weight) * sigreg
        logs["loss"] = loss
        logs["sigreg_loss"] = sigreg.detach()
    if recon_weight:
        level_idx = levels[-1]
        recon = model.decode(level_idx, z[:, 0])
        recon_loss = F.mse_loss(recon, x[:, 0])
        loss = loss + float(recon_weight) * recon_loss
        logs["loss"] = loss
        logs["recon_loss"] = recon_loss
    return logs


__all__ = [
    "MWMActionSpec",
    "MWMComponentSpec",
    "MWMWorldModel",
    "mwm_prediction_loss",
]
