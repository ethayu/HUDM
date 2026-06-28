from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from mwm.models.losses import matryoshka_base_loss, weighted_level_mean
from mwm.models.planning_costs import validate_fixed_level_rollout
from mwm.preprocessing.images import image_tensor_to_bchw, maybe_apply_image_preprocess


def _module_device_dtype(module: nn.Module, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters(), None)
    if param is not None:
        return param.device, param.dtype
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device, buffer.dtype
    return fallback.device, fallback.dtype


class PreJEPALevelPredictor(nn.Module):
    """Patch-sequence wrapper around a Stable-WM PreJEPA predictor."""

    def __init__(self, predictor: nn.Module, *, dim: int, num_patches: int) -> None:
        super().__init__()
        self.predictor = predictor
        self.dim = int(dim)
        self.num_patches = int(num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 4:
            raise ValueError(f"PreJEPA predictor expects (..., T, P, D), got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.dim:
            raise ValueError(f"PreJEPA predictor dim={self.dim} received input dim={int(x.shape[-1])}.")
        if int(x.shape[-2]) != self.num_patches:
            raise ValueError(
                f"PreJEPA predictor num_patches={self.num_patches} received {int(x.shape[-2])} patches."
            )
        prefix = tuple(x.shape[:-3])
        time, patches, dim = (int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1]))
        flat = x.reshape(-1, time * patches, dim)
        out = self.predictor(flat)
        if not torch.is_tensor(out):
            raise TypeError(f"PreJEPA predictor returned unsupported type {type(out).__name__}.")
        if out.ndim != 3 or int(out.shape[-1]) != dim:
            raise ValueError(f"PreJEPA predictor returned shape {tuple(out.shape)}, expected (B, T*P, {dim}).")
        if int(out.shape[1]) != time * patches:
            raise ValueError(
                f"PreJEPA predictor returned {int(out.shape[1])} tokens for {time * patches} input tokens."
            )
        return out.reshape(*prefix, time, patches, dim)


class PreJEPARuntimeStrategy(nn.Module):
    """Runtime semantics for Matryoshka PreJEPA/DINO-WM patch latents."""

    def __init__(
        self,
        *,
        extra_encoders: nn.ModuleDict,
        extra_order: Sequence[str],
        extra_dims: dict[str, int],
        extra_input_dims: dict[str, int],
        visual_dim: int,
        num_patches: int,
        action_key: str = "action",
        interpolate_pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.extra_encoders = extra_encoders
        self.extra_order = [str(key) for key in extra_order]
        self.extra_dims = {str(key): int(value) for key, value in extra_dims.items()}
        self.extra_input_dims = {str(key): int(value) for key, value in extra_input_dims.items()}
        self.visual_dim = int(visual_dim)
        self.num_patches = int(num_patches)
        self.action_key = str(action_key)
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)

    @property
    def non_action_extra_order(self) -> list[str]:
        return [key for key in self.extra_order if key != self.action_key]

    def level_dim(self, model: Any, level_idx: int) -> int:
        return int(model.K[int(level_idx)]) + sum(self.extra_dims[key] for key in self.extra_order)

    def _extra_slice(self, model: Any, level_idx: int, key: str) -> tuple[int, int]:
        start = int(model.K[int(level_idx)])
        for candidate in self.extra_order:
            width = self.extra_dims[candidate]
            end = start + width
            if candidate == key:
                return start, end
            start = end
        raise KeyError(f"Unknown PreJEPA extra encoder key {key!r}.")

    def _encode_pixels(
        self,
        model: Any,
        pixels: torch.Tensor,
        *,
        already_preprocessed: bool = False,
    ) -> torch.Tensor:
        if pixels.ndim < 4:
            raise ValueError(f"PreJEPA pixels must end with image dimensions, got {tuple(pixels.shape)}")
        original_shape = tuple(pixels.shape[:-3])
        flat = pixels.reshape(-1, *pixels.shape[-3:])
        flat = image_tensor_to_bchw(flat)
        flat = maybe_apply_image_preprocess(
            flat,
            model.preprocess,
            already_preprocessed=already_preprocessed,
        )
        device, dtype = _module_device_dtype(model.encoder, flat)
        flat = flat.to(device=device, dtype=dtype)
        kwargs = {"interpolate_pos_encoding": True} if self.interpolate_pos_encoding else {}
        try:
            out = model.encoder(flat, **kwargs)
        except TypeError:
            out = model.encoder(flat)
        if not hasattr(out, "last_hidden_state"):
            raise ValueError("PreJEPA/DINO-WM v1 supports image transformer backbones with last_hidden_state only.")
        tokens = out.last_hidden_state
        if tokens.ndim != 3:
            raise ValueError(f"PreJEPA image backbone returned tokens with shape {tuple(tokens.shape)}.")
        if int(tokens.shape[1]) == self.num_patches + 1:
            tokens = tokens[:, 1:, :]
        elif int(tokens.shape[1]) != self.num_patches:
            raise ValueError(
                f"PreJEPA image backbone returned {int(tokens.shape[1])} tokens; expected "
                f"{self.num_patches} patches or {self.num_patches + 1} tokens including cls."
            )
        if int(tokens.shape[-1]) != self.visual_dim:
            raise ValueError(
                f"PreJEPA image backbone returned visual dim {int(tokens.shape[-1])}; "
                f"expected D_visual={self.visual_dim}."
            )
        return tokens.detach().reshape(*original_shape, self.num_patches, self.visual_dim)

    def _encode_extra(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 2:
            raise ValueError(f"PreJEPA extra input {key!r} must end with (T, D), got {tuple(value.shape)}")
        expected_input_dim = self.extra_input_dims[key]
        if int(value.shape[-1]) != expected_input_dim:
            raise ValueError(
                f"PreJEPA extra input {key!r} expected dim={expected_input_dim}, got {int(value.shape[-1])}."
            )
        prefix = tuple(value.shape[:-2])
        time = int(value.shape[-2])
        flat = value.reshape(-1, time, int(value.shape[-1])).float()
        encoder = self.extra_encoders[key]
        device, dtype = _module_device_dtype(encoder, flat)
        out = encoder(flat.to(device=device, dtype=dtype))
        if not torch.is_tensor(out):
            raise TypeError(f"PreJEPA extra encoder {key!r} returned unsupported type {type(out).__name__}.")
        if out.ndim != 3 or int(out.shape[1]) != time:
            raise ValueError(
                f"PreJEPA extra encoder {key!r} returned shape {tuple(out.shape)}; expected (B, {time}, D)."
            )
        expected_dim = self.extra_dims[key]
        if int(out.shape[-1]) != expected_dim:
            raise ValueError(
                f"PreJEPA extra encoder {key!r} returned dim={int(out.shape[-1])}; expected {expected_dim}."
            )
        return out.reshape(*prefix, time, expected_dim)

    def encode(
        self,
        model: Any,
        info: dict[str, torch.Tensor] | torch.Tensor,
        *,
        already_preprocessed: bool = False,
        pixels_key: str = "pixels",
        target: str = "emb",
        prefix: str = "",
        emb_keys: Sequence[str] | None = None,
    ) -> Any:
        if torch.is_tensor(info):
            return self._encode_pixels(model, info, already_preprocessed=already_preprocessed)
        out = dict(info)
        keys = list(self.extra_order if emb_keys is None else emb_keys)
        pixels_embed = self._encode_pixels(model, out[pixels_key], already_preprocessed=already_preprocessed)
        out[f"pixels_{target}"] = pixels_embed
        embedding = pixels_embed
        for key in keys:
            source_key = f"{prefix}{key}"
            if source_key not in out:
                raise KeyError(f"PreJEPA encode expected extra input {source_key!r}.")
            extra_embed = self._encode_extra(key, out[source_key])
            out[f"{key}_{target}"] = extra_embed
            extra_tiled = extra_embed.unsqueeze(-2).expand(*extra_embed.shape[:-1], self.num_patches, extra_embed.shape[-1])
            embedding = torch.cat([embedding, extra_tiled], dim=-1)
        out[target] = embedding
        return out

    def compose_level_embedding(self, model: Any, encoded: dict[str, torch.Tensor], level_idx: int) -> torch.Tensor:
        k = int(model.K[int(level_idx)])
        pixels = encoded["pixels_emb"][..., :k]
        pieces = [pixels]
        for key in self.extra_order:
            extra = encoded[f"{key}_emb"]
            extra_tiled = extra.unsqueeze(-2).expand(*extra.shape[:-1], self.num_patches, extra.shape[-1])
            pieces.append(extra_tiled)
        level = torch.cat(pieces, dim=-1)
        expected_dim = self.level_dim(model, int(level_idx))
        if int(level.shape[-1]) != expected_dim:
            raise ValueError(f"PreJEPA level {level_idx} produced dim {int(level.shape[-1])}; expected {expected_dim}.")
        return level

    def predict_prefix(self, model: Any, level_idx: int, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        if emb.ndim < 4:
            raise ValueError("PreJEPA prediction expects patch latents shaped (..., T, P, D).")
        return model.transitions[int(level_idx)](emb)

    def decode(self, model: Any, level_idx: int, latent: torch.Tensor) -> torch.Tensor:
        del model, level_idx, latent
        raise NotImplementedError("PreJEPA/DINO-WM v1 exposes latent patch prediction, not image decoding.")

    def _non_action_parts(
        self,
        model: Any,
        level_idx: int,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        k = int(model.K[int(level_idx)])
        pred_parts = [pred[..., :k]]
        target_parts = [target[..., :k]]
        for key in self.non_action_extra_order:
            start, end = self._extra_slice(model, int(level_idx), key)
            pred_parts.append(pred[..., start:end])
            target_parts.append(target[..., start:end])
        return pred_parts, target_parts

    def training_loss(
        self,
        model: Any,
        batch: dict[str, torch.Tensor],
        *,
        level_weights: Sequence[float] | None = None,
        rollout_weight: float = 1.0,
        recon_latent_weight: float = 0.0,
        sigreg: nn.Module | None = None,
        sigreg_weight: float = 0.0,
        sigreg_scope: str = "shared_latent",
    ) -> dict[str, torch.Tensor]:
        del sigreg_scope
        if float(recon_latent_weight):
            raise ValueError("PreJEPA/DINO-WM v1 does not support decoder reconstruction losses.")
        if sigreg is not None and float(sigreg_weight):
            raise ValueError("PreJEPA/DINO-WM v1 does not support SIGReg regularization through this adapter.")
        work = dict(batch)
        work[self.action_key] = torch.nan_to_num(work[self.action_key], 0.0)
        encoded = self.encode(model, work, already_preprocessed=True)
        pred_losses: list[torch.Tensor] = []
        pixel_losses: list[torch.Tensor] = []
        extra_losses: dict[str, list[torch.Tensor]] = {key: [] for key in self.non_action_extra_order}

        for level_idx in range(model.num_levels):
            level_emb = self.compose_level_embedding(model, encoded, level_idx)
            history = level_emb[:, : model.history_size]
            pred = model.transitions[level_idx](history)
            target_start = int(model.num_preds)
            target_end = target_start + int(pred.shape[1])
            if target_end > int(level_emb.shape[1]):
                raise ValueError(
                    f"PreJEPA target window [{target_start}:{target_end}] exceeds sequence length "
                    f"{int(level_emb.shape[1])}."
                )
            target = level_emb[:, target_start:target_end]
            pred_parts, target_parts = self._non_action_parts(model, level_idx, pred, target)
            pred_losses.append((torch.cat(pred_parts, dim=-1) - torch.cat(target_parts, dim=-1)).pow(2).mean())
            k = int(model.K[level_idx])
            pixel_losses.append((pred[..., :k] - target[..., :k]).pow(2).mean())
            for key in self.non_action_extra_order:
                start, end = self._extra_slice(model, level_idx, key)
                extra_losses[key].append((pred[..., start:end] - target[..., start:end]).pow(2).mean())

        logs = matryoshka_base_loss(
            pred_losses,
            K=model.K,
            level_weights=level_weights,
            primary_log_prefix="pred_loss",
            primary_aliases=("pred_loss", "rollout_loss"),
            rollout_weight=rollout_weight,
        )
        pixel_loss, pixel_logs = weighted_level_mean(
            pixel_losses,
            level_weights=level_weights,
            log_prefix="pixels_loss",
        )
        logs["pixels_loss"] = pixel_loss.detach()
        logs.update(pixel_logs)
        for key, losses in extra_losses.items():
            extra_loss, extra_logs = weighted_level_mean(
                losses,
                level_weights=level_weights,
                log_prefix=f"{key}_loss",
            )
            logs[f"{key}_loss"] = extra_loss.detach()
            logs.update(extra_logs)
        return logs

    def _expand_samples(self, value: torch.Tensor, samples: int) -> torch.Tensor:
        return value.unsqueeze(1).expand(*value.shape[:1], int(samples), *value.shape[1:])

    def _sample_zero(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3:
            raise ValueError(f"Planning tensors must include batch, samples, and time dimensions, got {tuple(value.shape)}")
        return value[:, 0]

    def _replace_extra(
        self,
        model: Any,
        level_idx: int,
        embedding: torch.Tensor,
        key: str,
        extra_value: torch.Tensor,
    ) -> torch.Tensor:
        start, end = self._extra_slice(model, level_idx, key)
        tiled = extra_value.unsqueeze(-2).expand(*extra_value.shape[:-1], self.num_patches, extra_value.shape[-1])
        out = embedding.clone()
        out[..., start:end] = tiled
        return out

    def _store_predicted_parts(self, model: Any, infos: dict[str, Any], level_idx: int, embedding: torch.Tensor) -> None:
        k = int(model.K[int(level_idx)])
        infos["predicted_embedding"] = embedding
        infos["predicted_emb"] = embedding
        infos["predicted_pixels_emb"] = embedding[..., :k]
        for key in self.extra_order:
            start, end = self._extra_slice(model, int(level_idx), key)
            infos[f"predicted_{key}_emb"] = embedding[..., start:end]

    def rollout_at_level(self, model: Any, infos: dict[str, Any], action_sequence: torch.Tensor, level_idx: int) -> dict[str, Any]:
        if "pixels" not in infos:
            raise KeyError("pixels not in info_dict")
        if action_sequence.ndim != 4:
            raise ValueError(f"action_sequence must have shape (B,N,H,A), got {tuple(action_sequence.shape)}")
        batch, samples, horizon = (int(action_sequence.shape[0]), int(action_sequence.shape[1]), int(action_sequence.shape[2]))
        if int(action_sequence.shape[-1]) != model.action_dim:
            raise ValueError(f"Expected action_dim={model.action_dim}, got {int(action_sequence.shape[-1])}")
        history = int(infos["pixels"].shape[2])
        if horizon < history:
            raise ValueError(f"Action horizon {horizon} is shorter than pixel history {history}.")
        n_steps = horizon - history

        encoded: dict[str, torch.Tensor] = {
            "pixels_emb": self._encode_pixels(model, self._sample_zero(infos["pixels"]), already_preprocessed=False)
        }
        for key in self.non_action_extra_order:
            if key not in infos:
                raise KeyError(f"PreJEPA rollout requires non-action extra input {key!r}.")
            encoded[f"{key}_emb"] = self._encode_extra(key, self._sample_zero(infos[key]))

        actions = action_sequence.reshape(batch * samples, horizon, model.action_dim)
        action_emb = self._encode_extra(self.action_key, actions).reshape(
            batch,
            samples,
            horizon,
            self.extra_dims[self.action_key],
        )
        expanded: dict[str, torch.Tensor] = {
            "pixels_emb": self._expand_samples(encoded["pixels_emb"], samples),
            f"{self.action_key}_emb": action_emb[:, :, :history],
        }
        for key in self.non_action_extra_order:
            expanded[f"{key}_emb"] = self._expand_samples(encoded[f"{key}_emb"], samples)

        level_init = self.compose_level_embedding(model, expanded, level_idx).reshape(
            batch * samples,
            history,
            self.num_patches,
            self.level_dim(model, level_idx),
        )
        action_emb_flat = action_emb.reshape(batch * samples, horizon, self.extra_dims[self.action_key])
        emb_list = list(level_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, history + t - model.history_size)
            context = torch.stack(emb_list[lo:], dim=1)
            pred = model.transitions[int(level_idx)](context)[:, -1]
            pred_time = history + t
            if pred_time < horizon:
                pred = self._replace_extra(
                    model,
                    int(level_idx),
                    pred,
                    self.action_key,
                    action_emb_flat[:, pred_time],
                )
            emb_list.append(pred)
        predicted = torch.stack(emb_list, dim=1).reshape(
            batch,
            samples,
            history + n_steps + 1,
            self.num_patches,
            self.level_dim(model, level_idx),
        )
        self._store_predicted_parts(model, infos, int(level_idx), predicted)
        return infos

    def ensure_goal_emb(self, model: Any, infos: dict[str, Any]) -> None:
        if "pixels_goal_emb" in infos:
            return
        if "goal" not in infos:
            raise KeyError("goal not in info_dict")
        samples = int(infos["goal"].shape[1])
        goal_pixels = self._sample_zero(infos["goal"])
        pixels_goal = self._encode_pixels(model, goal_pixels, already_preprocessed=False)
        infos["pixels_goal_emb"] = self._expand_samples(pixels_goal, samples)
        for key in self.non_action_extra_order:
            goal_key = f"goal_{key}"
            if goal_key not in infos:
                raise KeyError(f"PreJEPA planning requires goal extra input {goal_key!r}.")
            goal_extra = self._encode_extra(key, self._sample_zero(infos[goal_key]))
            goal_extra = self._expand_samples(goal_extra, samples)
            infos[f"{key}_goal_emb"] = goal_extra.unsqueeze(-2).expand(
                *goal_extra.shape[:-1],
                self.num_patches,
                goal_extra.shape[-1],
            )

    @torch.no_grad()
    def get_cost_with_fidelity(self, model: Any, infos: dict[str, Any], candidates: torch.Tensor, decision: Any) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError(f"candidates must have shape (B,N,H,A), got {tuple(candidates.shape)}")
        level_idx, rollout_levels = validate_fixed_level_rollout(decision, int(candidates.shape[2]))
        if int(candidates.shape[-1]) != model.action_dim:
            raise ValueError(f"Expected action_dim={model.action_dim}, got {int(candidates.shape[-1])}")
        self.ensure_goal_emb(model, infos)
        out = self.rollout_at_level(model, infos, candidates, int(level_idx))
        k = int(model.K[int(level_idx)])
        pred_pixels = out["predicted_pixels_emb"][..., -1, :, :k]
        goal_pixels = out["pixels_goal_emb"][..., -1, :, :k].expand_as(pred_pixels)
        cost = (pred_pixels - goal_pixels.detach()).pow(2).sum(dim=(-2, -1))
        for key in self.non_action_extra_order:
            pred_extra = out[f"predicted_{key}_emb"][..., -1, :, :]
            goal_extra = out[f"{key}_goal_emb"][..., -1, :, :].expand_as(pred_extra)
            cost = cost + (pred_extra - goal_extra.detach()).pow(2).sum(dim=(-2, -1))
        model._last_cost_diagnostics = {
            "base_level_idx": int(level_idx),
            "rollout_level_indices": rollout_levels,
            "latent_work": int(candidates.shape[0] * candidates.shape[1] * candidates.shape[2] * self.level_dim(model, level_idx) * self.num_patches),
            "terminal_k": int(k),
            "level_dim": int(self.level_dim(model, int(level_idx))),
            "num_patches": int(self.num_patches),
            "prefix_criterion": True,
            "history_size": int(model.history_size),
        }
        return cost


__all__ = [
    "PreJEPALevelPredictor",
    "PreJEPARuntimeStrategy",
]
